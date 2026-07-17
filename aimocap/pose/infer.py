"""2D whole-body pose inference via CIGPose (ONNX Runtime, GPU).

Wraps cigpose's low-level API so we get raw keypoint data (not frame drawings).
The high-level ``cigpose.infer_persons`` draws onto the frame — useless for a
data pipeline — so we drive the pieces ourselves:

    frame -> YOLOX.detect -> [bbox]
           -> preprocess_person -> pose_session.run -> decode_simcc
           -> remap_to_frame  ->  Pose2D(keypoints, scores, bbox)

A ``PoseEstimator`` holds the loaded ONNX sessions (detector + pose model) and
the cigpose bookkeeping (input_w/h, split_ratio). Construct once, call
``.estimate(frame)`` per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# cigpose imports onnxruntime; importing aimocap first ensures CUDA DLLs load.
import aimocap  # noqa: F401
from cigpose import (
    YOLOXDetector,
    decode_simcc,
    load_pose_model,
    preprocess_person,
)
from aimocap.pose.transforms import get_crop_affine_transform, apply_affine_transform

from aimocap.pose.keypoints import N_KEYPOINTS

# Default model + detector paths (relative to project root).
_DEFAULT_POSE_MODEL = "models/cigpose-x_coco-wholebody_384x288.onnx"
_DEFAULT_DETECTOR = "models/yolox_nano.onnx"


@dataclass(slots=True)
class Pose2D:
    """One person's 2D pose in a single frame.

    keypoints: (133, 2) float32, pixel coords in the original frame.
    scores:    (133,)   float32, per-keypoint confidence in [0, ~1].
    bbox:      (4,)     float32, [x1, y1, x2, y2] person box in pixels.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    bbox: np.ndarray

    def __post_init__(self) -> None:
        if self.keypoints.shape != (N_KEYPOINTS, 2):
            raise ValueError(
                f"keypoints shape {self.keypoints.shape}, expected ({N_KEYPOINTS}, 2)"
            )
        if self.scores.shape != (N_KEYPOINTS,):
            raise ValueError(
                f"scores shape {self.scores.shape}, expected ({N_KEYPOINTS},)"
            )
        if self.bbox.shape != (4,):
            raise ValueError(f"bbox shape {self.bbox.shape}, expected (4,)")

    @property
    def area(self) -> float:
        """Bounding-box area in pixels^2 (used to pick the largest person)."""
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    def mean_score(self, mask: np.ndarray | slice | None = None) -> float:
        """Mean confidence over the given keypoint subset (default: all)."""
        s = self.scores if mask is None else self.scores[mask]
        return float(s.mean()) if s.size else 0.0


class PoseEstimator:
    """Loads CIGPose detector + pose model and runs per-frame inference.

    Parameters
    ----------
    pose_model : path to the whole-body ONNX pose model.
    detector   : path to the YOLOX-Nano ONNX detector (None = no detection;
                 the full frame is treated as one person — useful for pre-cropped
                 single-subject inputs).
    providers  : ONNX Runtime provider list. Defaults to CUDA-first.
    det_threshold / nms_threshold : YOLOX detection tuning.
    """

    def __init__(
        self,
        pose_model: str | Path = _DEFAULT_POSE_MODEL,
        detector: str | Path | None = _DEFAULT_DETECTOR,
        providers: list[str] | None = None,
        det_threshold: float = 0.3,
        nms_threshold: float = 0.45,
    ) -> None:
        self.providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        pose_path = str(pose_model)
        if not Path(pose_path).exists():
            raise FileNotFoundError(f"pose model not found: {pose_path}")

        self.pose_session, self.input_w, self.input_h, self.split_ratio = (
            load_pose_model(pose_path, providers=self.providers)
        )

        if detector is None:
            self.detector = None
        else:
            det_path = str(detector)
            if not Path(det_path).exists():
                raise FileNotFoundError(f"detector not found: {det_path}")
            self.detector = YOLOXDetector(
                det_path,
                conf_thresh=det_threshold,
                nms_thresh=nms_threshold,
                providers=self.providers,
            )

        # Surface which provider actually got used (for logging).
        self.active_providers = self.pose_session.get_providers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray, pick: str = "largest") -> list[Pose2D]:
        """Estimate poses for all detected persons in ``frame``.

        ``pick`` only affects the return when a detector is configured:
          - "all"      : return every detected person
          - "largest"  : return only the largest-area person (list of len 1)
          - "best"     : return only the highest mean-confidence person
        Without a detector, returns a single Pose2D for the whole frame.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be HxWx3 BGR, got {frame.shape}")

        bboxes = self._detect(frame)
        poses = [self._estimate_one(frame, np.asarray(b, dtype=np.float32))
                 for b in bboxes]

        if pick == "all" or not poses:
            return poses
        if pick == "largest":
            return [max(poses, key=lambda p: p.area)]
        if pick == "best":
            return [max(poses, key=lambda p: p.mean_score())]
        raise ValueError(f"unknown pick={pick!r}; use 'all'|'largest'|'best'")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> list[list[float]]:
        if self.detector is None:
            h, w = frame.shape[:2]
            return [[0.0, 0.0, float(w), float(h)]]
        return self.detector.detect(frame)

    def _estimate_one(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Pose2D:
        tensor, crop_region = preprocess_person(
            frame, bbox.tolist(), self.input_w, self.input_h
        )
        simcc_x, simcc_y = self.pose_session.run(None, {"input": tensor})
        kpts, scores = decode_simcc(
            simcc_x, simcc_y, self.input_w, self.input_h, self.split_ratio
        )
        
        # Rigorous geometric un-crop using explicit affine inverse
        h, w = frame.shape[:2]
        M, M_inv, _ = get_crop_affine_transform(bbox, self.input_w, self.input_h, w, h)
        kpts = apply_affine_transform(kpts, M_inv)
        
        return Pose2D(
            keypoints=kpts.astype(np.float32),
            scores=scores.astype(np.float32),
            bbox=bbox,
        )


__all__ = ["Pose2D", "PoseEstimator"]
