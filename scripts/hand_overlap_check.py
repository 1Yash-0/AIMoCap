"""Overlap analysis: which frames trigger hand-outside-bbox per camera?

Runs the detector only (no pose model) on the same 300-frame window as Stage 3,
loads keypoints from the Stage 3 NPZ, then for each camera records the exact
frame indices where a hand/wrist keypoint falls outside the YOLO box.

Outputs:
  - outputs/stage3_pose/hand_overlap.json  -- per-camera frame index lists + overlap stats
  - printed Venn summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aimocap  # noqa: F401
from cigpose import YOLOXDetector
import cv2

from aimocap.pose.keypoints import LEFT_WRIST, RIGHT_WRIST, LEFT_HAND_WRIST, RIGHT_HAND_WRIST

# ── Config (must match stage3_pose_audit.py exactly) ─────────────────────────
DETECTOR    = ROOT / "models" / "yolox_nano.onnx"
VIDEO_PATHS = [
    ROOT / "data" / "panoptic" / "171204_pose1" / "hdVideos" / "hd_00_00.mp4",
    ROOT / "data" / "panoptic" / "171204_pose1" / "hdVideos" / "hd_00_01.mp4",
    ROOT / "data" / "panoptic" / "171204_pose1" / "hdVideos" / "hd_00_02.mp4",
]
CAMERA_IDS  = ["hd_00_00", "hd_00_01", "hd_00_02"]
NPZ_PATH    = ROOT / "outputs" / "stage3_pose" / "kpts.npz"
OUT_DIR     = ROOT / "outputs" / "stage3_pose"
CONF_THRESH = 0.3
HIGH_CONF   = 0.65
START_SECOND = 5.0
MAX_FRAMES   = 300
PROVIDERS    = ["CUDAExecutionProvider", "CPUExecutionProvider"]

HAND_KPT_INDICES = [LEFT_WRIST, RIGHT_WRIST, LEFT_HAND_WRIST, RIGHT_HAND_WRIST]
HAND_KPT_NAMES   = ["left_wrist(9)", "right_wrist(10)", "left_hand_wrist(91)", "right_hand_wrist(112)"]


def detect_largest(detector: YOLOXDetector, frame: np.ndarray) -> list[float] | None:
    """Return [x1,y1,x2,y2] of the largest person box, or None."""
    blob, ratio = detector._letterbox(frame)
    raw = detector.session.run(None, {detector.input_name: blob})[0][0]
    preds = detector._decode(raw)
    scores = preds[:, 4] * preds[:, 5]
    keep = scores >= detector.conf_thresh
    if not np.any(keep):
        return None

    boxes_f = preds[keep, :4]
    scores_f = scores[keep]
    x1 = boxes_f[:, 0] - boxes_f[:, 2] / 2
    y1 = boxes_f[:, 1] - boxes_f[:, 3] / 2
    x2 = boxes_f[:, 0] + boxes_f[:, 2] / 2
    y2 = boxes_f[:, 1] + boxes_f[:, 3] / 2

    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)
    indices = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), scores_f.tolist(), detector.conf_thresh, 0.45
    )
    if len(indices) == 0:
        return None
    indices = indices.flatten()

    # pick largest area
    areas = [(x2[i] - x1[i]) * (y2[i] - y1[i]) for i in indices]
    best = indices[int(np.argmax(areas))]
    return [x1[best] / ratio, y1[best] / ratio, x2[best] / ratio, y2[best] / ratio]


def kpt_outside(kpt_xy: np.ndarray, bbox: list[float]) -> bool:
    x, y = float(kpt_xy[0]), float(kpt_xy[1])
    x1, y1, x2, y2 = bbox
    return not (x1 <= x <= x2 and y1 <= y <= y2)


def main() -> None:
    # Load Stage 3 keypoints/scores
    data = np.load(NPZ_PATH, allow_pickle=True)
    kpts   = data["keypoints"]   # (F, C, 133, 2)
    scores = data["scores"]      # (F, C, 133)
    cam_names = list(data["camera_names"])
    print(f"Loaded NPZ: {kpts.shape} keypoints, cameras={cam_names}")

    # Confirm camera order matches our config
    for expected, actual in zip(CAMERA_IDS, cam_names):
        if expected != actual:
            print(f"WARNING: camera order mismatch: expected {expected}, got {actual}")

    detector = YOLOXDetector(
        str(DETECTOR), conf_thresh=CONF_THRESH, nms_thresh=0.45, providers=PROVIDERS
    )
    print("Detector loaded.\n")

    # Per-camera: collect frame indices where a high-conf hand kpt is outside bbox
    cam_outside_frames: dict[str, set[int]] = {}

    for c_idx, (cam_id, video_path) in enumerate(zip(CAMERA_IDS, VIDEO_PATHS)):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * START_SECOND))

        outside_frames: set[int] = set()

        for f in range(MAX_FRAMES):
            ok, frame = cap.read()
            if not ok:
                break

            bbox = detect_largest(detector, frame)
            if bbox is None:
                continue

            # Check each hand keypoint for THIS camera, THIS frame
            for hi in HAND_KPT_INDICES:
                kpt_score = float(scores[f, c_idx, hi])
                if kpt_score < HIGH_CONF:
                    continue  # only count high-confidence keypoints
                kpt_xy = kpts[f, c_idx, hi]
                if not np.isfinite(kpt_xy).all():
                    continue
                if kpt_outside(kpt_xy, bbox):
                    outside_frames.add(f)
                    break  # one miss per frame is enough

        cap.release()
        cam_outside_frames[cam_id] = outside_frames
        print(f"[{cam_id}] {len(outside_frames)} / {MAX_FRAMES} frames with hand outside bbox")

    # Set operations
    sets = [cam_outside_frames[cid] for cid in CAMERA_IDS]
    s0, s1, s2 = sets

    all_three   = s0 & s1 & s2
    c0_c1_only  = (s0 & s1) - s2
    c0_c2_only  = (s0 & s2) - s1
    c1_c2_only  = (s1 & s2) - s0
    only_c0     = s0 - s1 - s2
    only_c1     = s1 - s0 - s2
    only_c2     = s2 - s0 - s1

    union = s0 | s1 | s2

    print(f"\n-- Hand-outside-bbox Frame Overlap --")
    print(f"Total unique frames affected (any camera): {len(union)}")
    print(f"All 3 cameras miss simultaneously:         {len(all_three)}  ({100*len(all_three)/MAX_FRAMES:.1f}%)")
    print(f"Cam00+Cam01 only (cam02 still sees it):    {len(c0_c1_only)}")
    print(f"Cam00+Cam02 only (cam01 still sees it):    {len(c0_c2_only)}")
    print(f"Cam01+Cam02 only (cam00 still sees it):    {len(c1_c2_only)}")
    print(f"Only cam00 misses:                         {len(only_c0)}")
    print(f"Only cam01 misses:                         {len(only_c1)}")
    print(f"Only cam02 misses:                         {len(only_c2)}")

    print(f"\nConclusion:")
    if len(all_three) == 0:
        print("  No frames where ALL three cameras lose the hand simultaneously.")
        print("  -> Triangulation's confidence weighting has full redundancy. No fix needed.")
    elif len(all_three) / MAX_FRAMES < 0.03:
        print(f"  Only {len(all_three)} frames ({100*len(all_three)/MAX_FRAMES:.1f}%) hit all three cameras.")
        print("  -> Marginal. Triangulation redundancy covers it in practice.")
    else:
        print(f"  {len(all_three)} frames ({100*len(all_three)/MAX_FRAMES:.1f}%) are missed by ALL cameras.")
        print("  -> Real data loss. Conditional padding or wrist-crop warranted.")

    # Save
    result = {
        "window": f"frames 0-{MAX_FRAMES} starting at t={START_SECOND}s",
        "high_conf_threshold": HIGH_CONF,
        "per_camera_outside_count": {cid: len(cam_outside_frames[cid]) for cid in CAMERA_IDS},
        "per_camera_outside_frames": {cid: sorted(cam_outside_frames[cid]) for cid in CAMERA_IDS},
        "overlap": {
            "all_three": sorted(all_three),
            "all_three_count": len(all_three),
            "all_three_pct": round(100 * len(all_three) / MAX_FRAMES, 2),
            "only_cam00": sorted(only_c0),
            "only_cam01": sorted(only_c1),
            "only_cam02": sorted(only_c2),
            "cam00_cam01_both": sorted(c0_c1_only),
            "cam00_cam02_both": sorted(c0_c2_only),
            "cam01_cam02_both": sorted(c1_c2_only),
            "union_count": len(union),
        },
    }
    out_path = OUT_DIR / "hand_overlap.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nFull frame lists saved: {out_path}")


if __name__ == "__main__":
    main()
