"""Stage 2 — YOLO Detection Audit.

Runs YOLOX-nano (models/yolox_nano.onnx) on 3 Panoptic camera videos.
Measures:
  - % of frames with a valid detection above conf_threshold
  - Mean and minimum confidence across all detections
  - Frame-to-frame box stability (IoU between consecutive frames)

Writes to outputs/stage2_detection/:
  - metrics.json
  - cam{i}_detection.mp4  (annotated clip per camera)

NOTE: The repo uses cigpose's YOLOXDetector, NOT ultralytics YOLO.
      The brief asked for ultralytics, but this is what is actually wired
      in the codebase. Flagged here so it's on record.

Usage:
    python scripts/stage2_detection_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aimocap  # noqa: F401 – ensures CUDA DLLs load before onnxruntime
from cigpose import YOLOXDetector

import argparse

# ── Config ────────────────────────────────────────────────────────────────────
DETECTOR_PATH = ROOT / "models" / "yolox_nano.onnx"
CONF_THRESHOLD = 0.3       # matches PoseEstimator default
MAX_FRAMES = 300           # cap per camera so this completes in ~minutes
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs="+", default=["00_00", "00_01", "00_02"])
    parser.add_argument("--outdir", default="outputs/stage2_detection")
    return parser.parse_args()

args = parse_args()
CAMERA_IDS = [f"hd_{c}" for c in args.cams]
VIDEO_PATHS = [ROOT / "data" / "panoptic" / "171204_pose1" / "hdVideos" / f"hd_{c}.mp4" for c in args.cams]
OUT_DIR = ROOT / args.outdir


# ── Detector wrapper that also returns confidence scores ──────────────────────

def detect_with_scores(
    detector: YOLOXDetector,
    frame: np.ndarray,
) -> list[tuple[list[float], float]]:
    """Return list of (bbox [x1,y1,x2,y2], score) tuples.

    cigpose's YOLOXDetector.detect() drops scores after NMS.
    This replicates the same logic (same weights, same NMS) but keeps them.
    No new math — just the bookkeeping the library skips.
    """
    blob, ratio = detector._letterbox(frame)
    raw = detector.session.run(None, {detector.input_name: blob})[0][0]
    preds = detector._decode(raw)

    scores = preds[:, 4] * preds[:, 5]          # objectness × person_class
    keep = scores >= detector.conf_thresh
    if not np.any(keep):
        return []

    boxes_filt = preds[keep, :4]
    scores_filt = scores[keep]

    x1 = boxes_filt[:, 0] - boxes_filt[:, 2] / 2
    y1 = boxes_filt[:, 1] - boxes_filt[:, 3] / 2
    x2 = boxes_filt[:, 0] + boxes_filt[:, 2] / 2
    y2 = boxes_filt[:, 1] + boxes_filt[:, 3] / 2

    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)
    indices = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), scores_filt.tolist(),
        detector.conf_thresh, detector.nms_thresh,
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten()
    return [
        ([x1[i] / ratio, y1[i] / ratio, x2[i] / ratio, y2[i] / ratio],
         float(scores_filt[i]))
        for i in indices
    ]


def iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def pick_largest(detections: list[tuple[list[float], float]]) -> tuple[list[float], float] | None:
    """Pick the detection with the largest box area (matches PoseEstimator behaviour)."""
    if not detections:
        return None
    return max(detections, key=lambda d: (d[0][2]-d[0][0])*(d[0][3]-d[0][1]))


def draw_box(frame: np.ndarray, bbox: list[float], score: float, label: str) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}  conf={score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(out, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def draw_no_det(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, f"NO DET  frame={frame_idx}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return out


# ── Per-camera audit ──────────────────────────────────────────────────────────

def audit_camera(
    cam_id: str,
    video_path: Path,
    detector: YOLOXDetector,
    out_dir: Path,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_path = out_dir / f"{cam_id}_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))

    frames_processed = 0
    frames_with_det = 0
    all_confs: list[float] = []
    iou_scores: list[float] = []
    prev_bbox: list[float] | None = None

    while frames_processed < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detect_with_scores(detector, frame)
        best = pick_largest(dets)

        if best is not None:
            bbox, conf = best
            frames_with_det += 1
            all_confs.append(conf)

            if prev_bbox is not None:
                iou_scores.append(iou(prev_bbox, bbox))
            prev_bbox = bbox

            annotated = draw_box(frame, bbox, conf, cam_id)
        else:
            prev_bbox = None   # gap in detection resets IoU chain
            annotated = draw_no_det(frame, frames_processed)

        # Frame counter overlay
        cv2.putText(annotated, f"f={frames_processed}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(annotated)
        frames_processed += 1

        if frames_processed % 30 == 0:
            print(f"  [{cam_id}] frame {frames_processed}/{MAX_FRAMES} "
                  f"| det_rate={frames_with_det/frames_processed:.1%} "
                  f"| conf={np.mean(all_confs):.3f}" if all_confs else
                  f"  [{cam_id}] frame {frames_processed}/{MAX_FRAMES} | no detections yet")

    cap.release()
    writer.release()

    det_rate = frames_with_det / frames_processed if frames_processed else 0.0
    mean_conf = float(np.mean(all_confs)) if all_confs else float("nan")
    min_conf  = float(np.min(all_confs))  if all_confs else float("nan")
    mean_iou  = float(np.mean(iou_scores)) if iou_scores else float("nan")
    min_iou   = float(np.min(iou_scores))  if iou_scores else float("nan")

    return {
        "camera_id": cam_id,
        "video_path": str(video_path),
        "total_video_frames": total_video_frames,
        "frames_processed": frames_processed,
        "frames_with_detection": frames_with_det,
        "detection_rate_pct": round(det_rate * 100, 2),
        "conf_threshold": CONF_THRESHOLD,
        "mean_confidence": round(mean_conf, 4),
        "min_confidence": round(min_conf, 4),
        "mean_iou_consecutive": round(mean_iou, 4),
        "min_iou_consecutive": round(min_iou, 4),
        "annotated_clip": str(clip_path),
        # Pass/Fail inline guidance
        "pass_det_rate": det_rate >= 0.90,
        "pass_mean_conf": mean_conf >= CONF_THRESHOLD,
        "pass_iou_stability": mean_iou >= 0.70 if not np.isnan(mean_iou) else False,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLOX-nano from {DETECTOR_PATH}")
    detector = YOLOXDetector(
        str(DETECTOR_PATH),
        conf_thresh=CONF_THRESHOLD,
        nms_thresh=0.45,
        providers=PROVIDERS,
    )
    print(f"Detector loaded. Providers: {detector.session.get_providers()}")
    print()

    results = []
    for cam_id, video_path in zip(CAMERA_IDS, VIDEO_PATHS):
        if not video_path.exists():
            print(f"[SKIP] {cam_id} — video not found: {video_path}")
            results.append({"camera_id": cam_id, "error": "video not found"})
            continue

        print(f"=== Auditing {cam_id} ({video_path.name}) ===")
        r = audit_camera(cam_id, video_path, detector, OUT_DIR)
        results.append(r)

        # Per-camera summary to stdout
        status = lambda ok: "PASS" if ok else "FAIL"
        print(f"\n  Detection rate : {r['detection_rate_pct']:.1f}%  [{status(r['pass_det_rate'])}]  (threshold >=90%)")
        print(f"  Mean conf      : {r['mean_confidence']:.3f}       [{status(r['pass_mean_conf'])}]  (threshold >={CONF_THRESHOLD})")
        print(f"  Min conf       : {r['min_confidence']:.3f}")
        print(f"  Mean IoU (consec): {r['mean_iou_consecutive']:.3f}  [{status(r['pass_iou_stability'])}]  (threshold >=0.70)")
        print(f"  Min  IoU (consec): {r['min_iou_consecutive']:.3f}")
        print()

    # Save JSON
    metrics_path = OUT_DIR / "metrics.json"
    payload = {
        "stage": 2,
        "detector": "yolox_nano.onnx (cigpose/YOLOXDetector)",
        "note": (
            "Brief requested ultralytics YOLO; repo uses cigpose YOLOXDetector "
            "with the same yolox_nano.onnx weights. Divergence flagged."
        ),
        "conf_threshold": CONF_THRESHOLD,
        "max_frames_per_camera": MAX_FRAMES,
        "cameras": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Metrics written to {metrics_path}")
    print(f"Annotated clips in {OUT_DIR}")

    # Final table
    print("\n-- Stage 2 Summary Table " + "-" * 32)
    for r in results:
        if "error" in r:
            print(f"{r['camera_id']:<12}  ERROR: {r['error']}")
            continue
        all_pass = r['pass_det_rate'] and r['pass_mean_conf'] and r['pass_iou_stability']
        print(f"{r['camera_id']:<12} {r['detection_rate_pct']:>5.1f}%"
              f" {r['mean_confidence']:>9.3f}"
              f" {r['min_confidence']:>8.3f}"
              f" {r['mean_iou_consecutive']:>8.3f}"
              f"  {'PASS' if all_pass else 'FAIL'}")
    print()


if __name__ == "__main__":
    main()
