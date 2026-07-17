"""Stage 3 — 2D Pose Estimation Audit.

Runs cigpose-x (wholebody 133-kpt) on the same 3 Panoptic cameras / 300 frames
as Stage 2.

Measures (per brief):
  - Mean confidence per keypoint across all frames.
  - % of frames where ANY keypoint confidence < 0.3.
  - Frame-to-frame jitter per keypoint (pixel distance between consecutive frames).
  - At matching frames: how many high-confidence (>=0.65) keypoints per camera.

Integration check (per brief):
  - % of frames where a hand/wrist keypoint (indices 9, 10, 91, 112) falls
    outside its YOLO bbox — the arms-spread problem as a number.

Writes to outputs/stage3_pose/:
  - metrics.json
  - kpts.npz          — (F, C, 133, 2) keypoints + (F, C, 133) scores
  - confidence_heatmap.png
  - cam{id}_pose.mp4  — skeleton-annotated clip per camera

Usage:
    python scripts/stage3_pose_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aimocap  # noqa: F401
from aimocap.pose.infer import PoseEstimator, Pose2D
from aimocap.pose.keypoints import (
    KEYPOINT_NAMES_133, SKELETON_133,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HAND_WRIST, RIGHT_HAND_WRIST,
)

import argparse

# ── Config ────────────────────────────────────────────────────────────────────
POSE_MODEL   = ROOT / "models" / "cigpose-x_coco-wholebody_384x288.onnx"
DETECTOR     = ROOT / "models" / "yolox_nano.onnx"
CONF_THRESH  = 0.3     # low-conf flag threshold (per brief)
HIGH_CONF    = 0.65    # "high-confidence keypoint" threshold (matches new min_conf)
START_SECOND = 5.0     # skip this many seconds before sampling
MAX_FRAMES   = 300
PROVIDERS    = ["CUDAExecutionProvider", "CPUExecutionProvider"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs="+", default=["00_00", "00_01", "00_02"])
    parser.add_argument("--outdir", default="outputs/stage3_pose")
    return parser.parse_args()

args = parse_args()
CAMERA_IDS = [f"hd_{c}" for c in args.cams]
VIDEO_PATHS = [ROOT / "data" / "panoptic" / "171204_pose1" / "hdVideos" / f"hd_{c}.mp4" for c in args.cams]
OUT_DIR = ROOT / args.outdir

# Wrist/hand keypoint indices for the integration check
HAND_KPT_INDICES = [LEFT_WRIST, RIGHT_WRIST, LEFT_HAND_WRIST, RIGHT_HAND_WRIST]

# Colour palette for skeleton drawing
_BODY_COLOR   = (0, 255, 128)
_HAND_COLOR   = (255, 180, 0)
_FACE_COLOR   = (180, 180, 255)
_LOW_COLOR    = (0, 0, 255)   # red for low-confidence keypoints
_KPT_RADIUS   = 3
_BONE_THICK   = 1


def _kpt_color(score: float) -> tuple[int, int, int]:
    if score >= HIGH_CONF:
        return _BODY_COLOR
    if score >= CONF_THRESH:
        return _HAND_COLOR
    return _LOW_COLOR


def draw_skeleton(
    frame: np.ndarray,
    pose: Pose2D,
    frame_idx: int,
    cam_id: str,
) -> np.ndarray:
    out = frame.copy()
    kpts, scores = pose.keypoints, pose.scores
    h, w = out.shape[:2]

    # Bones
    for i, j in SKELETON_133:
        if i >= len(kpts) or j >= len(kpts):
            continue
        if scores[i] < CONF_THRESH or scores[j] < CONF_THRESH:
            continue
        p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
        p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
        # skip if out of frame
        if not (0 <= p1[0] < w and 0 <= p1[1] < h):
            continue
        if not (0 <= p2[0] < w and 0 <= p2[1] < h):
            continue
        cv2.line(out, p1, p2, _BODY_COLOR, _BONE_THICK, cv2.LINE_AA)

    # Keypoints
    for idx in range(len(kpts)):
        score = float(scores[idx])
        if score < CONF_THRESH:
            continue
        x, y = int(kpts[idx, 0]), int(kpts[idx, 1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        cv2.circle(out, (x, y), _KPT_RADIUS, _kpt_color(score), -1, cv2.LINE_AA)

    # YOLO bbox
    x1, y1, x2, y2 = (int(v) for v in pose.bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # HUD
    high_conf_count = int(np.sum(scores >= HIGH_CONF))
    cv2.putText(out, f"{cam_id}  f={frame_idx}  hc={high_conf_count}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def kpt_outside_bbox(kpt_xy: np.ndarray, bbox: np.ndarray) -> bool:
    x, y = kpt_xy
    x1, y1, x2, y2 = bbox
    return not (x1 <= x <= x2 and y1 <= y <= y2)


# ── Per-camera run ────────────────────────────────────────────────────────────

def run_camera(
    cam_id: str,
    video_path: Path,
    estimator: PoseEstimator,
    out_dir: Path,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(fps * START_SECOND)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_path = out_dir / f"{cam_id}_pose.mp4"
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )

    all_kpts:   list[np.ndarray] = []   # (133, 2) per frame
    all_scores: list[np.ndarray] = []   # (133,)   per frame
    all_bboxes: list[np.ndarray] = []   # (4,)     per frame

    frames_no_det  = 0
    frames_low_any = 0   # frames where ANY keypoint < CONF_THRESH
    hand_outside   = 0   # integration check
    hand_total     = 0

    frame_idx = 0
    while frame_idx < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break

        poses = estimator.estimate(frame, pick="largest")

        if not poses:
            frames_no_det += 1
            all_kpts.append(np.full((133, 2), np.nan, dtype=np.float32))
            all_scores.append(np.zeros(133, dtype=np.float32))
            all_bboxes.append(np.zeros(4, dtype=np.float32))
            annotated = frame.copy()
            cv2.putText(annotated, f"NO DET  f={frame_idx}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            pose = poses[0]
            all_kpts.append(pose.keypoints.copy())
            all_scores.append(pose.scores.copy())
            all_bboxes.append(pose.bbox.copy())

            # Low-conf frame check
            if np.any(pose.scores < CONF_THRESH):
                frames_low_any += 1

            # Integration check: hand/wrist outside bbox?
            for hi in HAND_KPT_INDICES:
                if pose.scores[hi] >= HIGH_CONF:
                    hand_total += 1
                    if kpt_outside_bbox(pose.keypoints[hi], pose.bbox):
                        hand_outside += 1

            annotated = draw_skeleton(frame, pose, frame_idx, cam_id)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0:
            det_so_far = frame_idx - frames_no_det
            print(f"  [{cam_id}] frame {frame_idx}/{MAX_FRAMES} | det={det_so_far}/{frame_idx}")

    cap.release()
    writer.release()

    kpts_arr   = np.array(all_kpts,   dtype=np.float32)   # (F, 133, 2)
    scores_arr = np.array(all_scores, dtype=np.float32)   # (F, 133)
    bboxes_arr = np.array(all_bboxes, dtype=np.float32)   # (F, 4)

    # Per-keypoint mean confidence (ignoring frames with no detection)
    valid_mask = np.isfinite(kpts_arr[:, :, 0])  # (F, 133)
    mean_conf_per_kpt = np.where(
        valid_mask.any(axis=0),
        np.nanmean(np.where(valid_mask, scores_arr, np.nan), axis=0),
        0.0,
    )  # (133,)

    # Jitter: pixel distance between consecutive frames (only where both valid)
    diff   = np.diff(kpts_arr, axis=0)            # (F-1, 133, 2)
    jitter = np.linalg.norm(diff, axis=-1)         # (F-1, 133)
    both_valid = valid_mask[:-1] & valid_mask[1:]
    mean_jitter_per_kpt = np.where(
        both_valid.any(axis=0),
        np.nanmean(np.where(both_valid, jitter, np.nan), axis=0),
        np.nan,
    )  # (133,)

    # High-confidence keypoint count per frame
    hc_per_frame = np.sum(scores_arr >= HIGH_CONF, axis=1)  # (F,)

    # Chronically low keypoints (mean conf < CONF_THRESH)
    chronic_low = [
        {"index": int(i), "name": KEYPOINT_NAMES_133[i], "mean_conf": round(float(mean_conf_per_kpt[i]), 4)}
        for i in np.where(mean_conf_per_kpt < CONF_THRESH)[0]
    ]

    return {
        "camera_id": cam_id,
        "frames_processed": frame_idx,
        "frames_no_detection": frames_no_det,
        "frames_low_any_kpt_pct": round(frames_low_any / max(frame_idx, 1) * 100, 2),
        "mean_high_conf_kpts_per_frame": round(float(hc_per_frame.mean()), 1),
        "min_high_conf_kpts_per_frame": int(hc_per_frame.min()),
        "mean_jitter_body_px": round(float(np.nanmean(mean_jitter_per_kpt[:17])), 3),
        "max_jitter_body_px": round(float(np.nanmax(mean_jitter_per_kpt[:17])), 3),
        "chronic_low_kpts": chronic_low,
        "integration_hand_outside_bbox_pct": round(hand_outside / max(hand_total, 1) * 100, 2),
        "annotated_clip": str(clip_path),
        # raw arrays for NPZ assembly
        "_kpts":   kpts_arr,
        "_scores": scores_arr,
        "_bboxes": bboxes_arr,
        "_mean_conf_per_kpt": mean_conf_per_kpt,
        "_mean_jitter_per_kpt": mean_jitter_per_kpt,
    }


# ── Confidence heatmap ────────────────────────────────────────────────────────

def render_confidence_heatmap(
    results: list[dict],
    out_png: Path,
) -> None:
    cam_ids = [r["camera_id"] for r in results]
    matrix  = np.stack([r["_mean_conf_per_kpt"] for r in results], axis=0)  # (C, 133)

    fig, ax = plt.subplots(figsize=(20, max(3, 1.5 * len(cam_ids))))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Stage 3 — Mean 2D keypoint confidence (cigpose-x, 300 frames)")
    ax.set_xlabel("Keypoint index")
    ax.set_ylabel("Camera")
    ax.set_yticks(range(len(cam_ids)))
    ax.set_yticklabels(cam_ids)
    ax.axvline(16.5, color="white", lw=0.8, linestyle="--", label="body|feet")
    ax.axvline(22.5, color="yellow", lw=0.8, linestyle="--", label="feet|face")
    ax.axvline(90.5, color="orange", lw=0.8, linestyle="--", label="face|lhand")
    ax.axvline(111.5, color="red", lw=0.8, linestyle="--", label="lhand|rhand")
    fig.colorbar(im, ax=ax, label="Mean confidence")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved: {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PoseEstimator (cigpose-x wholebody)...")
    estimator = PoseEstimator(
        pose_model=POSE_MODEL,
        detector=DETECTOR,
        providers=PROVIDERS,
        det_threshold=CONF_THRESH,
    )
    print(f"Providers active: {estimator.active_providers}\n")

    results = []
    for cam_id, video_path in zip(CAMERA_IDS, VIDEO_PATHS):
        if not video_path.exists():
            print(f"[SKIP] {cam_id}: {video_path} not found")
            continue
        print(f"=== {cam_id} ===")
        r = run_camera(cam_id, video_path, estimator, OUT_DIR)
        results.append(r)
        print(f"  Low-kpt frames : {r['frames_low_any_kpt_pct']}%")
        print(f"  HC kpts/frame  : {r['mean_high_conf_kpts_per_frame']} mean, {r['min_high_conf_kpts_per_frame']} min")
        print(f"  Body jitter    : {r['mean_jitter_body_px']} px mean, {r['max_jitter_body_px']} px max")
        print(f"  Chronic low kpts ({len(r['chronic_low_kpts'])}): "
              + ", ".join(f"{k['name']}={k['mean_conf']}" for k in r['chronic_low_kpts'][:5])
              + ("..." if len(r['chronic_low_kpts']) > 5 else ""))
        print(f"  Hand outside bbox: {r['integration_hand_outside_bbox_pct']}%")
        print()

    if not results:
        print("No cameras processed.")
        return

    # Save NPZ
    npz_path = OUT_DIR / "kpts.npz"
    np.savez(
        npz_path,
        keypoints=np.stack([r["_kpts"] for r in results], axis=1),    # (F, C, 133, 2)
        scores=np.stack([r["_scores"] for r in results], axis=1),      # (F, C, 133)
        camera_names=np.array([r["camera_id"] for r in results]),
    )
    print(f"NPZ saved: {npz_path}")

    # Heatmap
    render_confidence_heatmap(results, OUT_DIR / "confidence_heatmap.png")

    # Strip raw arrays before JSON serialise
    clean = []
    for r in results:
        c = {k: v for k, v in r.items() if not k.startswith("_")}
        clean.append(c)

    metrics = {
        "stage": 3,
        "model": "cigpose-x_coco-wholebody_384x288.onnx",
        "conf_threshold": CONF_THRESH,
        "high_conf_threshold": HIGH_CONF,
        "max_frames_per_camera": MAX_FRAMES,
        "cameras": clean,
    }
    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved: {metrics_path}")

    # Summary table
    print("\n-- Stage 3 Summary " + "-" * 45)
    print(f"{'Camera':<12} {'LowKpt%':>8} {'HC/frame':>9} {'BodyJitter':>11} {'HandOutside%':>13}")
    print("-" * 58)
    for r in clean:
        print(f"{r['camera_id']:<12}"
              f" {r['frames_low_any_kpt_pct']:>7.1f}%"
              f" {r['mean_high_conf_kpts_per_frame']:>9.1f}"
              f" {r['mean_jitter_body_px']:>10.2f}px"
              f" {r['integration_hand_outside_bbox_pct']:>12.1f}%")
    print()


if __name__ == "__main__":
    main()
