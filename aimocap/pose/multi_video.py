"""Multi-camera frame-by-frame video pose runner.

Reads N synchronized video feeds, runs the PoseEstimator on each frame for each camera,
draws an annotated grid overlay, and writes a unified `(num_frames, num_cameras, 133, 2)`
tensor of 2D keypoints to disk.

Usage:
    from aimocap.pose.multi_video import run_multi_video
    run_multi_video(["cam1.mp4", "cam2.mp4"], "out.npz", grid_dst="grid.mp4")
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from aimocap.pose.infer import PoseEstimator
from aimocap.video import _pick_subject
from aimocap.viz.overlay import draw_poses


def _create_grid(frames: list[np.ndarray]) -> np.ndarray:
    """Arranges a list of frames into a roughly square grid."""
    if not frames:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=frames[0].dtype)
    
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        
        # Resize frame if it doesn't match the grid cell size (shouldn't happen in a proper multi-cam setup, but just in case)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
            
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
        
        # Add camera index label
        cv2.putText(
            grid, f"Cam {i}", (c * w + 10, r * h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        
    return grid


def run_multi_video(
    sources: list[str | Path],
    npz_dst: str | Path,
    grid_dst: str | Path | None = None,
    *,
    estimator: PoseEstimator | None = None,
    threshold: float = 0.3,
    max_frames: int | None = None,
    progress_every: int = 10,
) -> None:
    """Run pose estimation over multiple synchronized videos synchronously.
    
    Parameters
    ----------
    sources     : list of input video paths.
    npz_dst     : output path for the extracted 2D keypoint tensor (.npz).
    grid_dst    : optional output path for the annotated grid video.
    threshold   : confidence threshold for subject picking.
    max_frames  : stop after this many frames (None = whole video).
    """
    if not sources:
        raise ValueError("No sources provided.")
        
    npz_dst = Path(npz_dst)
    npz_dst.parent.mkdir(parents=True, exist_ok=True)
    
    if grid_dst:
        grid_dst = Path(grid_dst)
        grid_dst.parent.mkdir(parents=True, exist_ok=True)
        
    est = estimator or PoseEstimator()
    
    caps = []
    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Video not found: {src_path}")
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise OSError(f"Could not open video: {src_path}")
        caps.append(cap)
        
    try:
        # We assume all cameras have the same fps. We take fps from cam 0.
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
        
        writer = None
        
        all_keypoints = []  # will hold list of shape (num_cameras, 133, 2)
        all_scores = []     # will hold list of shape (num_cameras, 133)
        
        frame_idx = 0
        while True:
            # Read synchronously
            frames_ok = []
            frames = []
            for cap in caps:
                ok, f = cap.read()
                frames_ok.append(ok)
                frames.append(f)
                
            if not all(frames_ok):
                # Stop if ANY camera runs out of frames
                break
                
            if frame_idx == 0:
                image_sizes = [(f.shape[1], f.shape[0]) for f in frames]
                
            if max_frames is not None and frame_idx >= max_frames:
                break
                
            t0 = time.perf_counter()
            
            cam_keypoints = []
            cam_scores = []
            annotated_frames = []
            
            for f in frames:
                poses = est.estimate(f, pick="all")
                subject = _pick_subject(poses, threshold) if poses else None
                
                if subject is not None:
                    cam_keypoints.append(subject.keypoints.copy())
                    cam_scores.append(subject.scores.copy())
                    to_draw = [subject]
                else:
                    # No person detected in this view
                    cam_keypoints.append(np.full((133, 2), np.nan, dtype=np.float32))
                    cam_scores.append(np.zeros(133, dtype=np.float32))
                    to_draw = []
                    
                if grid_dst:
                    annotated = draw_poses(f, to_draw, threshold=threshold) if to_draw else f.copy()
                    annotated_frames.append(annotated)
            
            all_keypoints.append(cam_keypoints)
            all_scores.append(cam_scores)
            
            if grid_dst:
                grid = _create_grid(annotated_frames)
                if writer is None:
                    h, w = grid.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if grid_dst.suffix.lower() == ".mp4" else "MJPG"))
                    writer = cv2.VideoWriter(str(grid_dst), fourcc, fps, (w, h))
                    if not writer.isOpened():
                        raise OSError(f"Could not open VideoWriter for {grid_dst}")
                writer.write(grid)
                
            dt = (time.perf_counter() - t0) * 1000.0
            
            if progress_every and (frame_idx % progress_every == 0):
                det_count = sum(1 for s in cam_scores if s.mean() > 0)
                print(f"  multi-frame {frame_idx:>4} | {det_count}/{len(caps)} cameras detected | {dt:.1f}ms")
                
            frame_idx += 1
            
    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            writer.release()
            
    # Save to disk
    kpt_arr = np.array(all_keypoints, dtype=np.float32)  # (F, C, 133, 2)
    score_arr = np.array(all_scores, dtype=np.float32)   # (F, C, 133)
    img_sizes_arr = np.array(image_sizes, dtype=np.int32)
    camera_names_arr = np.array([Path(src).name for src in sources], dtype=str)
    
    np.savez(npz_dst, keypoints=kpt_arr, scores=score_arr, image_sizes=img_sizes_arr, camera_names=camera_names_arr)
    
    print(f"\nDone: {frame_idx} frames across {len(caps)} cameras.")
    print(f"Saved tensors to {npz_dst} -> keypoints {kpt_arr.shape}, scores {score_arr.shape}, image_sizes {img_sizes_arr.shape}")
    if grid_dst:
        print(f"Saved grid video to {grid_dst}")

__all__ = ["run_multi_video"]
