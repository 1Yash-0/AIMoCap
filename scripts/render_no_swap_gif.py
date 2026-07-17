"""Render a moving and rotating GIF of raw NO SWAP triangulation for the turn sequence."""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

SEQ = "171204_pose3"
LAGS = {"00_03": 0, "00_04": 0, "00_28": 18, "00_24": -1}
cams = list(LAGS.keys())
COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def main():
    print("Loading NO SWAP Triangulation data for frames 4750 to 4850...")
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    K_list = [calib[cn].K.astype(np.float64) for cn in cams]
    extrinsics = []
    for cn in cams:
        R = calib[cn].R.astype(np.float64)
        t = calib[cn].t.astype(np.float64).reshape(3, 1)
        extrinsics.append((R, t))

    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    frames = list(range(150, 250))
    num_frames = len(frames)
    
    all_kpts2d = []
    all_scores = []
    valid_gt = []
    
    for f in tqdm(frames, desc="Processing 2D and GT"):
        fpath = gt_dir / f"body3DScene_{f:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies") and len(d["bodies"]) > 0:
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                gt_pts = k[COCO_TO_PAN, :3]
                gt_scores = k[COCO_TO_PAN, 3]
                gt_pts[gt_scores == 0] = np.nan
                valid_gt.append(gt_pts)
            else:
                valid_gt.append(np.full((17, 3), np.nan))
        else:
            valid_gt.append(np.full((17, 3), np.nan))
            
        kpts = np.full((len(cams), 17, 2), np.nan)
        scores = np.zeros((len(cams), 17))
        for ci, cn in enumerate(cams):
            video_frame = f - LAGS[cn]
            img_path = frames_dir / f"hd_{cn}" / f"{video_frame:08d}.jpg"
            if img_path.exists():
                fr = cv2.imread(str(img_path))
                p = model.estimate(fr, pick="largest")
                if p:
                    # STRICTLY NO SWAP
                    kpts[ci] = p[0].keypoints[:17]
                    scores[ci] = p[0].scores[:17]
        all_kpts2d.append(kpts)
        all_scores.append(scores)
        
    print("Triangulating...")
    diag = triangulate_sequence_with_diagnostics(
        np.array(all_kpts2d), np.array(all_scores), K_list, extrinsics, min_conf=0.4, reproj_threshold_px=25.0
    )
    pts3d = diag.points3d
    
    # Evaluate MPJPE
    errors_body = []
    for i in range(num_frames):
        gt = valid_gt[i]
        pred = pts3d[i]
        for j in range(5, 17): # body only
            if np.isfinite(gt[j, 0]) and np.isfinite(pred[j, 0]):
                err_mm = np.linalg.norm(gt[j] - pred[j]) * 10.0
                errors_body.append(err_mm)
    print(f"Overall Body MPJPE: {np.mean(errors_body):.1f} mm")

    print("Rendering GIF...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-200, -100); ax.set_ylim(-50, 50); ax.set_zlim(0, 200)
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Z (depth, cm)"); ax.set_zlabel("Height (-Y, cm)")
    
    def plot_skel(kpts, color, label):
        val = ~np.isnan(kpts[:, 0])
        xs, ys, zs = kpts[:, 0], kpts[:, 2], -kpts[:, 1]
        sc = ax.scatter(xs[val], ys[val], zs[val], c=color, label=label, s=30)
        lines = []
        for p, c in BONES:
            if val[p] and val[c]:
                line, = ax.plot([xs[p], xs[c]], [ys[p], ys[c]], [zs[p], zs[c]], c=color, lw=2)
                lines.append(line)
        return [sc] + lines
    
    def update(frame_idx):
        ax.cla()
        ax.set_xlim(-200, -100); ax.set_ylim(-50, 50); ax.set_zlim(0, 200)
        ax.set_xlabel("X (cm)"); ax.set_ylabel("Z (depth, cm)"); ax.set_zlabel("Height (-Y, cm)")
        
        gt = valid_gt[frame_idx]
        pred = pts3d[frame_idx]
        actual_frame = frames[frame_idx]
        
        if not np.isnan(gt).all():
            plot_skel(gt, 'green', 'Ground Truth')
        if not np.isnan(pred).all():
            plot_skel(pred, 'red', 'Triangulated (NO SWAP)')
            
        ax.legend()
        ax.set_title(f"NO SWAP Triangulation - Frame {actual_frame}")
        
        # Rotate the view continuously over the 100 frames
        angle = (frame_idx / num_frames) * 360
        ax.view_init(elev=10, azim=angle)
        
    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
    out = ROOT / "no_swap_real.gif"
    anim.save(out, writer='pillow', fps=10)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
