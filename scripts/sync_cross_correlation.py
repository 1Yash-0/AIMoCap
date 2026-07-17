"""Stage 4D.2: Motion-Energy Cross-Correlation Sync

Extracts 1D motion-energy signals (velocity profiles) from both the 3D ground truth 
and the 2D video observations. Cross-correlates them to find the true global time lag.
"""

import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = "171204_pose3"
CAMS = ["00_03", "00_04", "00_28", "00_24"]
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

STABLE_JOINTS = [5, 6, 11, 12]  # L_Shoulder, R_Shoulder, L_Hip, R_Hip
ARTIFACTS_DIR = Path(r"C:\Users\prade\.gemini\antigravity-ide\brain\395c9de1-f24b-42ee-9272-3ef825c55485")

def calc_motion_energy(pts_t0, pts_t1, valid_t0, valid_t1, joints):
    """Calculates average 2D displacement across valid joints."""
    energy = 0.0
    count = 0
    for j in joints:
        if valid_t0[j] and valid_t1[j]:
            energy += np.linalg.norm(pts_t1[j] - pts_t0[j])
            count += 1
    return energy / count if count > 0 else 0.0

def main():
    print(f"Motion-Energy Sync for {SEQ}\\n")
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    # 1. Load GT 3D
    print("Loading Ground Truth...")
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    gt = {}
    for fi in range(0, 500): # Read up to 500 to allow for positive lags
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                gt[fi] = k[COCO17_TO_PAN19, :3]
                
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    
    frames_to_process = 450
    
    for cn in CAMS:
        print(f"\\n--- Processing Camera {cn} ---")
        c = calib[cn]
        rvec, _ = cv2.Rodrigues(c.R.astype(np.float64))
        
        # Array to store signals
        obs_energy = np.zeros(frames_to_process - 1)
        gt_energy = np.zeros(frames_to_process - 1)
        
        prev_obs_pts = None
        prev_obs_valid = None
        prev_gt_pts = None
        prev_gt_valid = None
        
        print("Extracting frames and running pose estimation...")
        for fi in tqdm(range(frames_to_process)):
            # --- VIDEO OBSERVATION ---
            img_path = frames_dir / f"hd_{cn}" / f"{fi:08d}.jpg"
            if not img_path.exists():
                obs_pts, obs_valid = None, None
            else:
                fr = cv2.imread(str(img_path))
                p = model.estimate(fr, pick="largest")
                if p:
                    obs_pts = p[0].keypoints
                    obs_valid = p[0].scores > 0.5
                else:
                    obs_pts, obs_valid = None, None
                    
            # --- GT PROJECTION ---
            if fi in gt:
                proj, _ = cv2.projectPoints(gt[fi], rvec, c.t.astype(np.float64), c.K.astype(np.float64), c.dist_coef)
                gt_pts = proj.reshape(-1, 2)
                gt_valid = ~np.isnan(gt[fi]).any(axis=1)
            else:
                gt_pts, gt_valid = None, None
                
            # --- COMPUTE ENERGY (t vs t-1) ---
            if fi > 0:
                e_idx = fi - 1
                if prev_obs_pts is not None and obs_pts is not None:
                    obs_energy[e_idx] = calc_motion_energy(prev_obs_pts, obs_pts, prev_obs_valid, obs_valid, STABLE_JOINTS)
                if prev_gt_pts is not None and gt_pts is not None:
                    gt_energy[e_idx] = calc_motion_energy(prev_gt_pts, gt_pts, prev_gt_valid, gt_valid, STABLE_JOINTS)
                    
            prev_obs_pts, prev_obs_valid = obs_pts, obs_valid
            prev_gt_pts, prev_gt_valid = gt_pts, gt_valid

        # Normalize signals to zero mean, unit variance for cross-correlation
        obs_norm = obs_energy - np.mean(obs_energy)
        obs_norm = obs_norm / (np.std(obs_norm) + 1e-8)
        
        gt_norm = gt_energy - np.mean(gt_energy)
        gt_norm = gt_norm / (np.std(gt_norm) + 1e-8)
        
        # Cross-correlate
        # mode='full' returns len(obs) + len(gt) - 1. 
        # The zero-lag is at index len(obs) - 1.
        corr = np.correlate(obs_norm, gt_norm, mode='full')
        lags = np.arange(-len(obs_norm) + 1, len(gt_norm))
        
        best_lag_idx = np.argmax(corr)
        best_lag = lags[best_lag_idx]
        
        # Wait, if obs = gt shifted right by 5 (so obs[5] matches gt[0]), 
        # cross-correlate(obs, gt) will peak at lag = +5.
        # This means GT is ahead of Video by 5 frames (GT frame 5 corresponds to Video frame 0).
        print(f"  Max Cross-Correlation Lag for {cn}: {best_lag} frames")
        
        # Plot signals
        np.savez(f'data/panoptic/{SEQ}/energy_{cn}.npz', obs=obs_norm, gt=gt_norm, lag=best_lag)
        
        plt.figure(figsize=(15, 6))
        plt.plot(gt_norm, label="GT Energy (Target)", alpha=0.7)
        
        # Shift observation for visualization
        if best_lag > 0:
            obs_shifted = np.pad(obs_norm, (0, best_lag), mode='constant')[:-best_lag]
        elif best_lag < 0:
            obs_shifted = np.pad(obs_norm, (-best_lag, 0), mode='constant')[-best_lag:]
        else:
            obs_shifted = obs_norm
            
        plt.plot(obs_shifted, label=f"Video Energy (Shifted by {best_lag})", alpha=0.7)
        plt.title(f"Motion Energy Cross-Correlation Sync - Camera {cn} (Lag: {best_lag})")
        plt.xlabel("Video Frame Index")
        plt.ylabel("Normalized Displacement")
        plt.legend()
        plt.tight_layout()
        plot_path = ARTIFACTS_DIR / f"sync_plot_{cn}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"  Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
