"""Stage 4D.3: Visual Audit of 2D/3D Joint Alignment"""

import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = "171204_pose3"
CAM = "00_03"
LAG = 0

# Original (possibly buggy) mapping
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
# Let's fix the mapping just in case, but plot Nose (0) which is 1 in both.
# COCO: 0=Nose, 1=LEye, 2=REye, 3=LEar, 4=REar
# Pan:  1=Nose, 15=REye, 16=LEye, 17=REar, 18=LEar
FIXED_COCO_TO_PAN = [
    1,   # 0: Nose -> Nose (1)
    16,  # 1: LEye -> lEye (16)
    15,  # 2: REye -> rEye (15)
    18,  # 3: LEar -> lEar (18)
    17,  # 4: REar -> rEar (17)
    3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14 # body is correct
]

ARTIFACTS_DIR = Path(r"C:\Users\prade\.gemini\antigravity-ide\brain\395c9de1-f24b-42ee-9272-3ef825c55485")

def main():
    print("Visual Audit of Nose Alignment")
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    gt_buggy = {}
    gt_fixed = {}
    for fi in range(150, 160): # 10 frames
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                gt_buggy[fi] = k[COCO17_TO_PAN19, :3]
                gt_fixed[fi] = k[FIXED_COCO_TO_PAN, :3]
                
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    
    c = calib[CAM]
    rvec, _ = cv2.Rodrigues(c.R.astype(np.float64))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    # Test frames where the subject is likely visible
    for fi in [150, 151, 152, 153, 154, 155]:
        gt_frame = fi + LAG
        if gt_frame not in gt_buggy: continue
        
        img_path = frames_dir / f"hd_{CAM}" / f"{fi:08d}.jpg"
        if not img_path.exists(): continue
        
        fr = cv2.imread(str(img_path))
        fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        p = model.estimate(fr, pick="largest")
        if not p: continue
        p = p[0]
        
        # Project buggy and fixed nose
        proj_buggy, _ = cv2.projectPoints(np.array([gt_buggy[gt_frame][0]]), rvec, c.t.astype(np.float64), c.K.astype(np.float64), c.dist_coef)
        proj_buggy = proj_buggy[0, 0]
        
        # In this specific case, Nose (0) is mapped to 1 in both, so buggy and fixed should be identical.
        # But we will plot the face to see what happens.
        # Let's project ALL 17 fixed joints to see if they align perfectly!
        proj_all_fixed, _ = cv2.projectPoints(gt_fixed[gt_frame], rvec, c.t.astype(np.float64), c.K.astype(np.float64), c.dist_coef)
        proj_all_fixed = proj_all_fixed.reshape(-1, 2)
        
        ax = axes[plot_idx]
        # Crop around the face (Nose detected by CIGPose)
        n_det = p.keypoints[0]
        
        x_min = max(0, int(n_det[0] - 150))
        x_max = min(fr_rgb.shape[1], int(n_det[0] + 150))
        y_min = max(0, int(n_det[1] - 150))
        y_max = min(fr_rgb.shape[0], int(n_det[1] + 150))
        
        ax.imshow(fr_rgb[y_min:y_max, x_min:x_max])
        
        # Shift coords to cropped
        ax.scatter(n_det[0] - x_min, n_det[1] - y_min, c='red', s=50, label="CIGPose (Detected)", marker='o')
        ax.scatter(proj_buggy[0] - x_min, proj_buggy[1] - y_min, c='lime', s=50, label="GT Nose", marker='x')
        
        # Draw eyes and ears too
        # LEye = 1, REye = 2, LEar = 3, REar = 4
        for jid in [1, 2, 3, 4]:
            if not np.isnan(gt_fixed[gt_frame][jid]).any():
                ax.scatter(proj_all_fixed[jid, 0] - x_min, proj_all_fixed[jid, 1] - y_min, c='blue', s=30, marker='x')
        
        ax.set_title(f"Frame {fi} / GT {gt_frame}")
        if plot_idx == 0:
            ax.legend()
            
        plot_idx += 1
        if plot_idx >= 6: break
        
    plt.tight_layout()
    plot_path = ARTIFACTS_DIR / f"visual_audit_nose_{CAM}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved visual audit plot to {plot_path}")

if __name__ == "__main__":
    main()
