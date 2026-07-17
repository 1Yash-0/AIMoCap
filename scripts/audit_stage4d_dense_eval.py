"""Stage 4D: Dense Reprojection Verification using Cross-Correlation Lags"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = "171204_pose3"
# The robust lags found via cross-correlation
LAGS = {
    "00_03": 0,
    "00_04": 0,
    "00_28": 18,
    "00_24": -1
}
FIXED_COCO_TO_PAN = [
    1,   # COCO 0: Nose -> Pan 1 (Nose)
    16,  # COCO 1: LEye -> Pan 16 (lEye)
    15,  # COCO 2: REye -> Pan 15 (rEye)
    18,  # COCO 3: LEar -> Pan 18 (lEar)
    17,  # COCO 4: REar -> Pan 17 (rEar)
    3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14
]

def main():
    print(f"Dense Reprojection Test for {SEQ}\\n")
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    gt = {}
    for fi in range(0, 500):
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                gt[fi] = k[FIXED_COCO_TO_PAN, :3]
                
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    
    for cn, lag in LAGS.items():
        print(f"\\n--- Evaluating Camera {cn} with Lag {lag} ---")
        c = calib[cn]
        rvec, _ = cv2.Rodrigues(c.R.astype(np.float64))
        
        all_errors = []
        valid_frames = 0
        
        # Test across the full 450 frames
        for fi in tqdm(range(450)):
            gt_frame = fi + lag
            if gt_frame not in gt or gt_frame < 0:
                continue
                
            img_path = frames_dir / f"hd_{cn}" / f"{fi:08d}.jpg"
            if not img_path.exists(): continue
            
            fr = cv2.imread(str(img_path))
            p = model.estimate(fr, pick="largest")
            if not p: continue
            p = p[0]
            
            e_sum = 0
            n = 0
            for jid in range(17):
                if p.scores[jid] > 0.5 and not np.isnan(gt[gt_frame][jid]).any():
                    proj, _ = cv2.projectPoints(np.array([gt[gt_frame][jid]]), rvec, c.t.astype(np.float64), c.K.astype(np.float64), c.dist_coef)
                    e_sum += np.linalg.norm(p.keypoints[jid] - proj[0,0])
                    n += 1
                    
            if n > 0:
                all_errors.append(e_sum/n)
                valid_frames += 1
                
        if all_errors:
            median_err = np.median(all_errors)
            p95_err = np.percentile(all_errors, 95)
            print(f"  Valid Frames: {valid_frames}/450")
            print(f"  Median Error: {median_err:.2f} px")
            print(f"  P95 Error   : {p95_err:.2f} px")
        else:
            print("  NO VALID DETECTIONS")

if __name__ == "__main__":
    main()
