"""Project 3D GT to 2D to verify camera calibration and 2D model ordering."""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = "171204_pose3"
FRAME = 150
COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def main():
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    fpath = gt_dir / f"body3DScene_{FRAME:08d}.json"
    with open(fpath) as fp: d = json.load(fp)
    gt3d = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)[COCO_TO_PAN, :3] # COCO-17 order
    
    # Let's project GT3D to Camera 00_03
    cn = "00_03"
    K = calib[cn].K.astype(np.float64)
    R = calib[cn].R.astype(np.float64)
    t = calib[cn].t.astype(np.float64).reshape(3, 1)
    
    # Projection
    gt_cam = (R @ gt3d.T + t).T
    gt_cam = gt_cam / gt_cam[:, 2:3]
    gt_pix = (K @ gt_cam.T).T[:, :2]
    
    print("--- PROJECTED GT 2D (COCO-17 order) ---")
    print(f"  Shoulder L: {gt_pix[5,0]:.1f}, R: {gt_pix[6,0]:.1f}")
    print(f"  Elbow    L: {gt_pix[7,0]:.1f}, R: {gt_pix[8,0]:.1f}")
    print(f"  Wrist    L: {gt_pix[9,0]:.1f}, R: {gt_pix[10,0]:.1f}")
    
    print("\n--- ACTUAL RAW 2D DETECTIONS (From empirical sweep) ---")
    print("  Shoulder L: 1128.7, R: 1241.1")
    print("  Elbow    L: 1111.9, R: 1253.9")
    print("  Wrist    L: 1103.0, R: 1250.0")

if __name__ == "__main__":
    main()
