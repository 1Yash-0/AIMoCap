"""Stage 4D.4: Quantifying the Occlusion Hallucination"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = "171204_pose3"
CAM = "00_03"
LAG = 0

FIXED_COCO_TO_PAN = [
    1,   # 0: Nose
    16,  # 1: LEye
    15,  # 2: REye
    18,  # 3: LEar
    17,  # 4: REar
    3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14
]

def main():
    print("Quantifying 2D Hallucinations on Camera 00_03 (Back View)")
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    gt = {}
    for fi in range(450):
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                gt[fi] = k[FIXED_COCO_TO_PAN, :3]
                
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    
    c = calib[CAM]
    rvec, _ = cv2.Rodrigues(c.R.astype(np.float64))
    
    body_errors = []
    face_errors = []
    
    for fi in range(450):
        gt_frame = fi + LAG
        if gt_frame not in gt: continue
        
        img_path = frames_dir / f"hd_{CAM}" / f"{fi:08d}.jpg"
        if not img_path.exists(): continue
        
        fr = cv2.imread(str(img_path))
        p = model.estimate(fr, pick="largest")
        if not p: continue
        p = p[0]
        
        # JIDs 0-4 are Face (Nose, Eyes, Ears) - heavily occluded from back
        # JIDs 5-16 are Body - more visible from back
        for jid in range(17):
            if p.scores[jid] > 0.5 and not np.isnan(gt[gt_frame][jid]).any():
                proj, _ = cv2.projectPoints(np.array([gt[gt_frame][jid]]), rvec, c.t.astype(np.float64), c.K.astype(np.float64), c.dist_coef)
                err = np.linalg.norm(p.keypoints[jid] - proj[0,0])
                
                if jid <= 4:
                    face_errors.append(err)
                else:
                    body_errors.append(err)
                    
    if body_errors and face_errors:
        print("\\n=== Reprojection Error Decomposition ===")
        print(f"Face Joints (Occluded from back):")
        print(f"  Samples : {len(face_errors)}")
        print(f"  Median  : {np.median(face_errors):.2f} px")
        print(f"  P95     : {np.percentile(face_errors, 95):.2f} px")
        
        print(f"\\nBody Joints (Mostly visible):")
        print(f"  Samples : {len(body_errors)}")
        print(f"  Median  : {np.median(body_errors):.2f} px")
        print(f"  P95     : {np.percentile(body_errors, 95):.2f} px")
        
        print(f"\\nTotal Mean Error: {np.mean(body_errors + face_errors):.2f} px")

if __name__ == "__main__":
    main()
