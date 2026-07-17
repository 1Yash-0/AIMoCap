"""Stage 4D: Drift Measurement across multiple anchors"""

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
CAMS = ["00_03", "00_04", "00_28", "00_24"]
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def main():
    print(f"Stage 4D Drift Measurement for {SEQ}\\n")
    
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    gt = {}
    for fi in range(0, 450):
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                gt[fi] = k[COCO17_TO_PAN19, :3]
                
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    
    # 10 anchors, each measuring across a 5-frame window for stability
    anchors = [20, 60, 100, 140, 180, 220, 260, 300, 340, 380]
    
    for cn in CAMS:
        print(f"--- Camera {cn} ---")
        c = calib[cn]
        rvec, _ = cv2.Rodrigues(c.R.astype(np.float64))
        
        anchor_best_offsets = []
        
        for anchor in anchors:
            test_frames = list(range(anchor, anchor + 5))
            
            # PRE-CACHE POSE ESTIMATIONS FOR THE 5 FRAMES
            cached_poses = {}
            for fi in test_frames:
                img_path = frames_dir / f"hd_{cn}" / f"{fi:08d}.jpg"
                if not img_path.exists(): continue
                fr = cv2.imread(str(img_path))
                p = model.estimate(fr, pick="largest")
                if p: cached_poses[fi] = p[0]
                
            errors_by_offset = {}
            for offset in range(-15, 16):
                errs = []
                for fi in test_frames:
                    if fi not in cached_poses: continue
                    p = cached_poses[fi]
                    
                    gt_frame = fi + offset
                    if gt_frame not in gt: continue
                    
                    e_sum = 0
                    n = 0
                    # Use stable torso/shoulder/hip joints for offset estimation: 
                    # 5: L_Shoulder, 6: R_Shoulder, 11: L_Hip, 12: R_Hip
                    for jid in [5, 6, 11, 12]:
                        if p.scores[jid] > 0.5 and not np.isnan(gt[gt_frame][jid]).any():
                            proj, _ = cv2.projectPoints(np.array([gt[gt_frame][jid]]), rvec, c.t.astype(np.float64), c.K.astype(np.float64), c.dist_coef)
                            e_sum += np.linalg.norm(p.keypoints[jid] - proj[0,0])
                            n += 1
                    if n > 0: errs.append(e_sum/n)
                
                if errs:
                    errors_by_offset[offset] = np.mean(errs)
                    
            if errors_by_offset:
                best_off = min(errors_by_offset.keys(), key=lambda k: errors_by_offset[k])
                anchor_best_offsets.append(best_off)
                print(f"  Anchor {anchor:03d}: offset = {best_off:>3} (err = {errors_by_offset[best_off]:.1f}px)")
            else:
                print(f"  Anchor {anchor:03d}: NO VALID DETECTIONS")
                
        if anchor_best_offsets:
            mi, ma = min(anchor_best_offsets), max(anchor_best_offsets)
            print(f"  -> Variation: {ma - mi} frames")
        print("")

if __name__ == "__main__":
    main()
