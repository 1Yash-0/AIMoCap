"""Stage 4C: Independent Sync Verification"""

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
    print(f"Stage 4C Sync Verification for {SEQ}\\n")
    
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    gt = {}
    for fi in range(130, 190):
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                gt[fi] = k[COCO17_TO_PAN19, :3]
                
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    
    # We will test frames 150 to 159.
    test_frames = list(range(150, 160))
    
    for cn in CAMS:
        print(f"Testing {cn} offsets [-10, +10]...")
        c = calib[cn]
        rvec, _ = cv2.Rodrigues(c.R.astype(np.float64))
        
        errors_by_offset = {}
        for offset in range(-10, 11):
            errs = []
            for fi in test_frames:
                img_path = frames_dir / f"hd_{cn}" / f"{fi:08d}.jpg"
                if not img_path.exists(): continue
                fr = cv2.imread(str(img_path))
                
                gt_frame = fi + offset
                if gt_frame not in gt: continue
                
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
                if n > 0: errs.append(e_sum/n)
            
            if errs:
                errors_by_offset[offset] = np.mean(errs)
                
        best_off = min(errors_by_offset.keys(), key=lambda k: errors_by_offset[k])
        print(f"  Best offset for {cn}: {best_off} (Error={errors_by_offset[best_off]:.1f}px)")
        
        if best_off != 0:
            print(f"  [!] ERROR: {cn} requires non-zero offset {best_off}!")
            
    print("\\nFinished checking all cameras.")

if __name__ == "__main__":
    main()
