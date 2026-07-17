"""Find the turning point."""

import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = "171204_pose3"
COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def get_camera_center(R, t): return -R.T @ t

def get_subject_forward(gt_kpts):
    if np.isnan(gt_kpts[0,0]) or np.isnan(gt_kpts[5,0]) or np.isnan(gt_kpts[6,0]): return None
    neck = (gt_kpts[5] + gt_kpts[6]) / 2.0
    fwd = gt_kpts[0] - neck
    fwd[1] = 0
    n = np.linalg.norm(fwd)
    if n < 1e-5: return None
    return fwd / n

def main():
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    R = calib["00_03"].R.astype(np.float64)
    t = calib["00_03"].t.astype(np.float64).reshape(3, 1)
    cam_pos = get_camera_center(R, t).flatten()
    
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    
    for f in range(5000):
        fpath = gt_dir / f"body3DScene_{f:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                gt = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)[COCO_TO_PAN, :3]
                fwd = get_subject_forward(gt)
                if fwd is not None:
                    pelvis = (gt[11] + gt[12]) / 2.0
                    v_cam = cam_pos - pelvis
                    v_cam[1] = 0
                    v_cam = v_cam / np.linalg.norm(v_cam)
                    dot = np.dot(fwd, v_cam)
                    if dot > 0.5:
                        print(f"Frame {f} dot is {dot:.2f}")
                        if dot > 0.9: return

if __name__ == "__main__":
    main()
