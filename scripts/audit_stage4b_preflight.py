"""Preflight 20 frames for Stage 4B Candidates"""

import sys
from pathlib import Path
import json
import urllib.request
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib

COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
CRITICAL_JOINTS = {
    "l_knee": 13, "r_knee": 14, 
    "l_ankle": 15, "r_ankle": 16, 
    "l_wrist": 9, "r_wrist": 10, 
    "l_hip": 11, "r_hip": 12
}

def get_calib(seq):
    url = f"http://domedb.perception.cs.cmu.edu/webdata/dataset/{seq}/calibration_{seq}.json"
    dst = ROOT / f"data/panoptic/{seq}/calibration_{seq}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        urllib.request.urlretrieve(url, dst)
    return load_panoptic_calib(dst)

def load_gt(seq, frame_range):
    gt_dir = ROOT / f"data/panoptic/{seq}/hdPose3d_stage1_coco19"
    if not gt_dir.exists():
        return None
    kpts = {}
    for fi in frame_range:
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                kpts[fi] = k[COCO17_TO_PAN19, :3]
    return kpts

def ray_angle(C1, C2, X):
    v1 = X - C1
    v2 = X - C2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos_t) * 180 / np.pi

def preflight(seq):
    print("\\n" + "="*80)
    print(f"PREFLIGHT: {seq}")
    calib = get_calib(seq)
    
    # Select 4 cameras with wide distribution
    cams = {c.name: c for c in calib.values() if c.name.startswith("00_")}
    cam_data = {}
    for n, c in cams.items():
        C = -c.R.astype(np.float64).T @ c.t.reshape(3, 1).astype(np.float64)
        az = np.arctan2(C[1,0], C[0,0]) * 180 / np.pi
        el = np.arctan2(C[2,0], np.linalg.norm(C[0:2,0])) * 180 / np.pi
        cam_data[n] = {"C": C.flatten(), "az": az, "el": el, "c": c}
        
    # We need 4 cameras. Typical wide spread in Panoptic (only -20 to -160 available)
    # Target: -30, -60, -110, -150
    selected = []
    for target in [-30, -60, -110, -150]:
        best = min(cam_data.keys(), key=lambda n: abs(cam_data[n]["az"] - target) if 0 <= cam_data[n]["el"] <= 35 else 999)
        selected.append(best)
    print(f"Selected Geometry Cameras: {selected}")
    for n in selected:
        print(f"  {n}: Az={cam_data[n]['az']:.1f}, El={cam_data[n]['el']:.1f}")

    start_frame = 150
    frame_range = range(start_frame, start_frame + 20)
    gt = load_gt(seq, frame_range)
    if not gt:
        print("GT not available or extracted yet.")
        return False
        
    model = PoseEstimator()
    errs_2d = {n: {fi: {} for fi in frame_range} for n in selected}
    
    for cn in selected:
        url = f"http://domedb.perception.cs.cmu.edu/webdata/dataset/{seq}/videos/hd_shared_crf20/hd_{cn}.mp4"
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        c = cam_data[cn]["c"]
        rvec, _ = cv2.Rodrigues(c.R.astype(np.float64))
        K = c.K.astype(np.float64)
        t = c.t.astype(np.float64)
        dist = c.dist_coef.astype(np.float64)
        
        for fi in frame_range:
            ret, frame = cap.read()
            if not ret: break
            p = model.estimate(frame, pick="largest")
            if p and fi in gt and not np.isnan(gt[fi]).all():
                p = p[0]
                for jid in range(17):
                    gate = 0.35 if jid in [11,12,13,14,15,16] else 0.5
                    if p.scores[jid] >= gate and not np.isnan(gt[fi][jid]).any():
                        proj, _ = cv2.projectPoints(np.array([gt[fi][jid]]), rvec, t, K, dist)
                        errs_2d[cn][fi][jid] = np.linalg.norm(p.keypoints[jid] - proj[0,0])
        cap.release()
        
    # Check Preflight Conditions
    joint_acc_views = {name: [] for name in CRITICAL_JOINTS}
    for fi in frame_range:
        if fi not in gt: continue
        for name, jid in CRITICAL_JOINTS.items():
            acc_count = 0
            for cn in selected:
                if jid in errs_2d[cn][fi] and errs_2d[cn][fi][jid] <= 20:
                    acc_count += 1
            joint_acc_views[name].append(acc_count)
            
    passed = True
    print("\\nVisibility Results (20 frames):")
    for name in CRITICAL_JOINTS:
        views = joint_acc_views[name]
        if not views: continue
        pct_ge2 = np.mean(np.array(views) >= 2) * 100
        print(f"  {name:<8}: {pct_ge2:5.1f}% frames have >= 2 accurate views")
        if pct_ge2 < 95.0:
            passed = False
            
    if passed:
        print("-> SEQUENCE PASSES PREFLIGHT!")
    else:
        print("-> SEQUENCE FAILS PREFLIGHT (Table/Occlusion likely present)")
    return passed

if __name__ == "__main__":
    for seq in ["171204_pose2", "171204_pose3"]:
        preflight(seq)
