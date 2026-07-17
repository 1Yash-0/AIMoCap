"""Stage 4B — Representative Raw-3D Validation"""

import sys
from pathlib import Path
import json
import itertools
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.math.triangulate import triangulate_robust

SEQ = "171204_pose3"
CAMS = ["00_03", "00_04", "00_28", "00_24"]

N_FRAMES = 300
START_FRAME = 150
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
CRITICAL_JOINTS = {
    "l_knee": 13, "r_knee": 14, 
    "l_ankle": 15, "r_ankle": 16, 
    "l_wrist": 9, "r_wrist": 10, 
    "l_hip": 11, "r_hip": 12
}

def load_gt():
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    kpts = {}
    for fi in range(N_FRAMES):
        f = START_FRAME + fi
        fpath = gt_dir / f"body3DScene_{f:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                kpts[fi] = k[COCO17_TO_PAN19, :3]
    return kpts

def load_calib():
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    cams = {}
    for cn in CAMS:
        c = calib[cn]
        P = c.K.astype(np.float64) @ np.hstack([c.R.astype(np.float64), c.t.reshape(3, 1).astype(np.float64)])
        C = -c.R.astype(np.float64).T @ c.t.reshape(3, 1).astype(np.float64)
        cams[cn] = {"name": cn, "K": c.K.astype(np.float64), "R": c.R.astype(np.float64), 
                     "t": c.t.reshape(3, 1).astype(np.float64), "distCoef": c.dist_coef, "P": P, "C": C.flatten()}
    return cams

def calc_rc_mpjpe(pred, gt):
    rp = (pred[11] + pred[12]) / 2.0
    rg = (gt[11] + gt[12]) / 2.0
    return np.nanmean(np.linalg.norm((pred - rp) - (gt - rg), axis=1) * 10.0)

def ray_angle(C1, C2, X):
    v1 = X - C1
    v2 = X - C2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos_t) * 180 / np.pi

def main():
    print(f"Loading {SEQ} Ground Truth...")
    gt = load_gt()
    cams = load_calib()
    
    print("Running CIGPose Inference...")
    model = PoseEstimator()
    preds = {cn: [None]*N_FRAMES for cn in CAMS}
    for cn in CAMS:
        url = ROOT / f"data/panoptic/{SEQ}/hdVideos/hd_{cn}.mp4"
        cap = cv2.VideoCapture(str(url))
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
        for fi in range(N_FRAMES):
            ret, frame = cap.read()
            if not ret: break
            p = model.estimate(frame, pick="largest")
            if p: preds[cn][fi] = p[0]
        cap.release()

    print("\\n" + "="*80 + "\\nSTAGE 4B: Raw-3D Validation\\n" + "="*80)
    
    errs = {name: [] for name in CRITICAL_JOINTS}
    rcs, bad300 = [], 0
    strong_pairs = {name: 0 for name in CRITICAL_JOINTS}
    gt_frames = {name: 0 for name in CRITICAL_JOINTS}
    
    # Coverage tracking
    triangulated = {name: [False]*N_FRAMES for name in CRITICAL_JOINTS}
    
    for fi in range(N_FRAMES):
        if fi not in gt: continue
        
        # 1. Strong pair check based on 2D GT projection
        for name, jid in CRITICAL_JOINTS.items():
            if np.isnan(gt[fi][jid]).any(): continue
            gt_frames[name] += 1
            sp_found = False
            for c1, c2 in itertools.combinations(CAMS, 2):
                p1, p2 = preds[c1][fi], preds[c2][fi]
                if not p1 or not p2: continue
                rvec1, _ = cv2.Rodrigues(cams[c1]["R"])
                rvec2, _ = cv2.Rodrigues(cams[c2]["R"])
                proj1, _ = cv2.projectPoints(np.array([gt[fi][jid]]), rvec1, cams[c1]["t"], cams[c1]["K"], cams[c1]["distCoef"])
                proj2, _ = cv2.projectPoints(np.array([gt[fi][jid]]), rvec2, cams[c2]["t"], cams[c2]["K"], cams[c2]["distCoef"])
                e1 = np.linalg.norm(p1.keypoints[jid] - proj1[0,0])
                e2 = np.linalg.norm(p2.keypoints[jid] - proj2[0,0])
                
                if e1 <= 20 and e2 <= 20:
                    ang = ray_angle(cams[c1]["C"], cams[c2]["C"], gt[fi][jid])
                    if 30 <= ang <= 150:
                        sp_found = True; break
            if sp_found: strong_pairs[name] += 1
        
        # 2. Triangulate all joints
        pts3d_pred = np.full((17, 3), np.nan)
        for jid in range(17):
            gate = 0.35 if jid in [11,12,13,14,15,16] else 0.5
            pts2d, Ps, scores = [], [], []
            for cn in CAMS:
                p = preds[cn][fi]
                if p and p.scores[jid] > gate:
                    c = cams[cn]
                    corr = cv2.undistortPoints(np.array([[p.keypoints[jid]]], dtype=np.float32), c["K"], c["distCoef"], P=c["K"])
                    pts2d.append(corr[0,0])
                    Ps.append(c["P"])
                    scores.append(p.scores[jid])
            if len(Ps) >= 2:
                pts3d_pred[jid] = triangulate_robust(np.array(pts2d), np.array(Ps), np.array(scores))
                
        # 3. Compute Errors
        for name, jid in CRITICAL_JOINTS.items():
            if not np.isnan(pts3d_pred[jid]).any() and not np.isnan(gt[fi][jid]).any():
                triangulated[name][fi] = True
                e = np.linalg.norm(pts3d_pred[jid] - gt[fi][jid]) * 10.0
                errs[name].append(e)
                if e > 300: bad300 += 1
        
        if not np.isnan(pts3d_pred).all():
            rcs.append(calc_rc_mpjpe(pts3d_pred, gt[fi]))

    print(f"Overall RC-MPJPE: {np.nanmean(rcs) if rcs else 999:.1f} mm")
    print(f"Errors > 300mm: {bad300}\\n")
    
    passed_all = True
    if np.nanmean(rcs) > 60: passed_all = False
    if bad300 > 0: passed_all = False
    
    # Check max unsupported run
    longest_unsupp = {name: 0 for name in CRITICAL_JOINTS}
    for name in CRITICAL_JOINTS:
        run, max_run = 0, 0
        for b in triangulated[name]:
            if not b: run += 1
            else: run = 0
            if run > max_run: max_run = run
        longest_unsupp[name] = max_run
        if max_run > 5: passed_all = False

    for name in ["l_knee", "r_knee", "l_ankle", "r_ankle", "l_wrist", "r_wrist"]:
        m = np.median(errs[name]) if errs[name] else 999
        p95 = np.percentile(errs[name], 95) if errs[name] else 999
        sp_pct = (strong_pairs[name] / max(1, gt_frames[name])) * 100
        cov_pct = (len(errs[name]) / max(1, gt_frames[name])) * 100
        print(f"{name:<8}:")
        print(f"  Median/p95: {m:.1f} mm / {p95:.1f} mm")
        print(f"  Coverage: {cov_pct:.1f}%")
        print(f"  StrongPair: {sp_pct:.1f}%")
        print(f"  Max Unsupp: {longest_unsupp[name]} frames")
        
        if m > 50 or p95 > 100 or cov_pct < 95 or sp_pct < 95:
            passed_all = False
            
    print("\\n" + "="*80 + "\\nFINAL VERDICT\\n" + "="*80)
    if passed_all:
        print("NEW DATASET RAW 3D PASSES — READY TO REVALIDATE STAGE 5")
    else:
        print("NO SUITABLE GT DATASET AVAILABLE — CONTROLLED USER CAPTURE REQUIRED")

if __name__ == "__main__":
    main()
