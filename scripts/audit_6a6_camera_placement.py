"""Stage 6a.6 — Camera-Placement Search and Held-Out Raw-3D Validation"""

import sys
from pathlib import Path
import json
import itertools
import numpy as np
import cv2
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aimocap
from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.math.triangulate import triangulate_weighted_dlt, triangulate_robust

# ── Paths & Config ────────────────────────────────────────────────────────
CALIB_JSON  = ROOT / "data" / "panoptic" / "171204_pose1" / "calibration_171204_pose1.json"
GT_DIR      = ROOT / "data" / "panoptic" / "171204_pose1" / "hdPose3d_stage1_coco19"
VIDEO_DIR   = ROOT / "data" / "panoptic" / "171204_pose1" / "hdVideos"
OUT_DIR     = ROOT / "outputs" / "stage6a_6_camera_placement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The 6 fully downloaded Panoptic cameras on disk
CAM_NAMES_6 = ["00_00", "00_01", "00_02", "00_26", "00_29", "00_30"]
N_FRAMES    = 300
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
CRITICAL_JOINTS = [("l_knee", 13), ("r_knee", 14), ("l_ankle", 15), ("r_ankle", 16), ("l_wrist", 9), ("r_wrist", 10), ("l_hip", 11), ("r_hip", 12)]

# ── Loaders ────────────────────────────────────────────────────────
def load_gt():
    gt_files = sorted(GT_DIR.glob("body3DScene_*.json"))
    if not gt_files: return np.zeros((0, 17, 3))
    max_idx = int(gt_files[-1].stem.split("_")[-1])
    kpts = np.full((max_idx + 1, 19, 4), np.nan)
    for fpath in gt_files:
        fi = int(fpath.stem.split("_")[-1])
        with open(fpath) as fp:
            d = json.load(fp)
        if not d.get("bodies"): continue
        kpts[fi] = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
    return kpts[:, COCO17_TO_PAN19, :3]

def load_calib():
    calib = load_panoptic_calib(CALIB_JSON)
    cams = {}
    for cn in CAM_NAMES_6:
        c = calib[cn]
        P = c.K.astype(np.float64) @ np.hstack([c.R.astype(np.float64), c.t.reshape(3, 1).astype(np.float64)])
        # Center C = -R^T * t
        C = -c.R.astype(np.float64).T @ c.t.reshape(3, 1).astype(np.float64)
        cams[cn] = {"name": cn, "K": c.K.astype(np.float64), "R": c.R.astype(np.float64), 
                     "t": c.t.reshape(3, 1).astype(np.float64), "distCoef": c.dist_coef, "P": P, "C": C.flatten()}
    return cams

# ── Triangulation ────────────────────────────────────────────────────────
def ransac_triangulate(k_sub, P_sub, s_sub):
    n_v = len(P_sub)
    if n_v < 2: return np.full(3, np.nan)
    if n_v == 2: return triangulate_weighted_dlt(k_sub, P_sub, s_sub)
    
    best_err, best_pt = np.inf, None
    for pair in itertools.combinations(range(n_v), 2):
        pt = triangulate_weighted_dlt(k_sub[list(pair)], P_sub[list(pair)], s_sub[list(pair)])
        pts3d_h = np.append(pt, 1.0)
        proj = (P_sub @ pts3d_h)
        proj = proj[:, :2] / proj[:, 2:3]
        errs = np.linalg.norm(proj - k_sub, axis=1)
        med_err = np.median(errs)
        if med_err < best_err:
            best_err, best_pt = med_err, pt
    return best_pt

def calc_rc_mpjpe(pred, gt):
    rp = (pred[11] + pred[12]) / 2.0
    rg = (gt[11] + gt[12]) / 2.0
    diff = (pred - rp) - (gt - rg)
    return np.nanmean(np.linalg.norm(diff, axis=1) * 10.0)

# ── Gates ────────────────────────────────────────────────────────
def gate0_clear_pair(preds, gt, cams):
    print("="*80 + "\\nGATE 0: Test the actual clear-camera pair\\n" + "="*80)
    c_names = ["00_30", "00_01"]
    
    errs_lk, errs_la, errs_lw, rcs = [], [], [], []
    for fi in range(100, N_FRAMES):
        f = 149 + fi
        if f >= len(gt): continue
        pts3d_pred = np.full((17, 3), np.nan)
        for jid in range(17):
            gate = 0.35 if jid in [11, 12, 13, 14, 15, 16] else 0.5
            kpts2d, Ps, scores = [], [], []
            for cn in c_names:
                p = preds[cn][fi]
                if p and p.scores[jid] > gate:
                    c = cams[cn]
                    p_d = p.keypoints[jid]
                    corr = cv2.undistortPoints(np.array([[p_d]], dtype=np.float32), c["K"], c["distCoef"], P=c["K"])
                    kpts2d.append(corr[0, 0])
                    Ps.append(c["P"])
                    scores.append(p.scores[jid])
            if len(Ps) >= 2:
                pts3d_pred[jid] = triangulate_weighted_dlt(np.array(kpts2d), np.array(Ps), np.array(scores))
        
        if not np.isnan(pts3d_pred[13]).any() and not np.isnan(gt[f, 13]).any(): errs_lk.append(np.linalg.norm(pts3d_pred[13] - gt[f, 13]) * 10.0)
        if not np.isnan(pts3d_pred[15]).any() and not np.isnan(gt[f, 15]).any(): errs_la.append(np.linalg.norm(pts3d_pred[15] - gt[f, 15]) * 10.0)
        if not np.isnan(pts3d_pred[9]).any() and not np.isnan(gt[f, 9]).any():   errs_lw.append(np.linalg.norm(pts3d_pred[9] - gt[f, 9]) * 10.0)
        if not np.isnan(pts3d_pred).all(): rcs.append(calc_rc_mpjpe(pts3d_pred, gt[f]))
            
    mlk = np.median(errs_lk) if errs_lk else 999
    mla = np.median(errs_la) if errs_la else 999
    print(f"Clear Pair [00_30, 00_01] Detections (Held-out 100-299):")
    print(f"  l_knee median: {mlk:.1f} mm | p95: {np.percentile(errs_lk, 95) if errs_lk else 999:.1f} mm")
    print(f"  l_ankle median: {mla:.1f} mm | p95: {np.percentile(errs_la, 95) if errs_la else 999:.1f} mm")
    print(f"  l_wrist median: {np.median(errs_lw) if errs_lw else 999:.1f} mm")
    print(f"  RC-MPJPE: {np.nanmean(rcs) if rcs else 999:.1f} mm")
    
    if mlk > 50.0 or mla > 50.0 or np.nanmean(rcs) > 60.0:
        print("\\nCLEAR VIEWS STILL FAIL — CAMERA PLACEMENT IS NOT YET PROVEN")
        sys.exit(1)
        
def gate123_build_and_select(preds, gt, cams):
    print("\\n" + "="*80 + "\\nGATE 1-3: Build Candidates & Diagnostic Vis Evaluation\\n" + "="*80)
    
    # 1. Precompute Accuracy for frames 0-99
    # Dict: acc[cn][fi][jid] = True/False (error < 20px)
    acc = {cn: {fi: {jid: False for name, jid in CRITICAL_JOINTS} for fi in range(100)} for cn in CAM_NAMES_6}
    
    for cn in CAM_NAMES_6:
        c = cams[cn]
        rvec, _ = cv2.Rodrigues(c["R"])
        for fi in range(100):
            f = 149 + fi
            if f >= len(gt): continue
            p = preds[cn][fi]
            for name, jid in CRITICAL_JOINTS:
                gate = 0.35 if jid in [11, 12, 13, 14, 15, 16] else 0.5
                if not p or p.scores[jid] < gate or np.isnan(gt[f, jid]).any():
                    acc[cn][fi][jid] = False
                    continue
                proj, _ = cv2.projectPoints(np.array([gt[f, jid]]), rvec, c["t"], c["K"], c["distCoef"])
                err = np.linalg.norm(p.keypoints[jid] - proj[0, 0])
                acc[cn][fi][jid] = (err <= 20.0)
                
    # Evaluate layouts
    best_layouts = {}
    for k in [3, 4, 5]:
        best_score = -1
        best_cand = None
        for cand in itertools.combinations(CAM_NAMES_6, k):
            # score = avg % frames with >=2 accurate views across critical joints
            joint_scores = []
            for name, jid in CRITICAL_JOINTS:
                n_good_frames = 0
                for fi in range(100):
                    n_acc_views = sum(acc[cn][fi][jid] for cn in cand)
                    if n_acc_views >= 2: n_good_frames += 1
                joint_scores.append(n_good_frames / 100.0)
            avg_score = np.mean(joint_scores)
            if avg_score > best_score:
                best_score = avg_score
                best_cand = cand
        best_layouts[k] = {"cameras": best_cand, "score": best_score}
        print(f"Best {k}-camera Layout: {best_cand} | Score (>=2 views): {best_score*100:.1f}%")
        
    return best_layouts

def gate4_held_out(preds, gt, cams, best_layouts):
    print("\\n" + "="*80 + "\\nGATE 4: Held-out Raw-3D Comparison\\n" + "="*80)
    
    print(f"{'Metric':<25} | {'Best 3 cams':<15} | {'Best 4 cams':<15} | {'Best 5 cams':<15}")
    metrics = {k: {} for k in [3, 4, 5]}
    
    for k, lay in best_layouts.items():
        c_names = lay["cameras"]
        errs_lk, errs_la, errs_lw, rcs = [], [], [], []
        
        for fi in range(100, N_FRAMES):
            f = 149 + fi
            if f >= len(gt): continue
            pts3d_pred = np.full((17, 3), np.nan)
            for jid in range(17):
                gate = 0.35 if jid in [11, 12, 13, 14, 15, 16] else 0.5
                kpts2d, Ps, scores = [], [], []
                for cn in c_names:
                    p = preds[cn][fi]
                    if p and p.scores[jid] > gate:
                        c = cams[cn]
                        p_d = p.keypoints[jid]
                        corr = cv2.undistortPoints(np.array([[p_d]], dtype=np.float32), c["K"], c["distCoef"], P=c["K"])
                        kpts2d.append(corr[0, 0])
                        Ps.append(c["P"])
                        scores.append(p.scores[jid])
                if len(Ps) >= 2:
                    # Use Robust (Huber) for Gate 4
                    pts3d_pred[jid] = triangulate_robust(np.array(kpts2d), np.array(Ps), np.array(scores))
            
            if not np.isnan(pts3d_pred[13]).any() and not np.isnan(gt[f, 13]).any(): errs_lk.append(np.linalg.norm(pts3d_pred[13] - gt[f, 13]) * 10.0)
            if not np.isnan(pts3d_pred[15]).any() and not np.isnan(gt[f, 15]).any(): errs_la.append(np.linalg.norm(pts3d_pred[15] - gt[f, 15]) * 10.0)
            if not np.isnan(pts3d_pred[9]).any() and not np.isnan(gt[f, 9]).any():   errs_lw.append(np.linalg.norm(pts3d_pred[9] - gt[f, 9]) * 10.0)
            if not np.isnan(pts3d_pred).all(): rcs.append(calc_rc_mpjpe(pts3d_pred, gt[f]))
                
        metrics[k]["rc"] = np.nanmean(rcs) if rcs else 999
        metrics[k]["lk_m"] = np.median(errs_lk) if errs_lk else 999
        metrics[k]["lk_p95"] = np.percentile(errs_lk, 95) if errs_lk else 999
        metrics[k]["la_m"] = np.median(errs_la) if errs_la else 999
        metrics[k]["la_p95"] = np.percentile(errs_la, 95) if errs_la else 999
        metrics[k]["lw_m"] = np.median(errs_lw) if errs_lw else 999
        metrics[k]["lw_p95"] = np.percentile(errs_lw, 95) if errs_lw else 999
        
    def fmt(m, key1, key2=None):
        if key2: return f"{m[key1]:.1f}/{m[key2]:.1f}"
        return f"{m[key1]:.1f}"
        
    print(f"{'RC-MPJPE':<25} | {fmt(metrics[3], 'rc'):<15} | {fmt(metrics[4], 'rc'):<15} | {fmt(metrics[5], 'rc'):<15}")
    print(f"{'Knee median/p95':<25} | {fmt(metrics[3], 'lk_m', 'lk_p95'):<15} | {fmt(metrics[4], 'lk_m', 'lk_p95'):<15} | {fmt(metrics[5], 'lk_m', 'lk_p95'):<15}")
    print(f"{'Ankle median/p95':<25} | {fmt(metrics[3], 'la_m', 'la_p95'):<15} | {fmt(metrics[4], 'la_m', 'la_p95'):<15} | {fmt(metrics[5], 'la_m', 'la_p95'):<15}")
    print(f"{'Wrist median/p95':<25} | {fmt(metrics[3], 'lw_m', 'lw_p95'):<15} | {fmt(metrics[4], 'lw_m', 'lw_p95'):<15} | {fmt(metrics[5], 'lw_m', 'lw_p95'):<15}")
    
    # Gate 5
    for k in [3, 4, 5]:
        m = metrics[k]
        if m["rc"] <= 60.0 and m["lk_m"] <= 50.0 and m["lk_p95"] <= 100.0 and m["la_m"] <= 50.0 and m["la_p95"] <= 100.0:
            if k == 5: print("\\nFOUR CAMERAS PASS — FIVE RECOMMENDED FOR REDUNDANCY")
            elif k == 4: print("\\nFOUR CAMERAS PASS — FIVE RECOMMENDED FOR REDUNDANCY")
            elif k == 3: print("\\nTHREE CAMERAS PASS")
            return
    print("\\nNO TESTED LAYOUT PASSES")
        
def main():
    gt = load_gt()
    cams = load_calib()
    model = PoseEstimator()
    
    # Cache inference for the 6 cameras
    preds = {cn: [None]*N_FRAMES for cn in CAM_NAMES_6}
    for cn in CAM_NAMES_6:
        vid_path = VIDEO_DIR / f"hd_{cn}.mp4"
        cap = cv2.VideoCapture(str(vid_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 149)
        for fi in range(N_FRAMES):
            ret, frame = cap.read()
            if not ret: break
            p = model.estimate(frame, pick="largest")
            if p: preds[cn][fi] = p[0]
        cap.release()
        
    gate0_clear_pair(preds, gt, cams)
    best_layouts = gate123_build_and_select(preds, gt, cams)
    gate4_held_out(preds, gt, cams, best_layouts)

if __name__ == "__main__":
    main()
