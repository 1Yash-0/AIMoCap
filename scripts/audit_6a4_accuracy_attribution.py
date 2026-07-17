"""Stage 6a.4 — Attribute and Fix the Four-Camera Accuracy Failure"""

import sys
from collections import defaultdict
from pathlib import Path
import json
import itertools
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.math.triangulate import triangulate_weighted_dlt, triangulate_robust
from aimocap.pose.keypoints import KEYPOINT_NAMES_133

# ── Paths & Config ────────────────────────────────────────────────────────
S3_NPZ_3CAM = ROOT / "outputs" / "stage3_check_c" / "kpts.npz"
S3_NPZ_4TH  = ROOT / "outputs" / "stage6a2_cams" / "kpts.npz"
CALIB_JSON  = ROOT / "data" / "panoptic" / "171204_pose1" / "calibration_171204_pose1.json"
GT_DIR      = ROOT / "data" / "panoptic" / "171204_pose1" / "hdPose3d_stage1_coco19"
OUT_DIR     = ROOT / "outputs" / "stage6a_4_accuracy_attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAM_NAMES_4 = ["00_26", "00_29", "00_30", "00_01"]
START_FRAME = 149
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
    return kpts[:, COCO17_TO_PAN19, :3] # Panoptic is in cm!

def load_calib():
    calib = load_panoptic_calib(CALIB_JSON)
    cams = []
    for cn in CAM_NAMES_4:
        c = calib[cn]
        P = c.K.astype(np.float64) @ np.hstack([c.R.astype(np.float64), c.t.reshape(3, 1).astype(np.float64)])
        cams.append({"name": cn, "K": c.K.astype(np.float64), "R": c.R.astype(np.float64), 
                     "t": c.t.reshape(3, 1).astype(np.float64), "distCoef": c.dist_coef, "P": P})
    return cams

def get_kpts4(cams):
    d3 = np.load(S3_NPZ_3CAM)
    d4 = np.load(S3_NPZ_4TH)
    k3, s3 = d3["keypoints"], d3["scores"]
    k4, s4 = d4["keypoints"][:, 1:2], d4["scores"][:, 1:2]
    k_all = np.concatenate([k3, k4], axis=1)
    s_all = np.concatenate([s3, s4], axis=1)
    
    # APPLY CORRECTION ATTEMPT 1: UNDISTORT 2D DETECTIONS
    # The oracle test proved that ignoring distortion causes 7.8cm error!
    for c_idx, c in enumerate(cams):
        pts = k_all[:, c_idx, :, :] # (F, 133, 2)
        # Reshape to (F*133, 1, 2) for cv2.undistortPoints
        pts_flat = pts.reshape(-1, 1, 2).astype(np.float32)
        undist = cv2.undistortPoints(pts_flat, c["K"], c["distCoef"], P=c["K"])
        k_all[:, c_idx, :, :] = undist.reshape(N_FRAMES, 133, 2)
        
    return k_all, s_all

# ── RC-MPJPE Helper ────────────────────────────────────────────────────────
def calc_rc_mpjpe(pred, gt):
    # Roots
    rp = (pred[11] + pred[12]) / 2.0
    rg = (gt[11] + gt[12]) / 2.0
    diff = (pred - rp) - (gt - rg)
    errs = np.linalg.norm(diff, axis=1) * 10.0 # cm to mm
    return np.nanmean(errs)

# ── Gates ────────────────────────────────────────────────────────
def gate0_unit_contract():
    print("="*60 + "\\nGATE 0: Explicit Unit Contract\\n" + "="*60)
    print("Panoptic GT: cm")
    print("Camera Translation (t): cm")
    print("Triangulated Outputs: cm")
    print("MPJPE calculations: mm (cm * 10)")
    
    gt = load_gt()
    # Check frame 149
    f = 149
    if f < len(gt) and not np.isnan(gt[f, 13]).any():
        hip_knee = np.linalg.norm(gt[f, 11] - gt[f, 13])
        knee_ank = np.linalg.norm(gt[f, 13] - gt[f, 15])
        print(f"Sanity Check (Frame 149):")
        print(f"  Hip-to-Knee Length: {hip_knee:.1f} cm")
        print(f"  Knee-to-Ankle Length: {knee_ank:.1f} cm")
        assert 30 < hip_knee < 60, f"Hip-to-knee {hip_knee} out of bounds! Unit scale error."
        assert 30 < knee_ank < 60, f"Knee-to-ankle {knee_ank} out of bounds! Unit scale error."

def gate1_actual_redundancy(scores):
    print("\\n" + "="*60 + "\\nGATE 1: Actual Redundancy\\n" + "="*60)
    
    for name, jid in CRITICAL_JOINTS:
        gate = 0.35 if jid in [11, 12, 13, 14, 15, 16] else 0.5
        dist = {0:0, 1:0, 2:0, 3:0, 4:0}
        combs = defaultdict(int)
        
        for fi in range(N_FRAMES):
            s2 = scores[fi, :, jid]
            valid = np.where(s2 > gate)[0]
            dist[len(valid)] += 1
            if len(valid) >= 2:
                combs[tuple(valid)] += 1
                
        print(f"{name}:")
        print(f"  Valid Cams: 0:{dist[0]}  1:{dist[1]}  2:{dist[2]}  3:{dist[3]}  4:{dist[4]}")
        
def gate2_oracle_calibration(gt, cams):
    print("\\n" + "="*60 + "\\nGATE 2: Oracle Calibration & Projection Test\\n" + "="*60)
    # 1. Project GT into each camera
    # 2. Triangulate
    errs_dist = []
    errs_nodist = []
    errs_corrected = []
    for fi in range(N_FRAMES):
        f = START_FRAME + fi
        if f >= len(gt) or np.isnan(gt[f, 15]).any(): continue
        gt_pt = gt[f, 15] # l_ankle
        
        pts2d_dist = []
        pts2d_nodist = []
        pts2d_corrected = []
        Ps = []
        for c in cams:
            rvec, _ = cv2.Rodrigues(c["R"])
            proj_d, _ = cv2.projectPoints(np.array([gt_pt]), rvec, c["t"], c["K"], c["distCoef"])
            proj_n, _ = cv2.projectPoints(np.array([gt_pt]), rvec, c["t"], c["K"], np.zeros(5))
            
            p_d = proj_d[0, 0]
            pts2d_dist.append(p_d)
            pts2d_nodist.append(proj_n[0, 0])
            
            # test correction
            corr = cv2.undistortPoints(np.array([[p_d]], dtype=np.float32), c["K"], c["distCoef"], P=c["K"])
            pts2d_corrected.append(corr[0, 0])
            Ps.append(c["P"])
            
        pt3d_dist = triangulate_weighted_dlt(np.array(pts2d_dist), np.array(Ps), np.ones(4))
        pt3d_nodist = triangulate_weighted_dlt(np.array(pts2d_nodist), np.array(Ps), np.ones(4))
        pt3d_corr = triangulate_weighted_dlt(np.array(pts2d_corrected), np.array(Ps), np.ones(4))
        
        errs_dist.append(np.linalg.norm(pt3d_dist - gt_pt) * 10.0) # mm
        errs_nodist.append(np.linalg.norm(pt3d_nodist - gt_pt) * 10.0) # mm
        errs_corrected.append(np.linalg.norm(pt3d_corr - gt_pt) * 10.0) # mm
        
    print(f"Oracle Left Ankle 4-cam (Linear Projection):")
    print(f"  Median Error: {np.median(errs_nodist):.2f} mm")
    print(f"  p95 Error: {np.percentile(errs_nodist, 95):.2f} mm")
    
    print(f"Oracle Left Ankle 4-cam (Distorted Projection -> Linear Triangulation):")
    print(f"  Median Error: {np.median(errs_dist):.2f} mm")
    print(f"  p95 Error: {np.percentile(errs_dist, 95):.2f} mm")
    
    print(f"Oracle Left Ankle 4-cam (Distorted -> Undistorted -> Triangulation):")
    print(f"  Median Error: {np.median(errs_corrected):.2f} mm")
    print(f"  p95 Error: {np.percentile(errs_corrected, 95):.2f} mm")
    
    if np.median(errs_corrected) > 1.0 or np.percentile(errs_corrected, 95) > 2.0:
        print("CALIBRATION/COORDINATE DEFECT STILL PRESENT AFTER CORRECTION")
        sys.exit(1)
        
def gate3_measure_2d_accuracy(gt, kpts2d, scores, cams):
    print("\\n" + "="*60 + "\\nGATE 3: Measure 2D Localization Accuracy\\n" + "="*60)
    for c_idx, c in enumerate(cams):
        print(f"\\nCamera: {c['name']}")
        for name, jid in [("l_knee", 13), ("l_ankle", 15)]:
            gate = 0.35
            errs_px = []
            for fi in range(N_FRAMES):
                f = START_FRAME + fi
                if f >= len(gt) or np.isnan(gt[f, jid]).any(): continue
                if scores[fi, c_idx, jid] < gate: continue
                
                gt_pt = gt[f, jid]
                proj, _ = cv2.projectPoints(np.array([gt_pt]), c["R"], c["t"], c["K"], c["distCoef"])
                gt_2d = proj[0, 0]
                pred_2d = kpts2d[fi, c_idx, jid]
                
                err = np.linalg.norm(pred_2d - gt_2d)
                errs_px.append(err)
            
            if errs_px:
                print(f"  {name}: n={len(errs_px):3d} | Median={np.median(errs_px):4.1f}px | p95={np.percentile(errs_px, 95):4.1f}px | >40px={np.mean(np.array(errs_px)>40)*100:4.1f}%")
            else:
                print(f"  {name}: n=0")

def gate4_pairwise_attribution(gt, kpts2d, scores, cams):
    print("\\n" + "="*60 + "\\nGATE 4: Pairwise & Leave-One-Out Attribution\\n" + "="*60)
    Ps = np.array([c["P"] for c in cams])
    
    pairs = list(itertools.combinations([0, 1, 2, 3], 2))
    trips = list(itertools.combinations([0, 1, 2, 3], 3))
    
    def test_comb(c_indices):
        errs = []
        for fi in range(N_FRAMES):
            f = START_FRAME + fi
            if f >= len(gt) or np.isnan(gt[f, 15]).any(): continue
            s2 = scores[fi, c_indices, 15]
            if (s2 > 0.35).sum() == len(c_indices):
                k2 = kpts2d[fi, c_indices, 15]
                pt3d = triangulate_weighted_dlt(k2, Ps[list(c_indices)], s2)
                errs.append(np.linalg.norm(pt3d - gt[f, 15]) * 10.0)
        if not errs: return 9999, 9999, 0
        return np.median(errs), np.percentile(errs, 95), len(errs)
        
    print("Left Ankle Combinations:")
    for p in pairs:
        med, p95, n = test_comb(p)
        print(f"  Pair {[cams[i]['name'] for i in p]}: n={n} med={med:.1f}mm p95={p95:.1f}mm")
    for t in trips:
        med, p95, n = test_comb(t)
        print(f"  Trip {[cams[i]['name'] for i in t]}: n={n} med={med:.1f}mm p95={p95:.1f}mm")

def gate5_robust_triangulation(gt, kpts2d, scores, cams):
    print("\\n" + "="*60 + "\\nGATE 5: Test Robust Triangulation\\n" + "="*60)
    Ps = np.array([c["P"] for c in cams])
    
    def ransac_triangulate(k_sub, P_sub, s_sub):
        n_v = len(P_sub)
        if n_v < 2: return np.full(3, np.nan)
        if n_v == 2: return triangulate_weighted_dlt(k_sub, P_sub, s_sub)
        
        best_err = np.inf
        best_pt = None
        for pair in itertools.combinations(range(n_v), 2):
            pt = triangulate_weighted_dlt(k_sub[list(pair)], P_sub[list(pair)], s_sub[list(pair)])
            # Project to all views
            pts3d_h = np.append(pt, 1.0)
            proj = (P_sub @ pts3d_h)
            proj = proj[:, :2] / proj[:, 2:3]
            errs = np.linalg.norm(proj - k_sub, axis=1)
            # Use median reprojection error across all views to score this pair
            med_err = np.median(errs)
            if med_err < best_err:
                best_err = med_err
                best_pt = pt
        return best_pt

    def eval_method(method):
        pts3d = np.full((N_FRAMES, 17, 3), np.nan)
        for fi in range(100, N_FRAMES):
            for jid in range(17):
                gate = 0.35 if jid in [11, 12, 13, 14, 15, 16] else 0.5
                s2 = scores[fi, :, jid]
                k2 = kpts2d[fi, :, jid]
                valid = np.where(s2 > gate)[0]
                if len(valid) >= 2:
                    P_sub = Ps[valid]
                    k_sub = k2[valid]
                    s_sub = s2[valid]
                    
                    if method == "weighted":
                        pts3d[fi, jid] = triangulate_weighted_dlt(k_sub, P_sub, s_sub)
                    elif method == "robust":
                        pts3d[fi, jid] = triangulate_robust(k_sub, P_sub, s_sub)
                    elif method == "ransac":
                        pts3d[fi, jid] = ransac_triangulate(k_sub, P_sub, s_sub)
        return pts3d
        
    def get_metrics(pts):
        res = {}
        for name, jid in [("l_knee", 13), ("l_ankle", 15)]:
            errs = []
            for fi in range(100, N_FRAMES):
                pred = pts[fi, jid]
                gt_pt = gt[START_FRAME + fi, jid]
                if not np.isnan(pred).any() and not np.isnan(gt_pt).any():
                    errs.append(np.linalg.norm(pred - gt_pt) * 10.0)
            res[name+"_med"] = np.median(errs) if errs else 999
            res[name+"_p95"] = np.percentile(errs, 95) if errs else 999
        return res
        
    pts_w = eval_method("weighted")
    pts_r = eval_method("robust")
    pts_ran = eval_method("ransac")
    m_w = get_metrics(pts_w)
    m_r = get_metrics(pts_r)
    m_ran = get_metrics(pts_ran)
    
    print(f"{'Metric':<20} | {'Weighted':<15} | {'Robust':<15} | {'RANSAC':<15}")
    for k in m_w:
        print(f"{k:<20} | {m_w[k]:<15.1f} | {m_r[k]:<15.1f} | {m_ran[k]:<15.1f}")
        
    if m_ran["l_ankle_med"] > 50 or m_ran["l_ankle_p95"] > 100 or m_ran["l_knee_med"] > 50 or m_ran["l_knee_p95"] > 100:
        print("\\nBAD CAMERA OBSERVATIONS - ROBUST TRIANGULATION FAILS")
    else:
        print("\\nFOUR-CAMERA RAW 3D PASSES")

def main():
    gate0_unit_contract()
    gt = load_gt()
    cams = load_calib()
    kpts2d, scores = get_kpts4(cams)
    
    gate1_actual_redundancy(scores)
    gate2_oracle_calibration(gt, cams)
    gate3_measure_2d_accuracy(gt, kpts2d, scores, cams)
    gate4_pairwise_attribution(gt, kpts2d, scores, cams)
    gate5_robust_triangulation(gt, kpts2d, scores, cams)

if __name__ == "__main__":
    main()
