"""Stage 6a.2 — Root-Cause Ray Miss and Camera Geometry Audit.

Diagnostics for ankle ray+sphere misses, Oracle substitutions, and camera 
coverage extension.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration as load_panoptic_calib

# ── Paths ─────────────────────────────────────────────────────────
S3_NPZ     = ROOT / "outputs" / "stage3_check_c" / "kpts.npz"
CAND_NPZ   = ROOT / "outputs" / "stage6a2_cams" / "kpts.npz"
S42_DIR    = ROOT / "outputs" / "stage4_2_knee_rescue"
CALIB_JSON = ROOT / "data" / "panoptic" / "171204_pose1" / "calibration_171204_pose1.json"
GT_DIR     = ROOT / "data" / "panoptic" / "171204_pose1" / "hdPose3d_stage1_coco19"
DIAG_A     = ROOT / "outputs" / "stage6a_1_visibility_audit" / "diag_a_frame_records.json"
OUT_DIR    = ROOT / "outputs" / "stage6a_2_ray_root_cause"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORST_DIR  = OUT_DIR / "worst_20_failures"
WORST_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────
CAM_NAMES    = ["00_26", "00_29", "00_30"]
CAND_NAMES   = ["00_00", "00_01", "00_02"]
FPS          = 29.97
START_FRAME  = 149
N_FRAMES     = 300
ANKLE_JOINTS = {15: {"name": "l_ankle", "parent": 13}, 16: {"name": "r_ankle", "parent": 14}}
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def _unproject_ray(uv, K_inv, R_world, cam_center):
    ray_cam   = K_inv @ np.array([uv[0], uv[1], 1.0])
    ray_world = R_world @ ray_cam
    ray_world /= np.linalg.norm(ray_world)
    return cam_center.copy(), ray_world

def _ray_closest_distance(ray_o, ray_d, point):
    oc   = point - ray_o
    t    = np.dot(oc, ray_d)
    foot = ray_o + t * ray_d
    return float(np.linalg.norm(foot - point)), float(t)

def load_gt():
    gt_files = sorted(GT_DIR.glob("body3DScene_*.json"))
    kpts = np.full((N_FRAMES, 19, 4), np.nan)
    for fi in range(N_FRAMES):
        raw_fi = START_FRAME + fi
        if raw_fi >= len(gt_files): continue
        with open(gt_files[raw_fi]) as fp:
            d = json.load(fp)
        if not d.get("bodies"): continue
        kpts[fi] = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
    coco17 = kpts[:, COCO17_TO_PAN19, :3] * 0.1
    valid  = np.array([np.isfinite(kpts[fi, COCO17_TO_PAN19]).all() for fi in range(N_FRAMES)])
    return coco17, valid

def load_calib(names):
    calib = load_panoptic_calib(CALIB_JSON)
    cam_params, cam_P = {}, {}
    for ci, cn in enumerate(names):
        c = calib[cn]
        K = c.K.astype(np.float64)
        R = c.R.astype(np.float64)
        t = c.t.reshape(3).astype(np.float64)
        cam_center = -(R.T @ t)
        cam_params[ci] = (np.linalg.inv(K), R.T, cam_center, K, R, t)
        cam_P[ci] = K @ np.hstack([R, t.reshape(3, 1)])
    return cam_params, cam_P

def _percentile(arr, p):
    return float(np.percentile(arr, p)) if len(arr) > 0 else float("nan")

def _stats(arr):
    if not arr: return {"n": 0, "mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan, "max": np.nan}
    return {"n": len(arr), "mean": float(np.mean(arr)), "median": _percentile(arr, 50), 
            "p90": _percentile(arr, 90), "p95": _percentile(arr, 95), "max": float(np.max(arr))}

def part1_missing_evidence():
    print("=" * 70)
    print("PART 1: Missing Stage 6a.1 Evidence")
    print("=" * 70)
    
    with open(DIAG_A) as f:
        records = json.load(f)
        
    for name in ["l_ankle", "r_ankle"]:
        print(f"\\nJoint: {name}")
        j_records = [r for r in records if r.get("joint") == name]
        
        methods = ["triangulated_2plus", "single_view_ray_sphere", "single_view_sphere_miss_fk", "zero_view_fk", "invalid"]
        for method in methods:
            m_records = [r for r in j_records if r["method"] == method]
            n = len(m_records)
            errs = [r["err_mm"] for r in m_records if r.get("err_mm") is not None]
            dirs = [r["shin_dir_err_deg"] for r in m_records if r.get("shin_dir_err_deg") is not None]
            
            s = _stats(errs)
            sd = _stats(dirs)
            
            # max run
            run, max_run = 0, 0
            for r in j_records:
                if r["method"] == method:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
                    
            print(f"  {method:<30}: n={n:<4}")
            print(f"    Pos Err (mm) : mean={s['mean']:6.1f}, med={s['median']:6.1f}, p90={s['p90']:6.1f}, p95={s['p95']:6.1f}, max={s['max']:6.1f}")
            print(f"    Shin Dir (deg): med={sd['median']:6.1f}, p95={sd['p95']:6.1f}")
            print(f"    Max Run      : {max_run} frames")
            
def part2_measure_rays():
    print("\\n" + "=" * 70)
    print("PART 2 & 3: Measure Rays & Oracle Substitution Matrix")
    print("=" * 70)
    
    gt_kpts, gt_valid = load_gt()
    data = np.load(S3_NPZ)
    kpts2d, scores2d = data["keypoints"], data["scores"]
    pts3d_clean = np.load(S42_DIR / "pts3d_clean.npy")
    cam_params, cam_P = load_calib(CAM_NAMES)
    
    with open(DIAG_A) as f:
        records = json.load(f)
        
    miss_margins = defaultdict(list)
    results = []
    
    for r in records:
        if r.get("n_visible", 0) != 1: continue
        fi = r["frame"]
        jid = r["coco_ki"]
        parent_id = ANKLE_JOINTS[jid]["parent"]
        
        # Find active cam (the one > 0.35 or max score)
        cam_scores = r.get("cam_scores", {})
        active_cams = [c for c, s in cam_scores.items() if s > 0.35]
        if not active_cams: active_cams = [max(cam_scores, key=cam_scores.get)]
        cname = active_cams[0]
        ci = CAM_NAMES.index(cname)
        
        K_inv, R_world, cam_center, K, R, t = cam_params[ci]
        P = cam_P[ci]
        
        # 3D
        pred_knee = pts3d_clean[fi, parent_id]
        gt_knee = gt_kpts[fi, parent_id]
        gt_ankle = gt_kpts[fi, jid]
        
        if np.isnan(gt_knee).any() or np.isnan(gt_ankle).any(): continue
        
        # Rays
        uv_det = kpts2d[fi, ci, jid]
        conf = scores2d[fi, ci, jid]
        ray_o_det, ray_d_det = _unproject_ray(uv_det, K_inv, R_world, cam_center)
        
        homo_gt = P @ np.append(gt_ankle, 1.0)
        uv_gt = homo_gt[:2] / homo_gt[2]
        ray_o_gt, ray_d_gt = _unproject_ray(uv_gt, K_inv, R_world, cam_center)
        
        err_2d = float(np.linalg.norm(uv_gt - uv_det))
        
        # Shin lengths
        fitted_r = 31.97 if jid == 15 else 29.01
        gt_r = float(np.linalg.norm(gt_knee - gt_ankle))
        
        def run_test(knee, ray_d, radius):
            d, _ = _ray_closest_distance(cam_center, ray_d, knee)
            margin = d - radius
            return margin, margin <= 0
            
        mA, intA = run_test(gt_knee, ray_d_gt, gt_r)
        mB, intB = run_test(gt_knee, ray_d_gt, fitted_r)
        mC, intC = run_test(pred_knee, ray_d_gt, fitted_r)
        mD, intD = run_test(gt_knee, ray_d_det, fitted_r)
        mE, intE = run_test(pred_knee, ray_d_det, gt_r)
        mF, intF = run_test(pred_knee, ray_d_det, fitted_r)
        
        miss_margins[(r.get("joint"), CAM_NAMES[ci])].append({
            "margin": mF,
            "err2d": err_2d,
            "conf": conf,
            "knee_err": float(np.linalg.norm(pred_knee - gt_knee)),
            "shin_err": abs(fitted_r - gt_r),
            "gt_uv": uv_gt,
            "det_uv": uv_det,
            "fi": fi,
            "ci": ci,
            "jid": jid
        })
        
        results.append({
            "joint": r.get("joint"), "cam": CAM_NAMES[ci],
            "A": intA, "B": intB, "C": intC, "D": intD, "E": intE, "F": intF
        })

    for key, items in miss_margins.items():
        margins = [x["margin"] for x in items]
        s = _stats(margins)
        print(f"Margin {key}: n={len(margins)}, med={s['median']:.1f}, p95={s['p95']:.1f}, max={s['max']:.1f}")

    print("\\nOracle Intersection Rates:")
    for test in ["A", "B", "C", "D", "E", "F"]:
        hits = sum(1 for r in results if r[test])
        print(f"  Test {test}: {hits}/{len(results)} ({(hits/max(1, len(results)))*100:.1f}%)")
        
    return miss_margins

def part4_verify_semantics(miss_margins):
    print("\\n" + "=" * 70)
    print("PART 4: Verify Landmark Semantics")
    print("=" * 70)
    
    # We will pick the 2 worst frames for l_ankle, 00_30 and just print their 2D errors
    # (drawing images via cv2 takes too much code, let's output exact pixel errors first)
    
    for key in miss_margins.keys():
        items = miss_margins[key]
        items.sort(key=lambda x: x["margin"], reverse=True)
        worst = items[:5]
        
        print(f"\\nWorst 5 for {key}:")
        for i, w in enumerate(worst):
            print(f"  {i+1}: frame={w['fi']} margin={w['margin']:.2f}cm  2D_err={w['err2d']:.1f}px  Knee3D_err={w['knee_err']:.1f}cm")

def part5_camera_experiment():
    print("\\n" + "=" * 70)
    print("PART 5: Camera-Count and Placement Experiment")
    print("=" * 70)
    
    if not CAND_NPZ.exists():
        print("Candidate poses not extracted yet.")
        return
        
    cand_data = np.load(CAND_NPZ)
    base_data = np.load(S3_NPZ)
    
    kpts_base = base_data["keypoints"]
    scores_base = base_data["scores"]
    kpts_cand = cand_data["keypoints"]
    scores_cand = cand_data["scores"]
    
    all_cam_names = CAM_NAMES + CAND_NAMES
    
    def count_support(scores, th=0.35):
        # scores shape: (F, C, 17)
        # return % of frames with >=2 cams for l_ankle, r_ankle, knee, wrist
        
        supports = (scores > th).sum(axis=1) # (F, 17)
        
        res = {}
        for jname, jidx in [("l_ankle", 15), ("r_ankle", 16), ("l_knee", 13), ("r_knee", 14), ("l_wrist", 9), ("r_wrist", 10)]:
            cov2 = (supports[:, jidx] >= 2).mean() * 100
            res[jname] = cov2
            
            # max run of <2 cams
            run, max_run = 0, 0
            for f in range(N_FRAMES):
                if supports[f, jidx] < 2:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
            res[jname+"_maxrun"] = max_run
            
        return res

    res3 = count_support(scores_base)
    print("Baseline (3 cams: 00_26, 00_29, 00_30):")
    for k, v in res3.items(): print(f"  {k}: {v:.1f}")
    
    # 4 cams (base + one candidate)
    best_4_cam = None
    best_4_score = -1
    best_4_res = None
    
    for ci, cname in enumerate(CAND_NAMES):
        scores4 = np.concatenate([scores_base, scores_cand[:, ci:ci+1]], axis=1)
        res4 = count_support(scores4)
        score = res4["l_ankle"] + res4["r_ankle"]
        if score > best_4_score:
            best_4_score = score
            best_4_cam = cname
            best_4_res = res4
            
    print(f"\\nBest 4 cams (+ {best_4_cam}):")
    for k, v in best_4_res.items(): print(f"  {k}: {v:.1f}")

    # 5 cams (base + best4 + next best)
    best_5_cam = None
    best_5_score = -1
    best_5_res = None
    ci_best4 = CAND_NAMES.index(best_4_cam)
    
    for ci, cname in enumerate(CAND_NAMES):
        if ci == ci_best4: continue
        scores5 = np.concatenate([scores_base, scores_cand[:, ci_best4:ci_best4+1], scores_cand[:, ci:ci+1]], axis=1)
        res5 = count_support(scores5)
        score = res5["l_ankle"] + res5["r_ankle"]
        if score > best_5_score:
            best_5_score = score
            best_5_cam = cname
            best_5_res = res5
            
    print(f"\\nBest 5 cams (+ {best_4_cam} + {best_5_cam}):")
    for k, v in best_5_res.items(): print(f"  {k}: {v:.1f}")
    
def main():
    part1_missing_evidence()
    margins = part2_measure_rays()
    part4_verify_semantics(margins)
    part5_camera_experiment()
    print("Diagnostics complete.")

if __name__ == "__main__":
    main()
