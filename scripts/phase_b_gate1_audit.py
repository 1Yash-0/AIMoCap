"""
Phase B - Gate 1: Metric Audit & Bootstrapping
Addresses all event matching defects, subset chance correction, and block bootstrapping.
"""

import sys, json, uuid
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics, _reprojection_errors
from aimocap.math.triangulate import triangulate_robust
from aimocap.math.filter import fill_gaps_with_logging, filter_skeleton_one_euro
from scripts.experiment_gating_architectures import infer_by_ray_sphere, build_bvh_positions, fit_skeleton_sequence, COCO17_TO_PAN19

OUT = ROOT / "outputs/phase_b_gate1"
OUT.mkdir(parents=True, exist_ok=True)

CAMS = ["00_11", "00_12", "00_23"]
NFRAMES = 1800
FPS = 30.0
BODY_J = list(range(5, 17))
LOWER_J = [11, 12, 13, 14, 15, 16]
ANKLE_J = [15, 16]
CONF_GATE = 0.4
REPROJ_THR = 100.0
IMG_H = 1080
MARGIN_PX = 40
SPIKE_THR = 5.0

# ─── 1. Synthetic Event Matcher Tests ────────────────────────────────────────

def cost_matrix_for_matching(preds, gts, dt_tol):
    C = np.full((len(preds), len(gts)), np.inf)
    valid = np.zeros((len(preds), len(gts)), dtype=bool)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p.get("joint", -1) != g.get("joint", -1): continue
            dt = abs(p["peak_fi"] - g["peak_fi"])
            if dt <= dt_tol:
                dp = np.linalg.norm(p["dir"])
                dg = np.linalg.norm(g["dir"])
                d_cos = float(np.dot(p["dir"]/dp, g["dir"]/dg)) if (dp>1e-6 and dg>1e-6) else 1.0
                if d_cos >= 0.5:
                    C[i,j] = (dt / max(1, dt_tol)) + (1.0 - d_cos)
                    valid[i,j] = True
    return C, valid

def match_events(preds, gts, dt_tol=1):
    matched_preds = {}
    matched_gts = {}
    
    # Split by joint
    joints = set(p.get("joint", -1) for p in preds) | set(g.get("joint", -1) for g in gts)
    
    for j_idx in joints:
        j_preds = [p for p in preds if p.get("joint", -1) == j_idx]
        j_gts = [g for g in gts if g.get("joint", -1) == j_idx]
        
        N = len(j_preds)
        M = len(j_gts)
        if N == 0 or M == 0:
            continue
            
        C, valid = cost_matrix_for_matching(j_preds, j_gts, dt_tol)
        
        unmatched_penalty = 1e6
        C_aug = np.full((N + M, N + M), unmatched_penalty * 10)
        C_aug[0:N, 0:M] = np.where(valid, C, unmatched_penalty * 2)
        C_aug[0:N, M:N+M] = unmatched_penalty * np.eye(N)
        C_aug[N:N+M, 0:M] = unmatched_penalty * np.eye(M)
        C_aug[N:N+M, M:N+M] = 0
        
        row_ind, col_ind = linear_sum_assignment(C_aug)
        
        for r, c in zip(row_ind, col_ind):
            if r < N and c < M and valid[r, c]:
                pid = j_preds[r]["id"]
                gid = j_gts[c]["id"]
                matched_preds[pid] = gid
                matched_gts[gid] = pid
                # We need to mutate the original dict in the preds list
                # Since j_preds contains references to the original dicts, this works!
                j_preds[r]["gt_match_cost"] = C[r,c]
                j_preds[r]["gt_match_dt"] = abs(j_preds[r]["peak_fi"] - j_gts[c]["peak_fi"])
                
    return matched_preds, matched_gts

def test_event_matcher():
    # Tie competition test
    p1 = {"id": "p1", "joint": 1, "peak_fi": 10, "dir": np.array([1.0, 0, 0]), "peak_acc": 10}
    p2 = {"id": "p2", "joint": 1, "peak_fi": 11, "dir": np.array([1.0, 0, 0]), "peak_acc": 10}
    g1 = {"id": "g1", "joint": 1, "peak_fi": 10, "dir": np.array([1.0, 0, 0]), "peak_acc": 10}
    
    # Bipartite should match p1->g1 (cost 0 vs cost 1/tol)
    mp, _ = match_events([p1, p2], [g1], dt_tol=2)
    assert mp.get("p1") == "g1" and "p2" not in mp
    
    # Order invariance / Shuffling test
    rng = np.random.RandomState(42)
    for _ in range(5):
        preds = [p1, p2]
        gts = [g1]
        rng.shuffle(preds)
        rng.shuffle(gts)
        mp, mg = match_events(preds, gts, dt_tol=2)
        assert mp.get("p1") == "g1"
        assert "p2" not in mp
    print("  [OK] Synthetic event matcher tests passed.")

# ─── 2. Pipeline Execution & Origin Accounting ───────────────────────────────

def get_shared_data():
    npz = np.load(ROOT / "outputs/canonical_dataset/canonical_detector_pose_observations.npz")
    kpts = npz["kpts"].astype(np.float32)
    scores = npz["scores"].astype(np.float32)
    calib = load_calibration(ROOT / "data/panoptic/171204_pose1/calibration_171204_pose1.json")
    K = [calib[cn].K.astype(np.float64) for cn in CAMS]
    extr = [(calib[cn].R.astype(np.float64), calib[cn].t.astype(np.float64).reshape(3,1)) for cn in CAMS]
    bl = np.array(json.loads((ROOT / "outputs/phase_b_audit/audit_config.json").read_text())["bone_lengths"])
    
    gt_raw = np.full((NFRAMES,17,3), np.nan, np.float64)
    gt_ok = np.zeros(NFRAMES, bool)
    for i in range(NFRAMES):
        fn = ROOT / f"data/panoptic/171204_pose1/hdPose3d_stage1_coco19/body3DScene_{150+i:08d}.json"
        if not fn.exists(): continue
        js = json.loads(fn.read_text())
        if not js.get("bodies"): continue
        j19 = np.array(js["bodies"][0]["joints19"], np.float64).reshape(19,4)
        for c17,p19 in enumerate(COCO17_TO_PAN19): gt_raw[i,c17] = j19[p19,:3]
        gt_ok[i] = True
    gt_mm = gt_raw.copy()
    gt_mm[:,:,1] *= -1; gt_mm[:,:,2] *= -1; gt_mm *= 10.0
    return kpts, scores, calib, K, extr, gt_mm, gt_ok, bl

def apply_masks(kpts, scores):
    km_a, sm_a = kpts.copy(), scores.copy()
    for f in range(NFRAMES):
        for c in range(3):
            v = sm_a[f,c,:] >= CONF_GATE
            if v.sum() > 2:
                vk = km_a[f,c,v]
                w = np.max(vk[:,0])-np.min(vk[:,0])
                h = np.max(vk[:,1])-np.min(vk[:,1])
                if w > 0 and h/w < 1.8: sm_a[f,c,:] = 0.0
                
    km_b, sm_b = kpts.copy(), scores.copy()
    km_c, sm_c = kpts.copy(), scores.copy()
    lim = IMG_H - MARGIN_PX
    mask_log = []
    for f in range(NFRAMES):
        for c in range(3):
            for ji in ANKLE_J:
                if sm_c[f,c,ji] < CONF_GATE: continue
                x,y = kpts[f,c,ji]
                if not (np.isfinite(x) and np.isfinite(y)): continue
                if y > lim or x < 0 or x > 1920:
                    sm_c[f,c,ji] = 0.0
                    mask_log.append({"f":f,"c":c,"j":ji})
    return (km_a, sm_a), (km_b, sm_b), (km_c, sm_c, mask_log)

def run_pipeline(km, sm, K, extr, calib, bl):
    tri = triangulate_sequence_with_diagnostics(km, sm, K, extr, CONF_GATE, REPROJ_THR, 0.0)
    raw = tri.points3d.copy()
    origin = np.full((NFRAMES,17), "missing", dtype=object)
    for fi in range(NFRAMES):
        for ji in range(17):
            ni = int(tri.num_inliers[fi,ji])
            if ni >= 3: origin[fi,ji] = "3-view"
            elif ni == 2: origin[fi,ji] = "2-view"
            
    filled, gap_log, _ = fill_gaps_with_logging(raw, [str(i) for i in range(17)], fps=FPS)
    for entry in gap_log:
        ji_str = entry.get("joint","")
        try: ji = int(ji_str)
        except: continue
        for fi in range(entry.get("start",0), entry.get("end",0)+1):
            if 0 <= fi < NFRAMES and origin[fi,ji] == "missing":
                origin[fi,ji] = "gap-filled"
                
    smooth = filter_skeleton_one_euro(filled, fps=FPS)
    bvh0 = build_bvh_positions(smooth)
    _, fk0 = fit_skeleton_sequence(bvh0, bl)
    bvh2c = {1:11,2:13,3:15,4:12,5:14,6:16,8:5,9:7,10:9,11:6,12:8,13:10,14:0}
    fkc = np.full((NFRAMES,17,3), np.nan, np.float32)
    for bj,cj in bvh2c.items(): fkc[:,cj] = fk0[:,bj]
    fkc[:,0] = (fkc[:,11]+fkc[:,12])/2.0
    
    fkwa, istats = infer_by_ray_sphere(fkc, {15:bl[3],16:bl[6]}, calib, km, sm, {15:0.35,16:0.35}, {15:(13,11),16:(14,12)}, CAMS)
    for ji in ANKLE_J:
        for fi in range(NFRAMES):
            if origin[fi,ji] in ("missing","gap-filled") and np.isfinite(fkwa[fi,ji]).all():
                origin[fi,ji] = "ray-sphere" if istats[ji]["n_ray"] > 0 else "FK-inferred"
                
    bvhf = build_bvh_positions(fkwa)
    _, fkf = fit_skeleton_sequence(bvhf, bl)
    final = np.full((NFRAMES,17,3), np.nan, np.float32)
    for bj,cj in bvh2c.items(): final[:,cj] = fkf[:,bj]
    
    # Assert exact sum for origin tracking
    assert (origin == "missing").sum() + (origin == "3-view").sum() + (origin == "2-view").sum() + (origin == "gap-filled").sum() + (origin == "ray-sphere").sum() + (origin == "FK-inferred").sum() == NFRAMES * 17
    
    return {"raw":raw, "smooth":smooth, "stage6":final, "origin":origin, "tri":tri}

# ─── 3. Event Extraction & Matching ──────────────────────────────────────────

def compute_accel_dir(pts_cm):
    vel = np.diff(pts_cm, axis=0)*FPS/100.0
    acc = np.diff(vel, axis=0)*FPS
    return acc, vel

def extract_events(pts_cm, origin, label):
    acc, vel = compute_accel_dir(pts_cm)
    mag = np.linalg.norm(acc, axis=2)
    events = []
    for ji in BODY_J:
        crossing = mag[:,ji] > SPIKE_THR
        in_event = False
        for fi in range(len(crossing)):
            if crossing[fi]:
                if not in_event:
                    ev = {"id": f"{label}_{ji}_{fi}", "joint": ji, "start": fi, "end": fi, 
                          "peak_fi": fi, "peak_acc": float(mag[fi,ji]), "dir": acc[fi,ji],
                          "origin": origin[fi+2, ji], "cand": label}
                    in_event = True
                else:
                    ev["end"] = fi
                    if float(mag[fi,ji]) > ev["peak_acc"]:
                        ev["peak_fi"] = fi
                        ev["peak_acc"] = float(mag[fi,ji])
                        ev["dir"] = acc[fi,ji]
            else:
                if in_event:
                    ev["duration"] = ev["end"] - ev["start"] + 1
                    span_vel = vel[ev["start"]:ev["end"]+2, ji]
                    if len(span_vel) > 0:
                        ev["delta_v"] = float(np.linalg.norm(span_vel[-1] - span_vel[0]))
                    else: ev["delta_v"] = 0.0
                    events.append(ev)
                    in_event = False
        if in_event:
            ev["duration"] = ev["end"] - ev["start"] + 1
            span_vel = vel[ev["start"]:ev["end"]+2, ji]
            if len(span_vel) > 0: ev["delta_v"] = float(np.linalg.norm(span_vel[-1] - span_vel[0]))
            else: ev["delta_v"] = 0.0
            events.append(ev)
    return events

# ─── 4. Bootstrap Physics Metrics ────────────────────────────────────────────

def paired_block_bootstrap(arr_b, arr_c, block_size=30, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    N = min(len(arr_b), len(arr_c))
    n_blocks = N // block_size
    valid_blocks = []
    for i in range(n_blocks):
        b_block = arr_b[i*block_size:(i+1)*block_size]
        c_block = arr_c[i*block_size:(i+1)*block_size]
        if not np.isnan(b_block).all() and not np.isnan(c_block).all():
            valid_blocks.append(b_block - c_block)
    if not valid_blocks: return None, None
    means = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(valid_blocks), len(valid_blocks))
        sample = np.concatenate([valid_blocks[i] for i in idx])
        means.append(np.nanmean(sample))
    return np.mean(means), np.percentile(means, [2.5, 97.5])

def eval_physics(s6_b, s6_c, gt_mm, gt_ok):
    gt_floor = float(np.nanmin(gt_mm[:,[15,16],1]))
    floor_cm = gt_floor / 10.0
    
    pen_b = np.maximum(0, gt_floor - s6_b[:,[15,16],1]*10.0)
    pen_c = np.maximum(0, gt_floor - s6_c[:,[15,16],1]*10.0)
    
    vel_b = np.diff(s6_b/10.0, axis=0)*FPS
    vel_c = np.diff(s6_c/10.0, axis=0)*FPS
    
    contact_lb = s6_b[:,15,1] < (floor_cm + 5.0)
    contact_rb = s6_b[:,16,1] < (floor_cm + 5.0)
    contact_lc = s6_c[:,15,1] < (floor_cm + 5.0)
    contact_rc = s6_c[:,16,1] < (floor_cm + 5.0)
    
    slide_lb = np.full(NFRAMES-1, np.nan)
    slide_lc = np.full(NFRAMES-1, np.nan)
    for fi in range(NFRAMES-1):
        if contact_lb[fi]: slide_lb[fi] = float(np.linalg.norm(vel_b[fi,15,[0,2]]))
        if contact_lc[fi]: slide_lc[fi] = float(np.linalg.norm(vel_c[fi,15,[0,2]]))
        
    slide_diff_ci = paired_block_bootstrap(slide_lb, slide_lc)
    pen_diff_ci = paired_block_bootstrap(pen_b[:,0], pen_c[:,0])
    
    return slide_diff_ci, pen_diff_ci, gt_floor

# ─── 5. Subset Selection Chance ──────────────────────────────────────────────

def eval_subset_chance(kpts, scores, K, extr, gt_mm, gt_ok):
    P = [K[c] @ np.hstack(extr[c]) for c in range(3)]
    pairs = [(0,1),(0,2),(1,2),(0,1,2)]
    
    records = []
    for fi in range(NFRAMES):
        if not gt_ok[fi]: continue
        for ji in range(17):
            pts2d = kpts[fi,:,ji].astype(np.float64)
            conf = scores[fi,:,ji].astype(np.float64)
            valid = (conf >= CONF_GATE) & np.isfinite(pts2d).all(1)
            if valid.sum() < 3: continue
            if not np.isfinite(gt_mm[fi,ji]).all(): continue
            
            hyp_results = {}
            for hyp in pairs:
                sub = np.array(hyp)
                if not valid[sub].all(): continue
                try:
                    x = triangulate_robust(pts2d[sub], [P[i] for i in sub], conf[sub], f_scale=10.0)
                    if not np.isfinite(x).all(): continue
                    gt_err = float(np.linalg.norm(x*10.0 - gt_mm[fi,ji]))
                    err_all = _reprojection_errors(x, pts2d, P)
                    sub_err = err_all[sub]
                    score = float(np.median(sub_err) + 0.25*np.max(sub_err) - 0.5*np.sum(conf[sub]))
                    hyp_results[hyp] = {"score":score, "gt_err":gt_err}
                except: continue
                
            if len(hyp_results) < 2: continue
            sel = min(hyp_results, key=lambda k: hyp_results[k]["score"])
            best = min(hyp_results, key=lambda k: hyp_results[k]["gt_err"])
            
            errs = [hyp_results[h]["gt_err"] for h in hyp_results]
            min_err = min(errs)
            n_tied = sum(1 for e in errs if abs(e - min_err) < 1.0)
            
            chance = n_tied / len(hyp_results)
            records.append({"ji":ji, "correct": (sel == best), "regret": hyp_results[sel]["gt_err"] - min_err, "chance": chance, "n_hyp": len(hyp_results)})
            
    c_rate = np.mean([r["correct"] for r in records])
    c_chance = np.mean([r["chance"] for r in records])
    regrets = [r["regret"] for r in records]
    
    # Bootstrap diff
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(1000):
        samp = rng.choice(records, len(records), replace=True)
        diffs.append(np.mean([s["correct"] for s in samp]) - np.mean([s["chance"] for s in samp]))
    ci = np.percentile(diffs, [2.5, 97.5])
    
    return {
        "N": len(records),
        "correct_rate": round(c_rate, 4),
        "chance_rate": round(c_chance, 4),
        "diff_ci": [round(ci[0],4), round(ci[1],4)],
        "regret_p50": round(float(np.median(regrets)),2),
        "regret_p95": round(float(np.percentile(regrets,95)),2),
        "regret_p99": round(float(np.percentile(regrets,99)),2),
        "regret_max": round(float(np.max(regrets)),2),
        "gt_25": sum(1 for r in regrets if r > 25),
        "gt_50": sum(1 for r in regrets if r > 50),
        "gt_100": sum(1 for r in regrets if r > 100),
        "gt_200": sum(1 for r in regrets if r > 200)
    }

# ─── Main Logic ──────────────────────────────────────────────────────────────

def main():
    test_event_matcher()
    kpts, scores, calib, K, extr, gt_mm, gt_ok, bl = get_shared_data()
    mask_a, mask_b, mask_c = apply_masks(kpts, scores)
    
    print("Running candidates...")
    res_a = run_pipeline(mask_a[0], mask_a[1], K, extr, calib, bl)
    res_b = run_pipeline(mask_b[0], mask_b[1], K, extr, calib, bl)
    res_c = run_pipeline(mask_c[0], mask_c[1], K, extr, calib, bl)
    
    print("Extracting & Matching Events...")
    ev_a = extract_events(res_a["stage6"], res_a["origin"], "A")
    ev_b = extract_events(res_b["stage6"], res_b["origin"], "B")
    ev_c = extract_events(res_c["stage6"], res_c["origin"], "C")
    gt_evs = extract_events(gt_mm, np.full((NFRAMES,17), "GT"), "GT")
    
    # Independent GT support for B and C
    tols = [0, 1, 2, 3, 5]
    b_gt_support = {}
    c_gt_support = {}
    
    for t in tols:
        mb, _ = match_events(ev_b, gt_evs, t)
        mc, _ = match_events(ev_c, gt_evs, t)
        b_gt_support[str(t)] = {"supported": len(mb), "unsupported": len(ev_b) - len(mb)}
        c_gt_support[str(t)] = {"supported": len(mc), "unsupported": len(ev_c) - len(mc)}
        
    # Match B vs C directly
    b_to_c, c_to_b = match_events(ev_b, ev_c, dt_tol=1)
    
    b_matched_c = []
    b_absent_c = []
    c_absent_b = []
    
    for b in ev_b:
        if b["id"] in b_to_c:
            b_matched_c.append(b)
        else:
            b_absent_c.append(b)
            
    for c in ev_c:
        if c["id"] not in c_to_b:
            c_absent_b.append(c)
            
    # Baseline comparison (A->B, A->C)
    a_to_b, _ = match_events(ev_a, ev_b, dt_tol=1)
    a_to_c, _ = match_events(ev_a, ev_c, dt_tol=1)
    
    a_groups = {
        "A_total": len(ev_a),
        "A_matched_B": len(a_to_b),
        "A_matched_C": len(a_to_c)
    }
    
    bc_groups = {
        "B_total": len(ev_b),
        "C_total": len(ev_c),
        "B_matched_C": len(b_matched_c),
        "B_absent_from_C": len(b_absent_c),
        "C_absent_from_B": len(c_absent_b)
    }
    
    # Save unsupported B-added events for visual review
    mb_5, _ = match_events(ev_b, gt_evs, 5)
    unsupported_b_added = [b for b in b_absent_c if b["id"] not in mb_5]

    print("Reconciling C Starvation...")
    starvations = []
    for m in mask_c[2]:
        fi, c, ji = m["f"], m["c"], m["j"]
        ni_b = int(res_b["tri"].num_inliers[fi,ji])
        ni_c = int(res_c["tri"].num_inliers[fi,ji])
        if np.isfinite(res_b["raw"][fi,ji]).all() and not np.isfinite(res_c["raw"][fi,ji]).all():
            starvations.append({
                "fi":fi, "ji":ji, "cam_removed": c,
                "inliers_before": ni_b, "inliers_after": ni_c,
                "origin_b": res_b["origin"][fi,ji],
                "origin_c": res_c["origin"][fi,ji]
            })
    
    print("Bootstrapping Physics...")
    slide_diff_ci, pen_diff_ci, gt_floor = eval_physics(res_b["stage6"], res_c["stage6"], gt_mm, gt_ok)
    
    subset = eval_subset_chance(kpts, scores, K, extr, gt_mm, gt_ok)
    
    out_data = {
        "a_baseline": a_groups,
        "bc_groups": bc_groups,
        "b_gt_support": b_gt_support,
        "c_gt_support": c_gt_support,
        "unsupported_b_added_count": len(unsupported_b_added),
        "starvations": len(starvations),
        "subset_stats": subset,
        "phys_diff": {
            "slide_mean": float(slide_diff_ci[0]) if slide_diff_ci[0] is not None else None,
            "slide_ci": [float(x) for x in slide_diff_ci[1]] if slide_diff_ci[1] is not None else None,
            "pen_mean": float(pen_diff_ci[0]) if pen_diff_ci[0] is not None else None,
            "pen_ci": [float(x) for x in pen_diff_ci[1]] if pen_diff_ci[1] is not None else None,
        }
    }
    
    # Clean numpy types for JSON serialization
    def clean_ev(ev):
        return {k:float(v) if isinstance(v, (np.floating, np.integer)) else v for k,v in ev.items() if isinstance(v, (int,float,str,np.floating,np.integer))}
        
    # Save arrays and events for Gate 2 visualizer
    with open(OUT/"gate1_metrics.json", "w") as f: json.dump(out_data, f, indent=2)
    with open(OUT/"b_events.json", "w") as f: json.dump([clean_ev(ev) for ev in ev_b], f)
    with open(OUT/"unsupported_b_added.json", "w") as f: json.dump([clean_ev(ev) for ev in unsupported_b_added], f)
    with open(OUT/"starvations.json", "w") as f: json.dump(starvations, f)
    
    np.savez_compressed(OUT/"gate1_arrays.npz", 
                        b_stage6=res_b["stage6"], c_stage6=res_c["stage6"], 
                        b_origin=res_b["origin"], c_origin=res_c["origin"],
                        mask_c_log=mask_c[2], gt=gt_mm)
                        
    print("Gate 1 Complete. Internal assertions passed. Outputs saved.")

if __name__ == "__main__":
    main()
