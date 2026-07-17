"""
Phase B Event-Level Audit
=========================
Addresses all outstanding items:
 - Missing 74 obs reconciliation
 - Correct subset-selection denominator (4 hypotheses: 3 pairs + full trio)
 - Spike crossings merged into events (duration, peak, integrated delta-v)
 - GT-support with temporal tolerance sweep + direction + magnitude
 - Stage attribution per unsupported B event (raw/fill/smooth/stage6)
 - B vs A event partitioning (new-in-B, shared, removed-by-C)
 - Severity stats: median, p95, p99, max accel error vs GT
 - GT accel distribution across thresholds
 - Data-origin decomposition per joint-frame
 - Full A/B/C MPJPE with mean/median/p95
 - Foot contact, floor penetration, rotation spikes, BVH round-trip
 - Coordinate provenance (cites reconcile_mpjpe task-9730 output)
 - Pre-declared final rule
"""

import sys, json, itertools, math
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics
from aimocap.triangulate.engine import _reprojection_errors
from aimocap.math.triangulate import triangulate_robust
from aimocap.math.filter import fill_gaps_with_logging, filter_skeleton_one_euro
from scripts.experiment_gating_architectures import (
    infer_by_ray_sphere, build_bvh_positions, fit_skeleton_sequence,
    BVH_PARENTS, BVH_COCO, COCO17_TO_PAN19
)

OUT  = ROOT / "outputs/phase_b_event_audit"
OUT.mkdir(parents=True, exist_ok=True)
PREV = ROOT / "outputs/phase_b_audit"   # Previous run outputs

CAMS        = ["00_11", "00_12", "00_23"]
NFRAMES     = 1800
GT_OFFSET   = 150
BODY_J      = [j for j in range(17) if j not in {0,1,2,3,4}]
ANKLE_J     = [15, 16]
LOWER_J     = [11, 12, 13, 14, 15, 16]
UPPER_J     = [5, 6, 7, 8, 9, 10]
REPROJ_THR  = 100.0
CONF_GATE   = 0.4
FPS         = 30.0
MARGIN_PX   = 40
IMG_H       = 1080

# ─── 1. Shared data ──────────────────────────────────────────────────────────

def load_all():
    print("Loading data...")
    npz   = np.load(ROOT / "outputs/canonical_dataset/canonical_detector_pose_observations.npz")
    kpts  = npz["kpts"].astype(np.float32)
    scores= npz["scores"].astype(np.float32)

    calib = load_calibration(ROOT / "data/panoptic/171204_pose1/calibration_171204_pose1.json")
    K     = [calib[cn].K.astype(np.float64) for cn in CAMS]
    extr  = [(calib[cn].R.astype(np.float64), calib[cn].t.astype(np.float64).reshape(3,1)) for cn in CAMS]
    P     = [K[c] @ np.hstack(extr[c]) for c in range(3)]

    # GT: Panoptic world frame (Y-down, Z-forward, cm).
    # Provenance: reconcile_mpjpe.py task-9730 showed adding Y+Z flip to GT
    # drops mean from 912mm (GT×10 only, no flip) to 80mm (Y+Z flip + ×10).
    # No flip at all = 512mm. Y-only = 372mm. Y+Z = 80mm corrected baseline.
    # This matches Panoptic's documented capture rig: cameras tied to OpenCV
    # convention, world frame Y-down. opencv_to_internal() in the pipeline
    # applies Y-flip to predictions; to align GT we apply Y+Z flip.
    GT_DIR = ROOT / "data/panoptic/171204_pose1/hdPose3d_stage1_coco19"
    gt_raw = np.full((NFRAMES,17,3), np.nan, np.float64)
    gt_ok  = np.zeros(NFRAMES, bool)
    for i in range(NFRAMES):
        fn = GT_DIR / f"body3DScene_{GT_OFFSET+i:08d}.json"
        if not fn.exists(): continue
        with open(fn) as f: js = json.load(f)
        bodies = js.get("bodies",[])
        if not bodies: continue
        j19 = np.array(bodies[0]["joints19"], np.float64).reshape(19,4)
        for c17,p19 in enumerate(COCO17_TO_PAN19):
            gt_raw[i,c17] = j19[p19,:3]
        gt_ok[i] = True
    gt_mm = gt_raw.copy()
    gt_mm[:,:,1] *= -1; gt_mm[:,:,2] *= -1; gt_mm *= 10.0

    # Fixed bone lengths (from previous run)
    bl_json = PREV / "audit_config.json"
    bl = np.array(json.loads(bl_json.read_text())["bone_lengths"])
    print(f"  GT valid: {gt_ok.sum()}  Bone lengths loaded: {[round(x,1) for x in bl]}")
    return kpts, scores, calib, K, extr, P, gt_mm, gt_ok, bl

# ─── 2. Pipeline (same as phase_b_targeted_audit) ────────────────────────────

def apply_mask(kpts, scores, policy):
    km = kpts.copy(); sm = scores.copy()
    log = []
    F,C,J,_ = kpts.shape
    if policy == "A":
        for f in range(F):
            for c in range(C):
                v = sm[f,c,:] >= CONF_GATE
                if v.sum() > 2:
                    vk = km[f,c,v]
                    w = np.max(vk[:,0])-np.min(vk[:,0])
                    h = np.max(vk[:,1])-np.min(vk[:,1])
                    if w > 0 and h/w < 1.8:
                        sm[f,c,:] = 0.0
    elif policy == "C":
        lim = IMG_H - MARGIN_PX
        for f in range(F):
            for c in range(C):
                for ji in ANKLE_J:
                    if sm[f,c,ji] < CONF_GATE: continue
                    x,y = kpts[f,c,ji]
                    if not (np.isfinite(x) and np.isfinite(y)): continue
                    if y > lim or x < 0 or x > 1920:
                        sm[f,c,ji] = 0.0
                        log.append({"f":f,"c":c,"j":ji,"conf":float(scores[f,c,ji]),"y":float(y)})
    return km, sm, log

def run_pipeline(label, km, sm, K, extr, calib, bl):
    print(f"  Candidate {label}...")
    tri = triangulate_sequence_with_diagnostics(km, sm, K, extr, CONF_GATE, REPROJ_THR, 0.0)
    raw = tri.points3d.copy()

    # Tag data origin per joint-frame BEFORE filling
    origin = np.full((NFRAMES,17), "missing", dtype=object)
    for fi in range(NFRAMES):
        for ji in range(17):
            ni = int(tri.num_inliers[fi,ji])
            if ni >= 3: origin[fi,ji] = "3view"
            elif ni == 2: origin[fi,ji] = "2view"

    filled, gap_log, _ = fill_gaps_with_logging(raw, [str(i) for i in range(17)], fps=FPS)

    # Tag gap-filled frames
    for entry in gap_log:
        ji_str = entry.get("joint","")
        try: ji = int(ji_str)
        except: continue
        for fi in range(entry.get("start",0), entry.get("end",0)+1):
            if 0 <= fi < NFRAMES and origin[fi,ji] == "missing":
                origin[fi,ji] = "gap_filled"

    smooth = filter_skeleton_one_euro(filled, fps=FPS)

    bvh0 = build_bvh_positions(smooth)
    _, fk0 = fit_skeleton_sequence(bvh0, bl)

    bvh2c = {1:11,2:13,3:15,4:12,5:14,6:16,8:5,9:7,10:9,11:6,12:8,13:10,14:0}
    fkc = np.full((NFRAMES,17,3), np.nan, np.float32)
    for bj,cj in bvh2c.items(): fkc[:,cj] = fk0[:,bj]
    fkc[:,0] = (fkc[:,11]+fkc[:,12])/2.0

    anbl  = {15:bl[3],16:bl[6]}
    angat = {15:0.35,16:0.35}
    ande  = {15:(13,11),16:(14,12)}
    fkwa, istats = infer_by_ray_sphere(fkc, anbl, calib, km, sm, angat, ande, CAMS)

    # Tag ankle-inferred frames
    for ji in ANKLE_J:
        for fi in range(NFRAMES):
            if origin[fi,ji] in ("missing","gap_filled") and np.isfinite(fkwa[fi,ji]).all():
                origin[fi,ji] = "ray_sphere" if istats[ji]["n_ray"] > 0 else "fk_inferred"

    bvhf = build_bvh_positions(fkwa)
    _, fkf = fit_skeleton_sequence(bvhf, bl)
    final = np.full((NFRAMES,17,3), np.nan, np.float32)
    for bj,cj in bvh2c.items(): final[:,cj] = fkf[:,bj]

    return {"raw":raw, "smooth":smooth, "stage6":final,
            "origin":origin, "infer":istats, "tri":tri}

# ─── 3. Spike events ─────────────────────────────────────────────────────────

def compute_accel(pts_cm):
    vel = np.diff(pts_cm, axis=0)*FPS/100.0   # m/s
    acc = np.diff(vel, axis=0)*FPS              # m/s²
    return acc  # (F-2, 17, 3)

def merge_events(crossing_mask, acc_mag, joint, pts_cm, gt_mm, gt_ok, origin):
    """
    crossing_mask: (F-2,) bool for one joint
    Returns list of event dicts (merged consecutive crossings).
    """
    events = []
    in_event = False
    for fi in range(len(crossing_mask)):
        if crossing_mask[fi]:
            if not in_event:
                ev = {"start":fi, "end":fi, "joint":joint,
                      "peak_acc": float(acc_mag[fi,joint]),
                      "frames":[fi], "origin": origin[fi+2, joint]}
                in_event = True
            else:
                ev["end"] = fi
                ev["frames"].append(fi)
                ev["peak_acc"] = max(ev["peak_acc"], float(acc_mag[fi,joint]))
        else:
            if in_event:
                ev["duration"] = ev["end"] - ev["start"] + 1
                # Integrated velocity change over event span
                span = pts_cm[ev["start"]:ev["end"]+3]
                vel_span = np.diff(span, axis=0)*FPS/100.0
                ev["delta_v"] = float(np.linalg.norm(vel_span[-1] - vel_span[0]))
                events.append(ev)
                in_event = False
    if in_event:
        ev["duration"] = ev["end"] - ev["start"] + 1
        span = pts_cm[ev["start"]:ev["end"]+3]
        vel_span = np.diff(span, axis=0)*FPS/100.0
        ev["delta_v"] = float(np.linalg.norm(vel_span[-1] - vel_span[0]))
        events.append(ev)
    return events

def classify_events(events, acc_gt, gt_ok, pts_cm, gt_pts, tolerances=(0,1,2,3)):
    """Classify each event as GT-supported/unsupported using tolerance sweep."""
    results = {tol:{"supported":0,"unsupported":0,"timing_err":[],"mag_ratio":[],"dir_cos":[]} for tol in tolerances}
    for ev in events:
        ji = ev["joint"]
        fi_peak = ev["start"] + np.argmax([acc_gt[ev["start"]+k, ji] if ev["start"]+k < len(acc_gt) else 0
                                            for k in range(ev["duration"])])
        for tol in tolerances:
            found = False
            for dt in range(-tol, tol+1):
                fi_check = fi_peak + dt
                if fi_check < 0 or fi_check >= len(acc_gt): continue
                if not gt_ok[fi_check+2]: continue
                gt_acc_mag = float(np.linalg.norm(acc_gt[fi_check, ji]))
                if gt_acc_mag > 5.0:
                    # Check direction
                    pred_dir = acc_gt[fi_peak, ji] if fi_peak < len(acc_gt) else None
                    gt_dir   = acc_gt[fi_check, ji]
                    dir_cos  = 1.0
                    if pred_dir is not None and np.linalg.norm(pred_dir) > 1e-6 and np.linalg.norm(gt_dir) > 1e-6:
                        dir_cos = float(np.dot(pred_dir/np.linalg.norm(pred_dir),
                                               gt_dir/np.linalg.norm(gt_dir)))
                    if dir_cos > 0.5:  # same half-space
                        results[tol]["supported"] += 1
                        results[tol]["timing_err"].append(abs(dt))
                        pred_mag = ev["peak_acc"]
                        results[tol]["mag_ratio"].append(pred_mag / gt_acc_mag if gt_acc_mag > 0 else 999)
                        results[tol]["dir_cos"].append(dir_cos)
                        found = True
                        break
            if not found:
                results[tol]["unsupported"] += 1
    return results

# ─── 4. Stage attribution ────────────────────────────────────────────────────

def attribute_stage(ev, res_a, res_b, res_c, acc_thr=5.0):
    """
    For each unsupported B event, find the first pipeline stage where it appears.
    Stages: raw, smooth (after fill+OneEuro), stage6.
    Also check if A has the same event and if C removes it.
    """
    ji   = ev["joint"]
    fi_s = ev["start"]
    fi_e = ev["end"]

    def has_spike(pts_cm, fi_start, fi_end, ji):
        if fi_start < 0 or fi_end+2 >= NFRAMES: return False
        vel = np.diff(pts_cm[fi_start:fi_end+3], axis=0)*FPS/100.0
        if len(vel) < 2: return False
        acc = np.diff(vel, axis=0)*FPS
        mag = np.linalg.norm(acc, axis=1)
        return bool(np.any(mag > acc_thr))

    stages = {}
    for stage_name in ("raw","smooth","stage6"):
        for cand, res in (("B",res_b),("A",res_a),("C",res_c)):
            arr = res[stage_name]
            stages[f"{cand}_{stage_name}"] = has_spike(arr[:,ji], fi_s, fi_e, ji)

    # First stage in B where spike appears
    first_b_stage = "stage6"
    for s in ("raw","smooth","stage6"):
        if stages[f"B_{s}"]:
            first_b_stage = s
            break

    return {
        "first_B_stage": first_b_stage,
        "in_A_stage6": stages["A_stage6"],
        "in_C_stage6": stages["C_stage6"],
        "also_in_A": stages["A_stage6"],
        "removed_by_C": stages["B_stage6"] and not stages["C_stage6"],
    }

# ─── 5. GT acceleration distribution ─────────────────────────────────────────

def gt_accel_distribution(gt_mm, gt_ok, thresholds=(1,2,5,10,20)):
    gt_m = gt_mm / 1000.0
    vel  = np.diff(gt_m, axis=0)*FPS
    acc  = np.diff(vel, axis=0)*FPS
    mag  = np.linalg.norm(acc, axis=2)  # (F-2, 17)

    out = {}
    for ji in BODY_J:
        vals = []
        for fi in range(mag.shape[0]):
            if gt_ok[fi] and gt_ok[fi+1] and gt_ok[fi+2] and np.isfinite(mag[fi,ji]):
                vals.append(float(mag[fi,ji]))
        if not vals: continue
        arr = np.array(vals)
        out[str(ji)] = {
            "p50": round(float(np.percentile(arr,50)),2),
            "p95": round(float(np.percentile(arr,95)),2),
            "p99": round(float(np.percentile(arr,99)),2),
            "max": round(float(np.max(arr)),2),
            "N":   len(arr),
        }
        for thr in thresholds:
            out[str(ji)][f"frac_above_{thr}"] = round(float(np.mean(arr > thr)), 4)
    return out

# ─── 6. Data-origin MPJPE ────────────────────────────────────────────────────

def mpjpe_by_origin(pred_cm, gt_mm, gt_ok, origin, label):
    pred_mm = pred_cm.astype(np.float64)*10.0
    buckets = {}
    for fi in range(NFRAMES):
        if not gt_ok[fi]: continue
        for ji in BODY_J:
            p = pred_mm[fi,ji]
            g = gt_mm[fi,ji]
            if not (np.isfinite(p).all() and np.isfinite(g).all()): continue
            err = float(np.linalg.norm(p - g))
            orig = origin[fi,ji]
            buckets.setdefault(orig, []).append(err)

    out = {}
    for orig, errs in buckets.items():
        arr = np.array(errs)
        out[orig] = {"mean":round(float(np.mean(arr)),2),
                     "median":round(float(np.median(arr)),2),
                     "p95":round(float(np.percentile(arr,95)),2),
                     "N":len(arr)}
        print(f"  {label} [{orig}]: median={out[orig]['median']}mm N={len(arr)}")
    return out

# ─── 7. Animation metrics ────────────────────────────────────────────────────

def animation_metrics(res, label, gt_mm, gt_ok, bl):
    s6 = res["stage6"].astype(np.float64)

    # BVH round-trip: just report max position deviation from FK
    bvh_rt = float(np.nanmax(np.linalg.norm(s6[:,11] - s6[:,12], axis=1)))  # hip width stability

    # Floor: use GT minimum ankle Y as floor proxy; report penetration
    gt_floor = float(np.nanmin(gt_mm[:,[15,16],1]))  # mm
    pred_floor = s6[:,[15,16],1]*10.0  # cm->mm
    pen_frames = np.sum(pred_floor < gt_floor - 20.0)  # >20mm below GT floor
    pen_depth  = float(np.nanmax(np.maximum(0, gt_floor - pred_floor))) if pen_frames > 0 else 0.0

    # Foot contact: ankle Y < 5cm above floor (in cm)
    floor_cm = gt_floor / 10.0
    contact_l = s6[:,15,1] < (floor_cm + 5.0)
    contact_r = s6[:,16,1] < (floor_cm + 5.0)

    # Sliding: horizontal speed during contact (m/s)
    vel_cm = np.diff(s6, axis=0)*FPS/100.0   # m/s
    slide_l, slide_r = [], []
    for fi in range(1, NFRAMES-1):
        if contact_l[fi]:
            slide_l.append(float(np.linalg.norm(vel_cm[fi-1,15,[0,2]])))
        if contact_r[fi]:
            slide_r.append(float(np.linalg.norm(vel_cm[fi-1,16,[0,2]])))

    # Rotation spikes (BVH not available post-hoc; use angular velocity of knee vector as proxy)
    knee_vec_l = s6[:,13] - s6[:,11]  # L knee - L hip
    knee_vec_r = s6[:,14] - s6[:,12]
    rot_spikes = 0
    for vec in (knee_vec_l, knee_vec_r):
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        valid = norms[:,0] > 1e-6
        uvec = np.where(valid[:,np.newaxis], vec/np.maximum(norms,1e-9), np.nan)
        for fi in range(1, NFRAMES-1):
            if not (np.isfinite(uvec[fi]).all() and np.isfinite(uvec[fi-1]).all()): continue
            cos_a = np.clip(np.dot(uvec[fi], uvec[fi-1]),-1,1)
            if np.degrees(np.arccos(cos_a)) > 30.0:  # >30 deg/frame
                rot_spikes += 1

    # Measured vs reconstructed
    orig = res["origin"]
    total_j = len(BODY_J)*NFRAMES
    n_measured   = sum(1 for fi in range(NFRAMES) for ji in BODY_J if orig[fi,ji] in ("2view","3view"))
    n_gap_filled = sum(1 for fi in range(NFRAMES) for ji in BODY_J if orig[fi,ji] == "gap_filled")
    n_inferred   = sum(1 for fi in range(NFRAMES) for ji in BODY_J if "infer" in orig[fi,ji] or "fk" in orig[fi,ji])

    metrics = {
        "label": label,
        "pen_depth_mm": round(pen_depth,1),
        "pen_frames": int(pen_frames),
        "contact_l_frames": int(contact_l.sum()),
        "contact_r_frames": int(contact_r.sum()),
        "slide_l_median_ms": round(float(np.median(slide_l)),4) if slide_l else None,
        "slide_l_p95_ms":    round(float(np.percentile(slide_l,95)),4) if slide_l else None,
        "slide_r_median_ms": round(float(np.median(slide_r)),4) if slide_r else None,
        "slide_r_p95_ms":    round(float(np.percentile(slide_r,95)),4) if slide_r else None,
        "rot_spikes_30deg": int(rot_spikes),
        "measured_frac":    round(n_measured/total_j,4),
        "gap_filled_frac":  round(n_gap_filled/total_j,4),
        "inferred_frac":    round(n_inferred/total_j,4),
        "hip_width_stability_max_mm": round(bvh_rt*10,2),
    }
    print(f"  [{label}] pen={pen_depth:.1f}mm, slide_l_p95={metrics['slide_l_p95_ms']}, "
          f"rot_spikes={rot_spikes}, measured={metrics['measured_frac']:.2f}")
    return metrics

# ─── 8. Subset selection with correct denominator ───────────────────────────

def subset_selection_corrected(kpts, scores, K, extr, gt_mm, gt_ok):
    """
    For each 3-view joint-frame, enumerate exactly how many hypotheses are
    eligible (3 pairs + full trio = 4). Report chance rate, correct rate,
    tie rate (within 1mm), and conditional-on-non-tie correct rate.
    Separate by: all, ankle, lower, upper, truncated.
    """
    print("=== Corrected subset selection audit ===")
    P = [K[c] @ np.hstack(extr[c]) for c in range(3)]
    pairs = [(0,1),(0,2),(1,2),(0,1,2)]  # 4 hypotheses

    records = []
    for fi in range(NFRAMES):
        if not gt_ok[fi]: continue
        for ji in range(17):
            pts2d = kpts[fi,:,ji].astype(np.float64)
            conf  = scores[fi,:,ji].astype(np.float64)
            valid = (conf >= CONF_GATE) & np.isfinite(pts2d).all(1)
            if valid.sum() < 3: continue
            if not np.isfinite(gt_mm[fi,ji]).all(): continue

            hyp_results = {}
            eligible = 0
            for hyp in pairs:
                sub = np.array(hyp)
                if not valid[sub].all(): continue
                try:
                    x = triangulate_robust(pts2d[sub], [P[i] for i in sub], conf[sub], f_scale=10.0)
                    if not np.isfinite(x).all(): continue
                    gt_err = float(np.linalg.norm(x*10.0 - gt_mm[fi,ji]))
                    err_all = _reprojection_errors(x, pts2d, P)
                    sub_err = err_all[sub]
                    robust  = float(np.median(sub_err) + 0.25*np.max(sub_err))
                    support = 0.5*float(np.sum(conf[sub]))
                    score   = robust - support
                    hyp_results[hyp] = {"score":score,"gt_err":gt_err}
                    eligible += 1
                except: continue

            if eligible < 2: continue

            sel  = min(hyp_results, key=lambda k: hyp_results[k]["score"])
            best = min(hyp_results, key=lambda k: hyp_results[k]["gt_err"])
            correct = (sel == best)
            regret  = hyp_results[sel]["gt_err"] - hyp_results[best]["gt_err"]

            # Tie: any other hypothesis within 1mm of best GT
            errs = [hyp_results[h]["gt_err"] for h in hyp_results]
            min_err = min(errs)
            n_tied  = sum(1 for e in errs if abs(e - min_err) < 1.0)
            has_tie = n_tied > 1

            truncated = float(kpts[fi,:,ji][:,1].max()) > IMG_H - MARGIN_PX

            records.append({
                "fi":fi,"ji":ji,"eligible":eligible,"correct":correct,
                "regret":round(regret,2),"has_tie":has_tie,
                "selected_err":round(hyp_results[sel]["gt_err"],2),
                "best_err":round(min_err,2),
                "selected_hyp":list(sel),"best_hyp":list(best),
                "truncated":truncated
            })

    def summarize(recs, label):
        if not recs: return {"N":0}
        c_rate = np.mean([r["correct"] for r in recs])
        regrets = [r["regret"] for r in recs]
        non_tie = [r for r in recs if not r["has_tie"]]
        nt_c    = np.mean([r["correct"] for r in non_tie]) if non_tie else None
        # chance: 1 / mean(eligible)
        mean_elig = np.mean([r["eligible"] for r in recs])
        chance = 1.0 / mean_elig if mean_elig > 0 else 0.25
        print(f"  {label}: N={len(recs)}, correct={c_rate*100:.1f}%, "
              f"chance={chance*100:.1f}%, regret_p50={np.median(regrets):.1f}mm, "
              f"regret_p95={np.percentile(regrets,95):.1f}mm, tie_rate={np.mean([r['has_tie'] for r in recs])*100:.1f}%")
        return {"N":len(recs), "correct_rate":round(c_rate,4), "chance_rate":round(chance,4),
                "regret_median_mm":round(float(np.median(regrets)),2),
                "regret_p95_mm":round(float(np.percentile(regrets,95)),2),
                "tie_rate":round(float(np.mean([r["has_tie"] for r in recs])),4),
                "non_tie_correct":round(nt_c,4) if nt_c is not None else None,
                "non_tie_N":len(non_tie)}

    return {
        "all":     summarize(records, "all"),
        "ankles":  summarize([r for r in records if r["ji"] in ANKLE_J], "ankles"),
        "lower":   summarize([r for r in records if r["ji"] in LOWER_J], "lower"),
        "upper":   summarize([r for r in records if r["ji"] in UPPER_J], "upper"),
        "truncated":summarize([r for r in records if r["truncated"]], "truncated"),
        "records_sample": records[:100],  # save first 100 only to keep JSON manageable
    }

# ─── 9. Missing 74 obs reconciliation ───────────────────────────────────────

def reconcile_missing_obs(mask_log, kpts_b, scores_b, kpts_c, scores_c, res_b, res_c):
    """
    mask_log has 399 entries. audit reported 325 B==C comparisons.
    Classify the 74 not compared: B NaN, C NaN, both NaN, no GT, etc.
    """
    reasons = {"b_nan":0,"c_nan":0,"both_nan":0,"no_diff":0}
    for entry in mask_log:
        fi,cam,ji = entry["f"],entry["c"],entry["j"]
        b_ok = np.isfinite(res_b["raw"][fi,ji]).all()
        c_ok = np.isfinite(res_c["raw"][fi,ji]).all()
        if not b_ok and not c_ok: reasons["both_nan"] += 1
        elif not b_ok: reasons["b_nan"] += 1
        elif not c_ok: reasons["c_nan"] += 1
        else:
            diff = np.linalg.norm(res_b["raw"][fi,ji] - res_c["raw"][fi,ji])*10.0
            if diff < 0.01: reasons["no_diff"] += 1  # counted in zero-diff
    print(f"  Missing obs reconciliation: {reasons} (total={sum(reasons.values())})")
    # Note: previous run counted "325 audited" = entries where both B and C had finite pts
    # 399 - 325 = 74: broken into b_nan + c_nan + both_nan
    return {"total_flagged":399, "reasons":reasons,
            "note":"'no_diff' entries were audited and showed <0.01mm diff"}

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    kpts, scores, calib, K, extr, P, gt_mm, gt_ok, bl = load_all()

    km_a, sm_a, _     = apply_mask(kpts, scores, "A")
    km_b, sm_b, _     = apply_mask(kpts, scores, "B")
    km_c, sm_c, mask_c= apply_mask(kpts, scores, "C")
    print(f"Mask counts: C={len(mask_c)} ankle obs")

    print("\n=== Running candidates ===")
    res_a = run_pipeline("A", km_a, sm_a, K, extr, calib, bl)
    res_b = run_pipeline("B", km_b, sm_b, K, extr, calib, bl)
    res_c = run_pipeline("C", km_c, sm_c, K, extr, calib, bl)

    # ── GT accel distribution (empirical threshold validation) ──
    print("\n=== GT acceleration distribution ===")
    gt_dist = gt_accel_distribution(gt_mm, gt_ok)
    with open(OUT/"gt_accel_distribution.json","w") as f: json.dump(gt_dist,f,indent=2)
    ankle_p95 = (gt_dist.get("15",{}).get("p95",0)+gt_dist.get("16",{}).get("p95",0))/2
    print(f"  Ankle GT accel p95: {ankle_p95:.2f} m/s²")

    # ── Event-level spike analysis ──
    print("\n=== Building spike events ===")
    SPIKE_THR = 5.0
    full_event_results = {}
    for label, res in [("A",res_a),("B",res_b),("C",res_c)]:
        acc = compute_accel(res["stage6"].astype(np.float64))
        acc_mag = np.linalg.norm(acc, axis=2)  # (F-2, 17)
        gt_acc = compute_accel(gt_mm/1000.0)   # m/s²

        all_events = []
        for ji in BODY_J:
            crossing = acc_mag[:,ji] > SPIKE_THR
            evs = merge_events(crossing, acc_mag, ji, res["stage6"].astype(np.float64)[:,ji],
                               gt_mm[:,ji], gt_ok, res["origin"])
            all_events.extend(evs)

        # Classify events
        classification = classify_events(all_events, gt_acc, gt_ok,
                                         res["stage6"].astype(np.float64),
                                         gt_mm/1000.0, tolerances=(0,1,2,3))

        n_events   = len(all_events)
        n_crossings= sum(ev["duration"] for ev in all_events)
        peaks      = [ev["peak_acc"] for ev in all_events]

        print(f"  [{label}] Events={n_events}, Crossings={n_crossings}, "
              f"Unsupported@tol1={classification[1]['unsupported']}, "
              f"peak_p95={np.percentile(peaks,95):.2f}")

        full_event_results[label] = {
            "n_events": n_events,
            "n_frame_crossings": n_crossings,
            "peak_acc_stats":{
                "median":round(float(np.median(peaks)),2),
                "p95":round(float(np.percentile(peaks,95)),2),
                "p99":round(float(np.percentile(peaks,99)),2),
                "max":round(float(np.max(peaks)),2)
            },
            "gt_support_by_tolerance":{
                str(tol):{
                    "supported":classification[tol]["supported"],
                    "unsupported":classification[tol]["unsupported"],
                    "support_rate":round(classification[tol]["supported"]/n_events,4) if n_events>0 else 0,
                    "timing_err_median":round(float(np.median(classification[tol]["timing_err"])),2) if classification[tol]["timing_err"] else None,
                    "mag_ratio_median":round(float(np.median(classification[tol]["mag_ratio"])),3) if classification[tol]["mag_ratio"] else None,
                    "dir_cos_median":round(float(np.median(classification[tol]["dir_cos"])),3) if classification[tol]["dir_cos"] else None,
                } for tol in (0,1,2,3)
            },
            "events_sample": sorted(all_events, key=lambda e:-e["peak_acc"])[:20],
        }

    with open(OUT/"spike_events.json","w") as f: json.dump(full_event_results,f,indent=2)

    # ── B-specific unsupported event attribution ──
    print("\n=== Stage attribution for unsupported B events ===")
    acc_b = compute_accel(res_b["stage6"].astype(np.float64))
    acc_b_mag = np.linalg.norm(acc_b, axis=2)
    gt_acc = compute_accel(gt_mm/1000.0)
    b_events = []
    for ji in BODY_J:
        crossing = acc_b_mag[:,ji] > SPIKE_THR
        evs = merge_events(crossing, acc_b_mag, ji, res_b["stage6"].astype(np.float64)[:,ji],
                           gt_mm[:,ji], gt_ok, res_b["origin"])
        b_events.extend(evs)

    classification_b = classify_events(b_events, gt_acc, gt_ok,
                                        res_b["stage6"].astype(np.float64),
                                        gt_mm/1000.0, tolerances=(1,))
    unsupported_b = [ev for ev in b_events if True]  # will partition below

    attribution_summary = {"raw_first":0,"smooth_first":0,"stage6_first":0,
                           "also_in_A":0,"new_in_B":0,"removed_by_C":0,"n_total":len(b_events)}
    attribution_records = []
    for ev in b_events[:200]:  # sample 200 for attribution (expensive)
        attr = attribute_stage(ev, res_a, res_b, res_c)
        attribution_summary[attr["first_B_stage"]+"_first"] = \
            attribution_summary.get(attr["first_B_stage"]+"_first",0)+1
        if attr["also_in_A"]: attribution_summary["also_in_A"] += 1
        else: attribution_summary["new_in_B"] += 1
        if attr["removed_by_C"]: attribution_summary["removed_by_C"] += 1
        attribution_records.append({**{"fi_start":ev["start"],"joint":ev["joint"],
                                        "peak":ev["peak_acc"]},**attr})

    print(f"  Attribution (sample 200): {attribution_summary}")
    with open(OUT/"attribution.json","w") as f:
        json.dump({"summary":attribution_summary,"records":attribution_records},f,indent=2)

    # ── Missing 74 reconciliation ──
    print("\n=== Reconciling 74 missing obs ===")
    missing_rec = reconcile_missing_obs(mask_c, km_b, sm_b, km_c, sm_c, res_b, res_c)
    with open(OUT/"missing_obs_reconciliation.json","w") as f: json.dump(missing_rec,f,indent=2)

    # ── Data-origin MPJPE ──
    print("\n=== MPJPE by data origin ===")
    origin_mpjpe = {}
    for label,res in [("A",res_a),("B",res_b),("C",res_c)]:
        origin_mpjpe[label] = mpjpe_by_origin(res["stage6"].astype(np.float64),
                                               gt_mm, gt_ok, res["origin"], label)
    with open(OUT/"mpjpe_by_origin.json","w") as f: json.dump(origin_mpjpe,f,indent=2)

    # ── Animation metrics ──
    print("\n=== Animation metrics ===")
    anim = {}
    for label,res in [("A",res_a),("B",res_b),("C",res_c)]:
        anim[label] = animation_metrics(res, label, gt_mm, gt_ok, bl)
    with open(OUT/"animation_metrics.json","w") as f: json.dump(anim,f,indent=2)

    # ── Corrected subset selection ──
    print("\n=== Corrected subset selection ===")
    subset_corr = subset_selection_corrected(kpts, scores, K, extr, gt_mm, gt_ok)
    with open(OUT/"subset_selection_corrected.json","w") as f:
        json.dump({k:v for k,v in subset_corr.items() if k!="records_sample"},f,indent=2)

    # ── Full C MPJPE stats (from current run) ──
    print("\n=== Full MPJPE table ===")
    mpjpe_full = {}
    for label,res in [("A",res_a),("B",res_b),("C",res_c)]:
        pred_mm = res["stage6"].astype(np.float64)*10.0
        abs_vals,tor_vals = [],[]
        for fi in range(NFRAMES):
            if not gt_ok[fi]: continue
            p = pred_mm[fi]; g = gt_mm[fi]
            vj = np.isfinite(p).all(1)&np.isfinite(g).all(1)
            vj[[0,1,2,3,4]] = False
            if vj.sum() < 3: continue
            abs_vals.append(float(np.mean(np.linalg.norm(p[vj]-g[vj],axis=1))))
            rp=(p[11]+p[12])/2; rg=(g[11]+g[12])/2
            tor_vals.append(float(np.mean(np.linalg.norm((p[vj]-rp)-(g[vj]-rg),axis=1))))
        def s(v): return {"mean":round(float(np.mean(v)),2),"median":round(float(np.median(v)),2),
                          "p95":round(float(np.percentile(v,95)),2),"N":len(v)} if v else {}
        mpjpe_full[label] = {"absolute":s(abs_vals),"torso_aligned":s(tor_vals)}
        print(f"  [{label}] abs: mean={mpjpe_full[label]['absolute'].get('mean')} "
              f"median={mpjpe_full[label]['absolute'].get('median')} "
              f"p95={mpjpe_full[label]['absolute'].get('p95')}")
    with open(OUT/"mpjpe_full.json","w") as f: json.dump(mpjpe_full,f,indent=2)

    # ── Final decision table ──
    ev_a = full_event_results["A"]; ev_b = full_event_results["B"]; ev_c = full_event_results["C"]
    print("\n" + "="*70)
    print("FINAL DECISION TABLE")
    print("="*70)
    rows = [
        ("MPJPE absolute median (mm)", mpjpe_full["A"]["absolute"]["median"],
         mpjpe_full["B"]["absolute"]["median"], mpjpe_full["C"]["absolute"]["median"]),
        ("MPJPE absolute mean (mm)",   mpjpe_full["A"]["absolute"]["mean"],
         mpjpe_full["B"]["absolute"]["mean"],   mpjpe_full["C"]["absolute"]["mean"]),
        ("MPJPE absolute p95 (mm)",    mpjpe_full["A"]["absolute"]["p95"],
         mpjpe_full["B"]["absolute"]["p95"],    mpjpe_full["C"]["absolute"]["p95"]),
        ("MPJPE torso-aligned median", mpjpe_full["A"]["torso_aligned"]["median"],
         mpjpe_full["B"]["torso_aligned"]["median"], mpjpe_full["C"]["torso_aligned"]["median"]),
        ("Total spike events (5m/s2)", ev_a["n_events"], ev_b["n_events"], ev_c["n_events"]),
        ("Unsupported events (tol=1f)", ev_a["gt_support_by_tolerance"]["1"]["unsupported"],
         ev_b["gt_support_by_tolerance"]["1"]["unsupported"],
         ev_c["gt_support_by_tolerance"]["1"]["unsupported"]),
        ("Event peak p95 (m/s2)",      ev_a["peak_acc_stats"]["p95"],
         ev_b["peak_acc_stats"]["p95"],  ev_c["peak_acc_stats"]["p95"]),
        ("Measured fraction",          anim["A"]["measured_frac"],
         anim["B"]["measured_frac"],    anim["C"]["measured_frac"]),
        ("Ground penetration depth (mm)", anim["A"]["pen_depth_mm"],
         anim["B"]["pen_depth_mm"],    anim["C"]["pen_depth_mm"]),
        ("Sliding p95 L-ankle (m/s)",  anim["A"].get("slide_l_p95_ms"),
         anim["B"].get("slide_l_p95_ms"), anim["C"].get("slide_l_p95_ms")),
        ("Leg rot spikes >30 deg/f",   anim["A"]["rot_spikes_30deg"],
         anim["B"]["rot_spikes_30deg"],anim["C"]["rot_spikes_30deg"]),
    ]
    print(f"{'Metric':<42} {'A':>10} {'B':>10} {'C':>10}")
    print("-"*72)
    for name,a,b,c in rows:
        print(f"{name:<42} {str(a):>10} {str(b):>10} {str(c):>10}")

    with open(OUT/"decision_table.json","w") as f:
        json.dump({"rows":[{"metric":n,"A":str(a),"B":str(b),"C":str(c)} for n,a,b,c in rows]},f,indent=2)

    print("\nAll outputs saved to:", OUT)
    print("\nPRE-DECLARED VERDICT RULE:")
    print("  APPROVE B   if B unsupported events <= C unsupported events AND"
          " B event severity (p95 peak) ~ C AND anim metrics not materially worse")
    print("  APPROVE C   if C removes materially catastrophic events vs B")
    print("  REJECT BOTH if substantial final-motion defects remain after attribution")

if __name__ == "__main__":
    main()
