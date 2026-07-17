import sys
import json
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from datetime import datetime

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
NPZ_PATH = ROOT / "outputs/phase_b_gate1/gate1_arrays.npz"
OUT_DIR = ROOT / "outputs/phase_b_matcher_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BODY_J = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
BODY_J_NAMES = {5: "LShoulder", 6: "RShoulder", 7: "LElbow", 8: "RElbow", 
                9: "LWrist", 10: "RWrist", 11: "LHip", 12: "RHip", 
                13: "LKnee", 14: "RKnee", 15: "LAnkle", 16: "RAnkle"}
FPS = 30.0
EVENT_THR = 5.0

def sha256_array(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class Event:
    source: str
    id: str
    joint: int
    start: int
    end: int
    peak_frame: int
    duration_frames: int
    peak_accel_mps2: float
    peak_vector_mps2: np.ndarray
    delta_v_mps: float
    finite_fraction: float
    global_start: int
    global_end: int
    global_peak: int

    def to_dict(self):
        return {
            "source": self.source,
            "id": self.id,
            "joint": self.joint,
            "start": self.start,
            "end": self.end,
            "peak_frame": self.peak_frame,
            "duration_frames": self.duration_frames,
            "peak_accel_mps2": float(self.peak_accel_mps2),
            "peak_vector_mps2": [float(x) for x in self.peak_vector_mps2],
            "delta_v_mps": float(self.delta_v_mps),
            "finite_fraction": float(self.finite_fraction),
            "global_start": self.global_start,
            "global_end": self.global_end,
            "global_peak": self.global_peak,
        }

def extract_events(pos_cm: np.ndarray, source: str, joint: int, global_offset: int = 0) -> List[Event]:
    T = pos_cm.shape[0]
    if T < 3: return []
    
    velocity_mps = (pos_cm[1:] - pos_cm[:-1]) * FPS / 100.0
    acceleration_mps2 = (velocity_mps[1:] - velocity_mps[:-1]) * FPS
    
    mag = np.linalg.norm(acceleration_mps2, axis=1)
    
    events = []
    in_event = False
    start = 0
    
    t = 0
    while t < len(mag):
        if mag[t] >= EVENT_THR or (in_event and t+1 < len(mag) and mag[t+1] >= EVENT_THR and not np.isnan(mag[t])):
            if not in_event:
                start = t
                in_event = True
        else:
            if in_event:
                end = t - 1
                if np.isnan(mag[start:end+1]).any():
                    in_event = False
                else:
                    peak_idx = start + int(np.argmax(mag[start:end+1]))
                    dur = end - start + 1
                    delta_v = np.linalg.norm(np.sum(acceleration_mps2[start:end+1], axis=0) * (1.0/FPS))
                    ev = Event(
                        source=source,
                        id=f"{source}:j{joint}:s{start}:e{end}:p{peak_idx}",
                        joint=joint,
                        start=start,
                        end=end,
                        peak_frame=peak_idx,
                        duration_frames=dur,
                        peak_accel_mps2=mag[peak_idx],
                        peak_vector_mps2=acceleration_mps2[peak_idx],
                        delta_v_mps=delta_v,
                        finite_fraction=1.0,
                        global_start=start + 1 + global_offset,
                        global_end=end + 1 + global_offset,
                        global_peak=peak_idx + 1 + global_offset
                    )
                    events.append(ev)
                    in_event = False
        t += 1
        
    if in_event:
        end = t - 1
        if not np.isnan(mag[start:end+1]).any():
            peak_idx = start + int(np.argmax(mag[start:end+1]))
            dur = end - start + 1
            delta_v = np.linalg.norm(np.sum(acceleration_mps2[start:end+1], axis=0) * (1.0/FPS))
            ev = Event(
                source=source,
                id=f"{source}:j{joint}:s{start}:e{end}:p{peak_idx}",
                joint=joint,
                start=start,
                end=end,
                peak_frame=peak_idx,
                duration_frames=dur,
                peak_accel_mps2=mag[peak_idx],
                peak_vector_mps2=acceleration_mps2[peak_idx],
                delta_v_mps=delta_v,
                finite_fraction=1.0,
                global_start=start + 1 + global_offset,
                global_end=end + 1 + global_offset,
                global_peak=peak_idx + 1 + global_offset
            )
            events.append(ev)
            
    return events

def match_events_pairwise(preds: List[Event], gts: List[Event], tol: int):
    counts = {"wrong_joint": 0, "interval_distance": 0, "peak_distance": 0, 
              "direction_cosine": 0, "duration_ratio": 0, "dummy": 0}
              
    joints = sorted(list(set([e.joint for e in preds] + [e.joint for e in gts])))
    
    matched = []
    unmatched_preds = []
    unmatched_gts = []
    eps = 1e-9
    
    for j in joints:
        j_preds = sorted([p for p in preds if p.joint == j], key=lambda x: x.id)
        j_gts = sorted([g for g in gts if g.joint == j], key=lambda x: x.id)
        
        N = len(j_preds)
        M = len(j_gts)
        
        if N == 0:
            unmatched_gts.extend(j_gts)
            continue
        if M == 0:
            unmatched_preds.extend(j_preds)
            continue
            
        C = np.zeros((N, M))
        valid = np.zeros((N, M), dtype=bool)
        max_cost = 0.0
        
        for r, p in enumerate(j_preds):
            for c, g in enumerate(j_gts):
                interval_distance = 0 if max(p.start, g.start) <= min(p.end, g.end) else max(p.start, g.start) - min(p.end, g.end)
                peak_distance = abs(p.peak_frame - g.peak_frame)
                
                nx = np.linalg.norm(p.peak_vector_mps2)
                ny = np.linalg.norm(g.peak_vector_mps2)
                if nx > 0 and ny > 0:
                    direction_cosine = np.dot(p.peak_vector_mps2, g.peak_vector_mps2) / (nx * ny)
                else:
                    direction_cosine = 0.0
                    
                dur_ratio = max(p.duration_frames, g.duration_frames) / max(1, min(p.duration_frames, g.duration_frames))
                mag_log = abs(np.log((p.peak_accel_mps2 + 1e-6)/(g.peak_accel_mps2 + 1e-6)))
                
                v = True
                if interval_distance > tol:
                    counts["interval_distance"] += 1
                    v = False
                elif peak_distance > tol + 2:
                    counts["peak_distance"] += 1
                    v = False
                elif direction_cosine < 0.5:
                    counts["direction_cosine"] += 1
                    v = False
                elif dur_ratio > 4.0:
                    counts["duration_ratio"] += 1
                    v = False
                    
                if v:
                    cost = (1.0 * interval_distance + 
                            0.5 * peak_distance + 
                            2.0 * (1.0 - direction_cosine) + 
                            0.5 * mag_log + 
                            0.1 * abs(p.duration_frames - g.duration_frames))
                    C[r, c] = cost
                    valid[r, c] = True
                    max_cost = max(max_cost, cost)
                    
        counts["wrong_joint"] += N * (len(gts) - M)
        unmatched_cost = max_cost + 1.0
        ineligible_cost = unmatched_cost + 1000000.0
        C_aug = np.full((N + M, N + M), ineligible_cost)
        
        for r in range(N):
            for c in range(M):
                if valid[r, c]:
                    C_aug[r, c] = C[r, c] + eps * (r * M + c)
                    
        for r in range(N): C_aug[r, M + r] = unmatched_cost
        for c in range(M): C_aug[N + c, c] = unmatched_cost
        C_aug[N:, M:] = 0.0
        
        row_ind, col_ind = linear_sum_assignment(C_aug)
        
        m_p = set()
        m_g = set()
        
        for r, c in zip(row_ind, col_ind):
            if r < N and c < M and valid[r, c]:
                matched.append((j_preds[r], j_gts[c], C[r, c]))
                m_p.add(r)
                m_g.add(c)
            elif (r < N and c >= M) or (r >= N and c < M):
                counts["dummy"] += 1
                
        for r in range(N):
            if r not in m_p: unmatched_preds.append(j_preds[r])
        for c in range(M):
            if c not in m_g: unmatched_gts.append(j_gts[c])
            
    return matched, unmatched_preds, unmatched_gts, counts

def build_provenance(status="PASS"):
    try:
        with open(__file__, 'rb') as f: script_sha = hashlib.sha256(f.read()).hexdigest()
    except:
        script_sha = ""
    return {
        "script_path": __file__,
        "script_sha256": script_sha,
        "npz_path": str(NPZ_PATH),
        "npz_sha256": sha256_file(NPZ_PATH) if NPZ_PATH.exists() else "",
        "timestamp": datetime.utcnow().isoformat(),
        "fps": FPS,
        "event_threshold": EVENT_THR,
        "joint_list": BODY_J,
        "units": "cm (input), m/s2 (accel)",
        "matching_formula": "1.0*int_dist + 0.5*peak_dist + 2.0*(1-cos) + 0.5*mag_log + 0.1*abs_dur",
        "eligibility_rules": "same_joint, int_dist<=T, peak_dist<=T+2, cos>=0.5, dur_ratio<=4.0",
        "unmatched_penalty_rule": "max_eligible_real_cost + 1.0",
        "seeds": "42 (spot checks)",
        "exclusions": "face/finger joints",
        "status": status
    }

def main():
    prov = build_provenance()
    manifest = {"keys": [], "arrays": {}, "provenance": prov}
    
    if not NPZ_PATH.exists():
        manifest["provenance"]["status"] = "FAIL"
        with open(OUT_DIR / "input_manifest.json", "w") as f: json.dump(manifest, f, indent=2)
        print("FAIL: NPZ not found")
        sys.exit(1)
        
    arrs = np.load(NPZ_PATH, allow_pickle=True)
    manifest["keys"] = list(arrs.keys())
    
    for k in arrs.keys():
        v = arrs[k]
        if v.dtype == object:
            manifest["arrays"][k] = {"shape": v.shape, "dtype": str(v.dtype)}
        else:
            manifest["arrays"][k] = {
                "shape": v.shape, "dtype": str(v.dtype),
                "finite_count": int(np.isfinite(v).sum()),
                "min": float(np.nanmin(v)) if v.size>0 else None,
                "max": float(np.nanmax(v)) if v.size>0 else None,
                "sha256": sha256_array(v)
            }
            
    required = ["b_stage6", "c_stage6", "gt"]
    for req in required:
        if req not in arrs:
            print(f"FAIL: Required array {req} missing.")
            sys.exit(1)
            
    b_s6 = arrs["b_stage6"]
    c_s6 = arrs["c_stage6"]
    gt = arrs["gt"]
    
    a_available = "a_stage6" in arrs.files
    a_s6 = arrs["a_stage6"] if a_available else None
    
    # Check shape 1800
    valid_dims = True
    for arr in [b_s6, c_s6, gt]:
        if arr.shape[0] != 1800:
            valid_dims = False
    if a_available and a_s6.shape[0] != 1800:
        valid_dims = False
        
    if not valid_dims:
        print("FAIL: B, C, and GT (and A if available) cannot all be arrays with 1,800 time steps.")
        sys.exit(1)
        
    with open(OUT_DIR / "input_manifest.json", "w") as f: json.dump(manifest, f, indent=2)
    
    # Convert from mm to cm
    pos_b = b_s6 / 10.0
    pos_c = c_s6 / 10.0
    pos_gt = gt / 10.0
    pos_a = a_s6 / 10.0 if a_available else None
    
    ev_a = []
    if a_available:
        for ji in BODY_J: ev_a.extend(extract_events(pos_a[:, ji], "A", ji))
    ev_b = []
    for ji in BODY_J: ev_b.extend(extract_events(pos_b[:, ji], "B", ji))
    ev_c = []
    for ji in BODY_J: ev_c.extend(extract_events(pos_c[:, ji], "C", ji))
    ev_gt = []
    for ji in BODY_J: ev_gt.extend(extract_events(pos_gt[:, ji], "GT", ji))
    
    # GT Sanity
    gt_seg = {i: 0 for i in range(6)}
    gt_j_counts = {ji: 0 for ji in BODY_J}
    for e in ev_gt:
        gt_j_counts[e.joint] += 1
        gt_seg[e.global_peak // 300] += 1
        
    peak_accels = [e.peak_accel_mps2 for e in ev_gt]
    durs = [e.duration_frames for e in ev_gt]
    
    if len(ev_gt) == 0:
        print("FAIL: total GT events == 0")
        sys.exit(1)
    if any(c == 0 for c in gt_j_counts.values()):
        print("FAIL: every body joint has 0 GT events")
        sys.exit(1)
    if max(peak_accels) < 5.0:
        print("FAIL: GT peak accel max < 5.0")
        sys.exit(1)
    if sum(1 for c in gt_seg.values() if c > 0) == 1:
        print("FAIL: all GT events fall in one segment")
        sys.exit(1)
        
    gt_sum = {
        "total": len(ev_gt),
        "per_joint": gt_j_counts,
        "per_segment": gt_seg,
        "peak_accel": {"median": float(np.median(peak_accels)), "p95": float(np.percentile(peak_accels, 95)), "max": float(np.max(peak_accels))},
        "duration": {"median": float(np.median(durs)), "p95": float(np.percentile(durs, 95))},
        "first_20": [e.to_dict() for e in ev_gt[:20]]
    }
    with open(OUT_DIR / "gt_event_summary.json", "w") as f: json.dump(gt_sum, f, indent=2)
    
    # Real Event Spot Checks
    import random
    rng = random.Random(42)
    b_by_joint = {ji: sorted([e for e in ev_b if e.joint == ji], key=lambda x: x.peak_frame) for ji in BODY_J}
    gt_by_joint = {ji: sorted([e for e in ev_gt if e.joint == ji], key=lambda x: x.peak_frame) for ji in BODY_J}
    
    b_cand = []
    for ji in BODY_J:
        for b in b_by_joint[ji]:
            # Find nearest GT
            if not gt_by_joint[ji]: continue
            near_g = min(gt_by_joint[ji], key=lambda g: abs(g.peak_frame - b.peak_frame))
            pd = abs(b.peak_frame - near_g.peak_frame)
            b_cand.append((b, near_g, pd))
            
    c_0_1 = [x for x in b_cand if 0 <= x[2] <= 1]
    c_2_5 = [x for x in b_cand if 2 <= x[2] <= 5]
    c_gt5 = [x for x in b_cand if x[2] > 5]
    
    sel_spot = []
    if c_0_1: sel_spot.extend(rng.sample(c_0_1, min(5, len(c_0_1))))
    if c_2_5: sel_spot.extend(rng.sample(c_2_5, min(5, len(c_2_5))))
    if c_gt5: sel_spot.extend(rng.sample(c_gt5, min(5, len(c_gt5))))
    
    # Fill remaining to 20
    rem = 20 - len(sel_spot)
    if rem > 0 and b_cand:
        pool = [x for x in b_cand if x not in sel_spot]
        sel_spot.extend(rng.sample(pool, min(rem, len(pool))))
        
    spot_out = []
    for b, g, pd in sel_spot:
        int_dist = 0 if max(b.start, g.start) <= min(b.end, g.end) else max(b.start, g.start) - min(b.end, g.end)
        nx = np.linalg.norm(b.peak_vector_mps2)
        ny = np.linalg.norm(g.peak_vector_mps2)
        cos = np.dot(b.peak_vector_mps2, g.peak_vector_mps2)/(nx*ny) if (nx>0 and ny>0) else 0.0
        dur_ratio = max(b.duration_frames, g.duration_frames) / max(1, min(b.duration_frames, g.duration_frames))
        mag_log = abs(np.log((b.peak_accel_mps2+1e-6)/(g.peak_accel_mps2+1e-6)))
        cost = 1.0*int_dist + 0.5*pd + 2.0*(1-cos) + 0.5*mag_log + 0.1*abs(b.duration_frames-g.duration_frames)
        
        elig = {}
        reason = None
        for T in [0,1,2,3,5]:
            v = True
            r = []
            if int_dist > T: v=False; r.append("interval_dist")
            if pd > T+2: v=False; r.append("peak_dist")
            if cos < 0.5: v=False; r.append("direction_cos")
            if dur_ratio > 4.0: v=False; r.append("duration_ratio")
            elig[str(T)] = v
            if not v and reason is None: reason = r
            
        spot_out.append({
            "b_event": b.to_dict(),
            "gt_event": g.to_dict(),
            "interval_distance": int_dist,
            "peak_distance": pd,
            "direction_cosine": cos,
            "duration_ratio": dur_ratio,
            "magnitude_ratio": mag_log,
            "cost": cost,
            "eligibility": elig,
            "rejection_reason": reason
        })
    with open(OUT_DIR / "real_event_spot_checks.json", "w") as f: json.dump(spot_out, f, indent=2)
    
    # Spot check asserts
    if not any(s["eligibility"]["5"] for s in spot_out):
        print("FAIL: No spot check eligible at T=5")
        sys.exit(1)
    all_reasons = set(tuple(s["rejection_reason"]) for s in spot_out if s["rejection_reason"])
    if len(all_reasons) == 1 and len(spot_out) >= 2:
        # All failed for the EXACT same reason? Wait, some might pass. 
        # "not all 20 fail for the same accidental reason"
        if sum(1 for s in spot_out if not s["eligibility"]["5"]) == len(spot_out):
            print("FAIL: All spot checks failed for the exact same reason")
            sys.exit(1)
            
    # Full GT Matching
    gt_match_results = {}
    
    def process_match(source_evs, gts, T):
        m, up, ug, counts = match_events_pairwise(source_evs, gts, T)
        if len(m) == 0:
            return {"matched": 0, "unsupported": len(up), "missed_gt": len(ug), "support_rate": 0, "recall": 0}
        
        t_errs = [abs(a.peak_frame - b.peak_frame) for a,b,c in m]
        c_errs = []
        for a,b,c in m:
            nx = np.linalg.norm(a.peak_vector_mps2)
            ny = np.linalg.norm(b.peak_vector_mps2)
            c_errs.append(np.dot(a.peak_vector_mps2, b.peak_vector_mps2)/(nx*ny) if nx>0 and ny>0 else 0)
        m_errs = [abs(np.log((a.peak_accel_mps2+1e-6)/(b.peak_accel_mps2+1e-6))) for a,b,c in m]
        costs = [c for a,b,c in m]
        
        return {
            "source_total": len(source_evs),
            "gt_total": len(gts),
            "matched_source": len(m),
            "unsupported_source": len(up),
            "matched_gt": len(gts) - len(ug),
            "missed_gt": len(ug),
            "support_rate": len(m) / max(1, len(source_evs)),
            "gt_recall": (len(gts) - len(ug)) / max(1, len(gts)),
            "timing_error": {"median": float(np.median(t_errs)), "p95": float(np.percentile(t_errs, 95))},
            "direction_cosine": {"median": float(np.median(c_errs)), "p95": float(np.percentile(c_errs, 95))},
            "magnitude_ratio": {"median": float(np.median(m_errs)), "p95": float(np.percentile(m_errs, 95))},
            "matched_cost": {"median": float(np.median(costs)), "p95": float(np.percentile(costs, 95))},
            "rejection_counts": counts
        }
        
    for T in [0, 1, 2, 3, 5]:
        gt_match_results[str(T)] = {
            "B": process_match(ev_b, ev_gt, T),
            "C": process_match(ev_c, ev_gt, T)
        }
        if a_available:
            gt_match_results[str(T)]["A"] = process_match(ev_a, ev_gt, T)
            
    with open(OUT_DIR / "gt_matching.json", "w") as f: json.dump(gt_match_results, f, indent=2)
    
    # HARD STOP
    if gt_match_results["5"]["B"]["matched"] == 0 and gt_match_results["5"]["C"]["matched"] == 0:
        prov["status"] = "INVALID"
        with open(OUT_DIR / "recovery_status.json", "w") as f: json.dump(prov, f, indent=2)
        print("FAIL: B and C both still have zero matches at T=5")
        sys.exit(1)
        
    # B <-> C Matching
    bc_match_results = {}
    b_c_matched_T3 = []
    b_unmatched_T3 = []
    c_unmatched_T3 = []
    
    for T in [1, 3, 5]:
        m, up, ug, counts = match_events_pairwise(ev_b, ev_c, T)
        assert len(m) + len(up) == len(ev_b)
        assert len(m) + len(ug) == len(ev_c)
        
        if T == 3:
            b_c_matched_T3 = m
            b_unmatched_T3 = up
            c_unmatched_T3 = ug
            
        t_errs = [abs(a.peak_frame - b.peak_frame) for a,b,c in m]
        c_errs = []
        for a,b,c in m:
            nx = np.linalg.norm(a.peak_vector_mps2)
            ny = np.linalg.norm(b.peak_vector_mps2)
            c_errs.append(np.dot(a.peak_vector_mps2, b.peak_vector_mps2)/(nx*ny) if nx>0 and ny>0 else 0)
        m_errs = [abs(np.log((a.peak_accel_mps2+1e-6)/(b.peak_accel_mps2+1e-6))) for a,b,c in m]
        dur_errs = [abs(a.duration_frames - b.duration_frames) for a,b,c in m]
        costs = [c for a,b,c in m]
        
        bc_match_results[str(T)] = {
            "b_total": len(ev_b),
            "c_total": len(ev_c),
            "matched": len(m),
            "b_unmatched_to_c": len(up),
            "c_unmatched_to_b": len(ug),
            "matched_cost": {"median": float(np.median(costs)) if costs else 0, "p95": float(np.percentile(costs, 95)) if costs else 0},
            "timing_error": {"median": float(np.median(t_errs)) if t_errs else 0, "p95": float(np.percentile(t_errs, 95)) if t_errs else 0},
            "direction_cosine": {"median": float(np.median(c_errs)) if c_errs else 0, "p95": float(np.percentile(c_errs, 95)) if c_errs else 0},
            "duration_diff": {"median": float(np.median(dur_errs)) if dur_errs else 0, "p95": float(np.percentile(dur_errs, 95)) if dur_errs else 0},
            "magnitude_ratio": {"median": float(np.median(m_errs)) if m_errs else 0, "p95": float(np.percentile(m_errs, 95)) if m_errs else 0},
            "rejection_counts": counts
        }
    with open(OUT_DIR / "bc_matching.json", "w") as f: json.dump(bc_match_results, f, indent=2)
    
    # Causal Sets
    m_b_gt, up_b, ug_b, _ = match_events_pairwise(ev_b, ev_gt, 3)
    m_c_gt, up_c, ug_c, _ = match_events_pairwise(ev_c, ev_gt, 3)
    
    b_supported_ids = set(a.id for a,b,c in m_b_gt)
    c_supported_ids = set(a.id for a,b,c in m_c_gt)
    
    b_unc_gts = [b for b in b_unmatched_T3 if b.id in b_supported_ids]
    b_unc_gtu = [b for b in b_unmatched_T3 if b.id not in b_supported_ids]
    c_unb_gts = [c for c in c_unmatched_T3 if c.id in c_supported_ids]
    c_unb_gtu = [c for c in c_unmatched_T3 if c.id not in c_supported_ids]
    
    material = []
    for b, c, cost in b_c_matched_T3:
        p_ratio = max(b.peak_accel_mps2, c.peak_accel_mps2) / max(1e-6, min(b.peak_accel_mps2, c.peak_accel_mps2))
        abs_diff = abs(b.peak_accel_mps2 - c.peak_accel_mps2)
        dur_diff = abs(b.duration_frames - c.duration_frames)
        if p_ratio >= 2.0 or abs_diff >= 10.0 or dur_diff >= 3:
            material.append({"b": b, "c": c})
            
    def summarize_set(lst):
        if not lst: return {"count": 0}
        durs = [e.duration_frames if hasattr(e, 'duration_frames') else e["b"].duration_frames for e in lst]
        peaks = [e.peak_accel_mps2 if hasattr(e, 'peak_accel_mps2') else e["b"].peak_accel_mps2 for e in lst]
        dvs = [e.delta_v_mps if hasattr(e, 'delta_v_mps') else e["b"].delta_v_mps for e in lst]
        return {
            "count": len(lst),
            "duration": {"median": float(np.median(durs)), "p95": float(np.percentile(durs, 95))},
            "peak": {"median": float(np.median(peaks)), "p95": float(np.percentile(peaks, 95)), "max": float(np.max(peaks))},
            "delta_v": {"median": float(np.median(dvs)), "p95": float(np.percentile(dvs, 95)), "max": float(np.max(dvs))},
            "joints": list(set(e.joint if hasattr(e, 'joint') else e["b"].joint for e in lst)),
            "first_20": [e.id if hasattr(e, 'id') else f"{e['b'].id} <-> {e['c'].id}" for e in lst[:20]]
        }
        
    causal = {
        "B_unmatched_to_C_supported": summarize_set(b_unc_gts),
        "B_unmatched_to_C_unsupported": summarize_set(b_unc_gtu),
        "C_unmatched_to_B_supported": summarize_set(c_unb_gts),
        "C_unmatched_to_B_unsupported": summarize_set(c_unb_gtu),
        "B_C_materially_changed": summarize_set(material)
    }
    with open(OUT_DIR / "causal_sets.json", "w") as f: json.dump(causal, f, indent=2)
    
    prov["status"] = "PASS"
    with open(OUT_DIR / "recovery_status.json", "w") as f: json.dump(prov, f, indent=2)
    print("SUCCESS: Recovery script complete.")

if __name__ == "__main__":
    main()
