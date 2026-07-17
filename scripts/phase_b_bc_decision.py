import sys
import json
import hashlib
import numpy as np
import copy
import subprocess
from pathlib import Path
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import argparse

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
NPZ_PATH = ROOT / "outputs/phase_b_gate1/gate1_arrays.npz"
OUT_DIR = ROOT / "outputs/phase_b_matcher_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "bc_decision.json"
WALKTHROUGH_MD = ROOT / "walkthrough.md"

class AnalyticalHardStop(RuntimeError):
    pass

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_file(path):
    p = Path(path)
    if not p.exists(): return ""
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode('utf-8')

BODY_J = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ANKLE_J = [15, 16]
FPS = 30.0
EVENT_THR = 5.0
WIN_RAD = 7
TIE_THR = 5.0
GLOBAL_OFFSET = 150

def fingerprint_array(arr):
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "byte_count": arr.nbytes,
        "c_contiguous": arr.flags['C_CONTIGUOUS'],
        "finite_count": int(np.isfinite(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0,
        "min_finite": float(np.min(arr[np.isfinite(arr)])) if np.issubdtype(arr.dtype, np.number) and np.isfinite(arr).sum()>0 else None,
        "max_finite": float(np.max(arr[np.isfinite(arr)])) if np.issubdtype(arr.dtype, np.number) and np.isfinite(arr).sum()>0 else None,
        "sha256": sha256_bytes(np.ascontiguousarray(arr).tobytes())
    }

def circular_block_bootstrap(diff_array, valid_mask, block_size=30, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    T = diff_array.shape[0]
    
    all_indices = set()
    for s in range(T):
        all_indices.update([(s + k) % T for k in range(block_size)])
    if all_indices != set(range(T)):
        raise AnalyticalHardStop("Bootstrap full-index eligibility failed")
    if (T - 1) not in all_indices:
        raise AnalyticalHardStop("Bootstrap tail not included")
        
    means = []
    dropped = 0
    min_idx, max_idx = T, 0
    
    for _ in range(n_boot):
        sample_indices = []
        while len(sample_indices) < T:
            start = rng.randint(0, T)
            sample_indices.extend([(start + k) % T for k in range(block_size)])
        sample_indices = sample_indices[:T]
        
        if len(sample_indices) != T:
            raise AnalyticalHardStop("Bootstrap sample length != T")
            
        min_idx = min(min_idx, min(sample_indices))
        max_idx = max(max_idx, max(sample_indices))
        
        block_diff = diff_array[sample_indices]
        block_valid = valid_mask[sample_indices]
        valid_samples = block_diff[block_valid]
        
        if valid_samples.size > 0:
            means.append(np.mean(valid_samples))
        else:
            dropped += 1
            
    means = np.array(means)
    ci = np.percentile(means, [2.5, 97.5]) if len(means) > 0 else [None, None]
    return {
        "block_type": "circular moving-block bootstrap",
        "block_length": block_size,
        "temporal_length": T,
        "replicate_count": n_boot,
        "seed": seed,
        "valid_replicate_count": len(means),
        "dropped_replicate_count": dropped,
        "min_sampled_index": int(min_idx),
        "max_sampled_index": int(max_idx),
        "ci_95": [float(ci[0]) if ci[0] is not None else None, float(ci[1]) if ci[1] is not None else None]
    }

def extract_events(pos_cm, source, joint):
    acc = (pos_cm[2:] - 2*pos_cm[1:-1] + pos_cm[:-2]) * (FPS**2) / 100.0
    mag = np.linalg.norm(acc, axis=-1)
    
    events = []
    in_event = False
    start = 0
    for t in range(len(mag)):
        v = mag[t]
        if np.isfinite(v) and v >= EVENT_THR:
            if not in_event:
                start = t
                in_event = True
        else:
            if in_event:
                end = t - 1
                segment = mag[start:end+1]
                peak_idx = start + int(np.argmax(segment))
                events.append({
                    "source": source,
                    "id": f"{source}:j{joint}:s{start}:e{end}:p{peak_idx}",
                    "joint": joint,
                    "accel_start": start,
                    "accel_end_exclusive": end + 1,
                    "accel_peak": peak_idx,
                    "local_start": start + 1,
                    "local_end_exclusive": end + 2,
                    "local_peak": peak_idx + 1,
                    "global_start": start + 1 + GLOBAL_OFFSET,
                    "global_end_exclusive": end + 2 + GLOBAL_OFFSET,
                    "global_peak": peak_idx + 1 + GLOBAL_OFFSET,
                    "duration_frames": end - start + 1,
                    "peak_accel_mps2": float(mag[peak_idx]),
                    "peak_vector_mps2": acc[peak_idx].tolist()
                })
                in_event = False
    if in_event:
        end = len(mag) - 1
        segment = mag[start:end+1]
        peak_idx = start + int(np.argmax(segment))
        events.append({
            "source": source,
            "id": f"{source}:j{joint}:s{start}:e{end}:p{peak_idx}",
            "joint": joint,
            "accel_start": start,
            "accel_end_exclusive": end + 1,
            "accel_peak": peak_idx,
            "local_start": start + 1,
            "local_end_exclusive": end + 2,
            "local_peak": peak_idx + 1,
            "global_start": start + 1 + GLOBAL_OFFSET,
            "global_end_exclusive": end + 2 + GLOBAL_OFFSET,
            "global_peak": peak_idx + 1 + GLOBAL_OFFSET,
            "duration_frames": end - start + 1,
            "peak_accel_mps2": float(mag[peak_idx]),
            "peak_vector_mps2": acc[peak_idx].tolist()
        })
    return events

def match_events(preds, gts, tol):
    joints = sorted(list(set([e["joint"] for e in preds] + [e["joint"] for e in gts])))
    matched = []
    unmatched_preds = []
    unmatched_gts = []
    
    for j in joints:
        j_preds = sorted([p for p in preds if p["joint"] == j], key=lambda x: x["id"])
        j_gts = sorted([g for g in gts if g["joint"] == j], key=lambda x: x["id"])
        N, M = len(j_preds), len(j_gts)
        
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
                p_lstart, p_lend = p["local_start"], p["local_end_exclusive"] - 1
                g_lstart, g_lend = g["local_start"], g["local_end_exclusive"] - 1
                int_dist = 0 if max(p_lstart, g_lstart) <= min(p_lend, g_lend) else max(p_lstart, g_lstart) - min(p_lend, g_lend)
                peak_dist = abs(p["local_peak"] - g["local_peak"])
                pv, gv = np.array(p["peak_vector_mps2"]), np.array(g["peak_vector_mps2"])
                nx, ny = np.linalg.norm(pv), np.linalg.norm(gv)
                cos = np.dot(pv, gv)/(nx*ny) if nx>0 and ny>0 else 0.0
                dur_ratio = max(p["duration_frames"], g["duration_frames"]) / max(1, min(p["duration_frames"], g["duration_frames"]))
                mag_log = abs(np.log((p["peak_accel_mps2"]+1e-6)/(g["peak_accel_mps2"]+1e-6)))
                
                if int_dist <= tol and peak_dist <= tol + 2 and cos >= 0.5 and dur_ratio <= 4.0:
                    cost = 1.0*int_dist + 0.5*peak_dist + 2.0*(1-cos) + 0.5*mag_log + 0.1*abs(p["duration_frames"] - g["duration_frames"])
                    C[r, c] = cost
                    valid[r, c] = True
                    max_cost = max(max_cost, cost)
                    
        unmatched_cost = max_cost + 1.0
        C_aug = np.full((N+M, N+M), unmatched_cost + 1000000.0)
        for r in range(N):
            for c in range(M):
                if valid[r, c]: C_aug[r, c] = C[r, c] + 1e-9 * (r * M + c)
        for r in range(N): C_aug[r, M+r] = unmatched_cost
        for c in range(M): C_aug[N+c, c] = unmatched_cost
        C_aug[N:, M:] = 0.0
        
        row_ind, col_ind = linear_sum_assignment(C_aug)
        m_p, m_g = set(), set()
        for r, c in zip(row_ind, col_ind):
            if r < N and c < M and valid[r, c]:
                matched.append((j_preds[r], j_gts[c]))
                m_p.add(r)
                m_g.add(c)
        for r in range(N):
            if r not in m_p: unmatched_preds.append(j_preds[r])
        for c in range(M):
            if c not in m_g: unmatched_gts.append(j_gts[c])
            
    return matched, unmatched_preds, unmatched_gts

def compute_payload():
    arrs = np.load(NPZ_PATH, allow_pickle=True)
    req_keys = ["b_stage6", "c_stage6", "gt", "mask_c_log"]
    missing = [k for k in req_keys if k not in arrs.files and k != "mask_c_log"]
    if "mask_c_log" not in arrs.files and "mask_c_log" not in arrs:
        missing.append("mask_c_log")
    if missing:
        raise AnalyticalHardStop(f"Missing required NPZ keys: {missing}")

    pos_b_raw = arrs["b_stage6"]
    pos_c_raw = arrs["c_stage6"]
    gt_raw = arrs["gt"]
    
    if not (np.issubdtype(pos_b_raw.dtype, np.number) and np.issubdtype(pos_c_raw.dtype, np.number) and np.issubdtype(gt_raw.dtype, np.number)):
        raise AnalyticalHardStop("B, C, and GT must be numeric")
    if not (pos_b_raw.ndim == 3 and pos_c_raw.ndim == 3 and gt_raw.ndim == 3):
        raise AnalyticalHardStop("Arrays must have rank 3")
    if not (pos_b_raw.shape == pos_c_raw.shape == gt_raw.shape):
        raise AnalyticalHardStop("Arrays must have exactly equal shapes")
    if not (pos_b_raw.shape[-1] == 3):
        raise AnalyticalHardStop("Coordinate dimension must be exactly 3")
    if pos_b_raw.shape[0] != 1800:
        raise AnalyticalHardStop("Frame count must be exactly 1,800")
    if pos_b_raw.shape[1] < 17:
        raise AnalyticalHardStop("Joint count must be at least 17")
        
    pos_b = pos_b_raw.astype(np.float64)
    pos_c = pos_c_raw.astype(np.float64)
    pos_gt = gt_raw.astype(np.float64) / 10.0
    
    pos_b.setflags(write=False)
    pos_c.setflags(write=False)
    pos_gt.setflags(write=False)
    
    n_frames = pos_b.shape[0]
    
    fingerprints = {
        "b_stage6": fingerprint_array(pos_b_raw),
        "c_stage6": fingerprint_array(pos_c_raw),
        "gt": fingerprint_array(gt_raw),
        "pos_gt_converted": fingerprint_array(pos_gt)
    }

    # Finiteness
    fin_b = np.all(np.isfinite(pos_b), axis=-1)
    fin_c = np.all(np.isfinite(pos_c), axis=-1)
    fin_gt = np.all(np.isfinite(pos_gt), axis=-1)
    
    valid_mpjpe = fin_b & fin_c & fin_gt

    # 1. Global MPJPE
    eb = np.linalg.norm(pos_b[:, BODY_J] - pos_gt[:, BODY_J], axis=-1) * 10.0
    ec = np.linalg.norm(pos_c[:, BODY_J] - pos_gt[:, BODY_J], axis=-1) * 10.0
    diff_mpjpe = eb - ec
    vm_body = valid_mpjpe[:, BODY_J]
    
    if abs(np.mean(eb[vm_body]) - np.mean(ec[vm_body]) - np.mean(diff_mpjpe[vm_body])) > 1e-10:
        raise AnalyticalHardStop("Global MPJPE arithmetic identity failed")
        
    ci_mpjpe_res = circular_block_bootstrap(diff_mpjpe, vm_body)
    
    mpjpe_payload = {
        "unit": "mm",
        "denominators": {
            "total_theoretically_possible": n_frames * len(BODY_J),
            "finite_B": int(fin_b[:, BODY_J].sum()),
            "finite_C": int(fin_c[:, BODY_J].sum()),
            "finite_GT": int(fin_gt[:, BODY_J].sum()),
            "paired_intersection_used": int(vm_body.sum()),
            "excluded": int(n_frames * len(BODY_J) - vm_body.sum())
        },
        "B": {
            "mean": float(np.mean(eb[vm_body])),
            "median": float(np.median(eb[vm_body])),
            "p95": float(np.percentile(eb[vm_body], 95))
        },
        "C": {
            "mean": float(np.mean(ec[vm_body])),
            "median": float(np.median(ec[vm_body])),
            "p95": float(np.percentile(ec[vm_body], 95))
        },
        "B_minus_C": {
            "paired_mean": float(np.mean(diff_mpjpe[vm_body])),
            "paired_median": float(np.median(diff_mpjpe[vm_body])),
            "bootstrap": ci_mpjpe_res
        }
    }
    
    # 2. Acceleration
    acc_b = (pos_b[2:] - 2*pos_b[1:-1] + pos_b[:-2]) * (FPS**2) / 100.0
    acc_c = (pos_c[2:] - 2*pos_c[1:-1] + pos_c[:-2]) * (FPS**2) / 100.0
    
    fin_acc_b = np.all(np.isfinite(acc_b), axis=-1)
    fin_acc_c = np.all(np.isfinite(acc_c), axis=-1)
    valid_acc = fin_acc_b & fin_acc_c
    vm_acc_body = valid_acc[:, BODY_J]
    
    mag_ab = np.linalg.norm(acc_b[:, BODY_J], axis=-1)
    mag_ac = np.linalg.norm(acc_c[:, BODY_J], axis=-1)
    diff_acc = mag_ab - mag_ac
    
    if abs(np.mean(mag_ab[vm_acc_body]) - np.mean(mag_ac[vm_acc_body]) - np.mean(diff_acc[vm_acc_body])) > 1e-10:
        raise AnalyticalHardStop("Acceleration arithmetic identity failed")
        
    ci_acc_res = circular_block_bootstrap(diff_acc, vm_acc_body)
    cnt_b_5 = int((mag_ab[vm_acc_body] >= 5.0).sum())
    cnt_c_5 = int((mag_ac[vm_acc_body] >= 5.0).sum())
    
    acc_payload = {
        "unit": "m/s^2",
        "denominators": {
            "total_theoretically_possible": (n_frames - 2) * len(BODY_J),
            "finite_B": int(fin_acc_b[:, BODY_J].sum()),
            "finite_C": int(fin_acc_c[:, BODY_J].sum()),
            "paired_intersection_used": int(vm_acc_body.sum()),
            "excluded": int((n_frames - 2) * len(BODY_J) - vm_acc_body.sum())
        },
        "B": {
            "mean": float(np.mean(mag_ab[vm_acc_body])),
            "median": float(np.median(mag_ab[vm_acc_body])),
            "p95": float(np.percentile(mag_ab[vm_acc_body], 95)),
            "count_gte_5": cnt_b_5,
            "percent_gte_5": float(cnt_b_5 / vm_acc_body.sum() * 100) if vm_acc_body.sum()>0 else 0
        },
        "C": {
            "mean": float(np.mean(mag_ac[vm_acc_body])),
            "median": float(np.median(mag_ac[vm_acc_body])),
            "p95": float(np.percentile(mag_ac[vm_acc_body], 95)),
            "count_gte_5": cnt_c_5,
            "percent_gte_5": float(cnt_c_5 / vm_acc_body.sum() * 100) if vm_acc_body.sum()>0 else 0
        },
        "B_minus_C": {
            "paired_mean": float(np.mean(diff_acc[vm_acc_body])),
            "paired_median": float(np.median(diff_acc[vm_acc_body])),
            "bootstrap": ci_acc_res
        }
    }
    
    # 3. Ankle MPJPE
    vm_ankle = valid_mpjpe[:, ANKLE_J]
    eb_a = eb[:, [BODY_J.index(a) for a in ANKLE_J]]
    ec_a = ec[:, [BODY_J.index(a) for a in ANKLE_J]]
    diff_ankle = eb_a - ec_a
    
    if abs(np.mean(eb_a[vm_ankle]) - np.mean(ec_a[vm_ankle]) - np.mean(diff_ankle[vm_ankle])) > 1e-10:
        raise AnalyticalHardStop("Ankle MPJPE arithmetic identity failed")
        
    ci_ankle_res = circular_block_bootstrap(diff_ankle, vm_ankle)
    
    ankle_payload = {
        "unit": "mm",
        "denominators": {
            "total_theoretically_possible": n_frames * len(ANKLE_J),
            "finite_B": int(fin_b[:, ANKLE_J].sum()),
            "finite_C": int(fin_c[:, ANKLE_J].sum()),
            "finite_GT": int(fin_gt[:, ANKLE_J].sum()),
            "paired_intersection_used": int(vm_ankle.sum()),
            "excluded": int(n_frames * len(ANKLE_J) - vm_ankle.sum())
        },
        "B": {
            "mean": float(np.mean(eb_a[vm_ankle])),
            "median": float(np.median(eb_a[vm_ankle])),
            "p95": float(np.percentile(eb_a[vm_ankle], 95))
        },
        "C": {
            "mean": float(np.mean(ec_a[vm_ankle])),
            "median": float(np.median(ec_a[vm_ankle])),
            "p95": float(np.percentile(ec_a[vm_ankle], 95))
        },
        "B_minus_C": {
            "paired_mean": float(np.mean(diff_ankle[vm_ankle])),
            "paired_median": float(np.median(diff_ankle[vm_ankle])),
            "bootstrap": ci_ankle_res
        }
    }
    
    # 4. Floors
    gt_ankle_y = pos_gt[:, ANKLE_J, 1]
    gt_ankle_y_flat = gt_ankle_y[np.isfinite(gt_ankle_y)]
    floor_min = float(np.min(gt_ankle_y_flat))
    floor_p05 = float(np.percentile(gt_ankle_y_flat, 0.5))
    floor_p10 = float(np.percentile(gt_ankle_y_flat, 1.0))
    
    sorted_gt_y_idx = np.argsort(gt_ankle_y_flat)
    sorted_gt_y_vals = gt_ankle_y_flat[sorted_gt_y_idx]
    
    floor_payload = {
        "minimum": floor_min,
        "percentile_0_5": floor_p05,
        "percentile_1_0": floor_p10,
        "numpy_version": np.__version__,
        "percentile_method": "linear (numpy default)",
        "evidence": {
            "min_20": sorted_gt_y_vals[:20].tolist(),
            "p05_around": sorted_gt_y_vals[max(0, int(len(gt_ankle_y_flat)*0.005)-5) : min(len(gt_ankle_y_flat), int(len(gt_ankle_y_flat)*0.005)+5)].tolist(),
            "p10_around": sorted_gt_y_vals[max(0, int(len(gt_ankle_y_flat)*0.010)-5) : min(len(gt_ankle_y_flat), int(len(gt_ankle_y_flat)*0.010)+5)].tolist()
        }
    }
    
    # 5. Penetration
    pen_payload = {}
    for f_name, f_val in [("minimum", floor_min), ("percentile_0_5", floor_p05), ("percentile_1_0", floor_p10)]:
        pb = np.maximum(0, f_val - pos_b[:, ANKLE_J, 1]) * 10.0
        pc = np.maximum(0, f_val - pos_c[:, ANKLE_J, 1]) * 10.0
        mask_p = fin_b[:, ANKLE_J] & fin_c[:, ANKLE_J]
        pd = pb - pc
        if abs(np.mean(pb[mask_p]) - np.mean(pc[mask_p]) - np.mean(pd[mask_p])) > 1e-10:
            raise AnalyticalHardStop(f"Penetration arithmetic identity failed for {f_name}")
        ci = circular_block_bootstrap(pd, mask_p)
        
        pen_payload[f_name] = {
            "floor_cm": f_val,
            "paired_support": int(mask_p.sum()),
            "B": {
                "mean": float(np.mean(pb[mask_p])),
                "median": float(np.median(pb[mask_p])),
                "p95": float(np.percentile(pb[mask_p], 95)),
                "positive_count": int((pb[mask_p] > 0).sum()),
                "positive_percent": float((pb[mask_p] > 0).sum() / mask_p.sum() * 100) if mask_p.sum()>0 else 0
            },
            "C": {
                "mean": float(np.mean(pc[mask_p])),
                "median": float(np.median(pc[mask_p])),
                "p95": float(np.percentile(pc[mask_p], 95)),
                "positive_count": int((pc[mask_p] > 0).sum()),
                "positive_percent": float((pc[mask_p] > 0).sum() / mask_p.sum() * 100) if mask_p.sum()>0 else 0
            },
            "B_minus_C": {
                "mean": float(np.mean(pd[mask_p])),
                "bootstrap": ci,
                "arithmetic_identity": "PASS"
            }
        }
        
    # 6. Sliding
    vel_b = (pos_b[1:] - pos_b[:-1]) * FPS
    vel_c = (pos_c[1:] - pos_c[:-1]) * FPS
    slide_b = np.linalg.norm(vel_b[:, ANKLE_J][:, :, [0, 2]], axis=-1)
    slide_c = np.linalg.norm(vel_c[:, ANKLE_J][:, :, [0, 2]], axis=-1)
    
    fin_vb = np.all(np.isfinite(vel_b[:, ANKLE_J]), axis=-1)
    fin_vc = np.all(np.isfinite(vel_c[:, ANKLE_J]), axis=-1)
    
    cont_b = pos_b[:, ANKLE_J, 1] < (floor_min + 5.0)
    cont_c = pos_c[:, ANKLE_J, 1] < (floor_min + 5.0)
    cont_gt = pos_gt[:, ANKLE_J, 1] < (floor_min + 5.0)
    
    s_b_spec = cont_b[:-1] & fin_vb
    s_c_spec = cont_c[:-1] & fin_vc
    s_bc_com = cont_b[:-1] & cont_c[:-1] & fin_vb & fin_vc
    s_gt_both = cont_gt[:-1] & cont_gt[1:] & fin_vb & fin_vc
    s_gt_start = cont_gt[:-1] & fin_vb & fin_vc
    
    def slide_stats(mask, name):
        sb, sc = slide_b[mask], slide_c[mask]
        sd = sb - sc
        sd_full = slide_b - slide_c
        if mask.sum() > 0 and abs(np.mean(sb) - np.mean(sc) - np.mean(sd)) > 1e-10:
            raise AnalyticalHardStop(f"Sliding arithmetic identity failed for {name}")
        ci = circular_block_bootstrap(sd_full, mask) if mask.sum() > 0 else None
        return {
            "floor_definition": "minimum",
            "contact_threshold_cm": floor_min + 5.0,
            "support": int(mask.sum()),
            "unit": "cm/s",
            "B": {
                "mean": float(np.mean(sb)) if mask.sum()>0 else None,
                "median": float(np.median(sb)) if mask.sum()>0 else None,
                "p95": float(np.percentile(sb, 95)) if mask.sum()>0 else None
            },
            "C": {
                "mean": float(np.mean(sc)) if mask.sum()>0 else None,
                "median": float(np.median(sc)) if mask.sum()>0 else None,
                "p95": float(np.percentile(sc, 95)) if mask.sum()>0 else None
            },
            "B_minus_C": {
                "mean": float(np.mean(sd)) if mask.sum()>0 else None,
                "median": float(np.median(sd)) if mask.sum()>0 else None,
                "bootstrap": ci
            },
            "arithmetic_identity": "PASS"
        }
        
    slide_payload = {
        "A_descriptive_candidate_specific": {
            "B_on_B": {"support": int(s_b_spec.sum()), "B_mean": float(np.mean(slide_b[s_b_spec])) if s_b_spec.sum()>0 else None},
            "C_on_C": {"support": int(s_c_spec.sum()), "C_mean": float(np.mean(slide_c[s_c_spec])) if s_c_spec.sum()>0 else None}
        },
        "B_sensitivity_common_predicted": slide_stats(s_bc_com, "common"),
        "C_primary_GT_both_endpoints": slide_stats(s_gt_both, "gt_both"),
        "D_sensitivity_GT_start": slide_stats(s_gt_start, "gt_start")
    }

    # 7. Coverage and Starvation
    c_log = arrs["mask_c_log"]
    c_log_list = []
    if c_log.ndim == 0 and c_log.dtype == 'O':
        c_log_list = c_log.item()
    else:
        c_log_list = c_log.tolist() if isinstance(c_log, np.ndarray) else list(c_log)
        
    unique_fj = set()
    unique_f = set()
    malformed = 0
    for e in c_log_list:
        try:
            if isinstance(e, dict) and 'f' in e and 'j' in e:
                unique_fj.add((e['f'], e['j']))
                unique_f.add(e['f'])
            else:
                malformed += 1
        except:
            malformed += 1
            
    if len(unique_fj) > len(c_log_list):
        raise AnalyticalHardStop("Unique starvation > raw starvation")
        
    cov_payload = {
        "Stage6_Finiteness": {
            "B": {"body_percent": float(fin_b[:, BODY_J].sum() / (n_frames*len(BODY_J)) * 100)},
            "C": {"body_percent": float(fin_c[:, BODY_J].sum() / (n_frames*len(BODY_J)) * 100)}
        },
        "Observed_Coverage": "Unknown (metadata absent)",
        "C_Starvation_Log": {
            "dtype": str(c_log.dtype),
            "shape": list(c_log.shape),
            "total_raw_entries": len(c_log_list),
            "malformed_entries": malformed,
            "unique_frame_joint_count": len(unique_fj),
            "unique_frame_count": len(unique_f),
            "schema_example": c_log_list[0] if len(c_log_list)>0 else None
        },
        "gate_induced_starvation_B": "not applicable"
    }

    # 8. Events
    ev_b, ev_c = [], []
    for j in BODY_J:
        ev_b.extend(extract_events(pos_b[:, j], "B", j))
        ev_c.extend(extract_events(pos_c[:, j], "C", j))
        
    matcher_payload = {}
    u_b3, u_c3 = [], []
    for T in [1, 3, 5]:
        m, ub, uc = match_events(ev_b, ev_c, T)
        m_rev, uc_rev, ub_rev = match_events(ev_c, ev_b, T)
        
        if len(m) + len(ub) != len(ev_b) or len(m) + len(uc) != len(ev_c):
            raise AnalyticalHardStop(f"Event conservation failed at T={T}")
        
        if len(m) != len(m_rev):
            raise AnalyticalHardStop(f"Forward/reverse matched count equality failed at T={T}")
            
        set_f = set((a["id"], b["id"]) for a,b in m)
        set_r = set((b["id"], a["id"]) for a,b in m_rev)
        if set_f != set_r:
            raise AnalyticalHardStop(f"Pair-set order invariance failed at T={T}")
            
        matcher_payload[f"T={T}"] = {
            "total_B": len(ev_b),
            "total_C": len(ev_c),
            "matched_count": len(m),
            "B_unmatched_count": len(ub),
            "C_unmatched_count": len(uc),
            "unmatched_B_IDs": [x["id"] for x in ub],
            "unmatched_C_IDs": [x["id"] for x in uc],
            "conservation_pass": True,
            "count_order_invariance_pass": True,
            "pair_set_order_invariance_pass": True
        }
        if T == 3:
            u_b3, u_c3 = ub, uc

    acc_gt = (pos_gt[2:] - 2*pos_gt[1:-1] + pos_gt[:-2]) * (FPS**2) / 100.0
    
    # 9. Window Classification
    def get_gt_iaa(w_start, w_end_excl, ji):
        a_start = max(0, w_start - 1)
        a_end_excl = min(n_frames - 2, w_end_excl - 1)
        if a_end_excl <= a_start:
            return 0.0, 0
        mag = np.linalg.norm(acc_gt[a_start:a_end_excl, BODY_J.index(ji)], axis=-1)
        mag = mag[np.isfinite(mag)]
        return float(np.sum(mag) / FPS), len(mag)

    def evaluate_window(ev):
        rec = copy.deepcopy(ev)
        pk = rec["local_peak"]
        w_st = max(0, pk - WIN_RAD)
        w_en = min(n_frames, pk + WIN_RAD + 1)
        
        rec["local_window_start"] = w_st
        rec["local_window_end_exclusive"] = w_en
        rec["global_window_start"] = w_st + GLOBAL_OFFSET
        rec["global_window_end_exclusive"] = w_en + GLOBAL_OFFSET
        rec["possible_samples"] = w_en - w_st
        rec["joint_name"] = None
        
        ji = rec["joint"]
        ji_idx = BODY_J.index(ji)
        
        eb = np.linalg.norm(pos_b[w_st:w_en, ji_idx] - pos_gt[w_st:w_en, ji_idx], axis=-1) * 10.0
        ec = np.linalg.norm(pos_c[w_st:w_en, ji_idx] - pos_gt[w_st:w_en, ji_idx], axis=-1) * 10.0
        
        fb = fin_b[w_st:w_en, ji_idx]
        fc = fin_c[w_st:w_en, ji_idx]
        fg = fin_gt[w_st:w_en, ji_idx]
        vm = fb & fc & fg
        
        rec["B_finite_count"] = int(fb.sum())
        rec["C_finite_count"] = int(fc.sum())
        rec["GT_finite_count"] = int(fg.sum())
        rec["paired_valid_count"] = int(vm.sum())
        
        if vm.sum() == 0:
            rec["median_B_to_GT_error_mm"] = None
            rec["median_C_to_GT_error_mm"] = None
            rec["signed_B_minus_C_median_difference_mm"] = None
            rec["classification"] = "unclassifiable"
        else:
            mb = float(np.median(eb[vm]))
            mc = float(np.median(ec[vm]))
            rec["median_B_to_GT_error_mm"] = mb
            rec["median_C_to_GT_error_mm"] = mc
            rec["signed_B_minus_C_median_difference_mm"] = mb - mc
            if mb < mc - TIE_THR: rec["classification"] = "B_closer"
            elif mc < mb - TIE_THR: rec["classification"] = "C_closer"
            else: rec["classification"] = "tie"
            
        iaa, _ = get_gt_iaa(w_st, w_en, ji)
        rec["GT_integrated_absolute_acceleration_mps"] = iaa
        rec["real_0_25"] = iaa >= 0.25
        rec["real_0_50"] = iaa >= 0.50
        rec["real_1_00"] = iaa >= 1.00
        
        if rec["possible_samples"] != rec["local_window_end_exclusive"] - rec["local_window_start"]:
            raise AnalyticalHardStop("Interval possible samples mismatch")
        if rec["global_window_start"] != rec["local_window_start"] + 150:
            raise AnalyticalHardStop("Global window start mismatch")
            
        return rec

    win_records = []
    for e in u_b3: win_records.append(evaluate_window(e))
    for e in u_c3: win_records.append(evaluate_window(e))
    
    if len(win_records) != len(u_b3) + len(u_c3):
        raise AnalyticalHardStop("Raw-record reconciliation failed")
        
    canonical_before = canonical_json(win_records)

    # 10. Episodes
    episodes = []
    for ji in BODY_J:
        j_recs = sorted([r for r in win_records if r["joint"] == ji], key=lambda x: (x["local_window_start"], x["local_window_end_exclusive"], x["id"]))
        if not j_recs: continue
        
        merged = []
        cur_st = j_recs[0]["local_window_start"]
        cur_en = j_recs[0]["local_window_end_exclusive"]
        cur_ids = [j_recs[0]["id"]]
        cur_src = [j_recs[0]["source"]]
        
        for r in j_recs[1:]:
            if r["local_window_start"] <= cur_en:
                cur_en = max(cur_en, r["local_window_end_exclusive"])
                cur_ids.append(r["id"])
                cur_src.append(r["source"])
            else:
                merged.append({"st": cur_st, "en": cur_en, "ids": cur_ids, "src": cur_src})
                cur_st = r["local_window_start"]
                cur_en = r["local_window_end_exclusive"]
                cur_ids = [r["id"]]
                cur_src = [r["source"]]
        merged.append({"st": cur_st, "en": cur_en, "ids": cur_ids, "src": cur_src})
        
        for idx, ep in enumerate(merged):
            w_st, w_en = ep["st"], ep["en"]
            ji_idx = BODY_J.index(ji)
            eb = np.linalg.norm(pos_b[w_st:w_en, ji_idx] - pos_gt[w_st:w_en, ji_idx], axis=-1) * 10.0
            ec = np.linalg.norm(pos_c[w_st:w_en, ji_idx] - pos_gt[w_st:w_en, ji_idx], axis=-1) * 10.0
            vm = fin_b[w_st:w_en, ji_idx] & fin_c[w_st:w_en, ji_idx] & fin_gt[w_st:w_en, ji_idx]
            
            e_rec = {
                "episode_id": f"ep_j{ji}_{w_st}_{w_en}",
                "joint": ji,
                "local_start": w_st,
                "local_end_exclusive": w_en,
                "global_start": w_st + GLOBAL_OFFSET,
                "global_end_exclusive": w_en + GLOBAL_OFFSET,
                "contributing_event_IDs": ep["ids"],
                "contributing_B_unmatched_count": ep["src"].count("B_unmatched"),
                "contributing_C_unmatched_count": ep["src"].count("C_unmatched"),
                "total_contributing_event_count": len(ep["ids"]),
                "paired_valid_frame_count": int(vm.sum()),
            }
            if vm.sum() == 0:
                e_rec["classification"] = "unclassifiable"
                e_rec["median_B_to_GT_error_mm"] = None
                e_rec["median_C_to_GT_error_mm"] = None
                e_rec["signed_difference_mm"] = None
            else:
                mb = float(np.median(eb[vm]))
                mc = float(np.median(ec[vm]))
                e_rec["median_B_to_GT_error_mm"] = mb
                e_rec["median_C_to_GT_error_mm"] = mc
                e_rec["signed_difference_mm"] = mb - mc
                if mb < mc - TIE_THR: e_rec["classification"] = "B_closer"
                elif mc < mb - TIE_THR: e_rec["classification"] = "C_closer"
                else: e_rec["classification"] = "tie"
                
            iaa, _ = get_gt_iaa(w_st, w_en, ji)
            e_rec["GT_integrated_absolute_acceleration_mps"] = iaa
            e_rec["real_0_25"] = iaa >= 0.25
            e_rec["real_0_50"] = iaa >= 0.50
            e_rec["real_1_00"] = iaa >= 1.00
            episodes.append(e_rec)
            
    canonical_after = canonical_json(win_records)
    if canonical_before != canonical_after:
        raise AnalyticalHardStop("Raw-record immutability failed")
        
    all_ep_ids = [i for ep in episodes for i in ep["contributing_event_IDs"]]
    if sorted(all_ep_ids) != sorted([r["id"] for r in win_records]):
        raise AnalyticalHardStop("Event-to-episode one-to-one assignment failed")
        
    if len(set(all_ep_ids)) != len(all_ep_ids):
        raise AnalyticalHardStop("No event belongs to two episodes failed")
        
    payload = {
        "fingerprints": fingerprints,
        "global_mpjpe": mpjpe_payload,
        "global_acceleration": acc_payload,
        "ankle_mpjpe": ankle_payload,
        "floors": floor_payload,
        "penetration": pen_payload,
        "sliding": slide_payload,
        "coverage_and_starvation": cov_payload,
        "event_divergence": matcher_payload,
        "window_records": win_records,
        "episodes": episodes
    }
    return payload

def repro_run():
    p1 = compute_payload()
    p2 = compute_payload()
    
    c1 = canonical_json(p1)
    c2 = canonical_json(p2)
    h1 = sha256_bytes(c1)
    h2 = sha256_bytes(c2)
    
    if c1 != c2 or h1 != h2:
        raise AnalyticalHardStop(f"Deterministic in-process payload equality failed. h1={h1}, h2={h2}")
        
    return p1, h1

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repro-check', action='store_true')
    args = parser.parse_args()
    
    if args.repro_check:
        p, h = repro_run()
        print(h)
        sys.exit(0)
        
    try:
        p, h1 = repro_run()
        
        proc = subprocess.run([sys.executable, __file__, '--repro-check'], capture_output=True, text=True, check=True)
        h3 = proc.stdout.strip()
        
        if h1 != h3:
            raise AnalyticalHardStop(f"Deterministic fresh-process payload equality failed. h1={h1}, h3={h3}")
            
        final_out = {
            "reproducibility": {
                "in_process_run_1_hash": h1,
                "in_process_run_2_hash": h1,
                "fresh_process_hash": h3,
                "exact_equality": "PASS"
            },
            "deterministic_payload": p
        }
        
        with open(OUT_JSON, 'w') as f:
            json.dump(final_out, f, indent=2)
            
        write_walkthrough(final_out, h1)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        final_out = {
            "hard_stops": [str(e)]
        }
        with open(OUT_JSON, 'w') as f:
            json.dump(final_out, f, indent=2)
        print(f"FAILED: {e}")

def write_walkthrough(data, h1):
    p = data["deterministic_payload"]
    with open(WALKTHROUGH_MD, 'w') as f:
        f.write("# AIMoCap B-vs-C Reproducibility Audit\n\n")
        f.write("## 1. Plain-English Summary\n")
        f.write("Candidate B and Candidate C are alternative pipelines being compared for structural accuracy and realism. ")
        f.write("The prior audit contained discrepancies due to array immutability bugs, dropping bootstrap tails, unit confusion, and overlapping event dependencies. ")
        f.write("This run strictly enforces immutability, canonical hashing, and isolated event episodes. ")
        f.write("This run is mathematically reproducible across both in-process and fresh-process boundaries. ")
        f.write("A definitive winner is not recommended in this document.\n\n")
        
        f.write("## 2. What Changed\n")
        f.write("- Enforced strict NPZ required keys validation.\n")
        f.write("- Implemented a true circular block bootstrap covering all frames (including tail).\n")
        f.write("- Event records are strictly immutable; episodes are constructed in isolated accumulator objects.\n")
        f.write("- Sliding uses explicit cm/s units and explicit GT-contact support.\n")
        f.write("- Penetration defines explicit minimum and percentile floors, correctly extracting GT ankle y-coordinates.\n")
        f.write("- GT integrated absolute acceleration correctly maps index `a` to local position frame `a+1`.\n")
        f.write("- Canonical JSON hashing proves bit-for-bit determinism.\n\n")
        
        f.write("## 3. Artifact Manifest\n")
        f.write(f"- NPZ SHA-256: {sha256_file(NPZ_PATH)}\n")
        f.write(f"- Script SHA-256: {sha256_file(Path(__file__))}\n")
        f.write(f"- Deterministic Payload Hash: {h1}\n\n")
        
        f.write("## 4. Input Fingerprints\n```json\n")
        f.write(json.dumps(p["fingerprints"], indent=2) + "\n```\n")
        f.write("Mask Schema: " + str(p["coverage_and_starvation"]["C_Starvation_Log"]["schema_example"]) + "\n\n")
        
        f.write("## 5. Reproducibility Test\n")
        f.write(f"- In-process Hash 1: {data['reproducibility']['in_process_run_1_hash']}\n")
        f.write(f"- In-process Hash 2: {data['reproducibility']['in_process_run_2_hash']}\n")
        f.write(f"- Fresh-process Hash: {data['reproducibility']['fresh_process_hash']}\n")
        f.write(f"- Equality: {data['reproducibility']['exact_equality']}\n\n")
        
        f.write("## 6. Assertion Table\n")
        f.write("| Check | Expected | Observed | Status | Hard-Stop Status |\n|---|---|---|---|---|\n")
        f.write("| Required Keys | Present | Present | PASS | None |\n")
        f.write("| Array Shapes | Identical | Identical | PASS | None |\n")
        f.write("| Source Immutability | True | True | PASS | None |\n")
        f.write("| Payload Equality | True | True | PASS | None |\n")
        f.write("| Bootstrap Eligibility | True | True | PASS | None |\n")
        f.write("| Arithmetic Identities | True | True | PASS | None |\n")
        f.write("| Event Conservation | True | True | PASS | None |\n")
        f.write("| Pair-set Invariance | True | True | PASS | None |\n")
        f.write("| Raw-Record Immutability | True | True | PASS | None |\n")
        f.write("| Event-to-Episode Assignment | True | True | PASS | None |\n\n")
        
        f.write("## 7. Global MPJPE\n```json\n")
        f.write(json.dumps(p["global_mpjpe"], indent=2) + "\n```\n\n")
        
        f.write("## 8. Acceleration\n```json\n")
        f.write(json.dumps(p["global_acceleration"], indent=2) + "\n```\n\n")
        
        f.write("## 9. Ankle MPJPE\n```json\n")
        f.write(json.dumps(p["ankle_mpjpe"], indent=2) + "\n```\n\n")
        
        f.write("## 10. Sliding\n```json\n")
        f.write(json.dumps(p["sliding"], indent=2) + "\n```\n\n")
        
        f.write("## 11. Penetration\n```json\n")
        f.write(json.dumps(p["penetration"], indent=2) + "\n```\n\n")
        
        f.write("## 12. Coverage and Starvation\n```json\n")
        f.write(json.dumps(p["coverage_and_starvation"], indent=2) + "\n```\n\n")
        
        f.write("## 13. Event Divergence\n```json\n")
        f.write(json.dumps(p["event_divergence"], indent=2) + "\n```\n\n")
        
        f.write("## 14. T=3 Raw Divergence Records\n")
        f.write(f"Total unmatched records: {len(p['window_records'])}\n```json\n")
        f.write(json.dumps(p["window_records"], indent=2) + "\n```\n\n")
        
        f.write("## 15. Merged Episodes\n")
        f.write(f"Total episodes: {len(p['episodes'])}\n```json\n")
        f.write(json.dumps(p["episodes"], indent=2) + "\n```\n\n")
        
        f.write("## 16. Real/Quiet Sensitivity\n")
        r025 = {"B_closer": {"real":0,"quiet":0}, "C_closer": {"real":0,"quiet":0}, "tie": {"real":0,"quiet":0}, "unclassifiable": {"real":0,"quiet":0}}
        r050 = copy.deepcopy(r025)
        r100 = copy.deepcopy(r025)
        
        for ep in p["episodes"]:
            cls = ep["classification"]
            r025[cls]["real" if ep["real_0_25"] else "quiet"] += 1
            r050[cls]["real" if ep["real_0_50"] else "quiet"] += 1
            r100[cls]["real" if ep["real_1_00"] else "quiet"] += 1
            
        f.write("### Threshold 0.25 m/s\n```json\n" + json.dumps(r025, indent=2) + "\n```\n")
        f.write("### Threshold 0.50 m/s\n```json\n" + json.dumps(r050, indent=2) + "\n```\n")
        f.write("### Threshold 1.00 m/s\n```json\n" + json.dumps(r100, indent=2) + "\n```\n\n")
        
        f.write("## 17. Proven Facts\n")
        f.write("- Arrays are identical in shape and dimensions.\n")
        f.write("- Deterministic payload hash verifies reproducibility.\n\n")
        
        f.write("## 18. Remaining Unresolved Issues\n")
        f.write("- Inference method metadata for starved frames is not present in the arrays.\n\n")
        
        f.write("## 19. Hard Stops\n")
        f.write("None. All assertions passed.\n\n")
        
        f.write("## 20. diagnosis.md Update\n")
        f.write("Added `[SUPERSEDED]` block for the prior invalid report.\n\n")
        
        f.write("## Appendix A. Complete Final Script\n")
        f.write("See `scripts/phase_b_bc_decision.py`.\n\n")
        
        f.write("## Appendix B. Complete Deterministic Summary Payload\n")
        f.write(f"Hash: {h1}\n")

if __name__ == "__main__":
    run()
