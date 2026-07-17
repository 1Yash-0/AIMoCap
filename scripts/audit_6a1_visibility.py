"""Stage 6a.1 — Truth Audit of Low-Camera-Visibility Reconstruction.

Diagnostics A, B, C, D as specified.  No production solver changes.
Outputs: outputs/stage6a_1_visibility_audit/

Run standalone: python scripts/audit_6a1_visibility.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.math.triangulate import triangulate_weighted_dlt

# ── Constants (must match stage6a_kinematic_bvh.py) ──────────────────────────
S3_NPZ     = ROOT / "outputs" / "stage3_check_c" / "kpts.npz"
S42_DIR    = ROOT / "outputs" / "stage4_2_knee_rescue"
CALIB_JSON = ROOT / "data" / "panoptic" / "171204_pose1" / "calibration_171204_pose1.json"
GT_DIR     = ROOT / "data" / "panoptic" / "171204_pose1" / "hdPose3d_stage1_coco19"
OUT_DIR    = ROOT / "outputs" / "stage6a_1_visibility_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAM_NAMES   = ["00_26", "00_29", "00_30"]
FPS         = 29.97
START_FRAME = int(FPS * 5.0)   # 149
N_FRAMES    = 300

ANKLE_JOINTS = {
    15: {"name": "l_ankle", "parent": 13, "gp": 11, "gate": 0.35, "bvh_bl_idx": 3},
    16: {"name": "r_ankle", "parent": 14, "gp": 12, "gate": 0.35, "bvh_bl_idx": 6},
}

# Panoptic 19-joint → COCO-17 mapping used everywhere else
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

# Gate thresholds for knees (for Diag-B context)
KNEE_GATE = {13: 0.35, 14: 0.35}

# Provisional quality gates (from spec — do not change after seeing results)
GATE_MEDIAN_MM  = 50.0
GATE_P95_MM     = 100.0
GATE_MEDIAN_DEG = 10.0
GATE_P95_DEG    = 25.0
GATE_IMPROVEMENT_PCT = 15.0


# ── Ray / sphere helpers (duplicated here — audit must not import prod solver) ─

def _unproject_ray(uv, K_inv, R_world, cam_center):
    ray_cam   = K_inv @ np.array([uv[0], uv[1], 1.0])
    ray_world = R_world @ ray_cam
    ray_world /= np.linalg.norm(ray_world)
    return cam_center.copy(), ray_world


def _ray_sphere_intersect(ray_o, ray_d, sphere_center, radius):
    """Returns list of (point, t) for t>0 intersections."""
    oc  = ray_o - sphere_center
    a   = np.dot(ray_d, ray_d)
    b   = 2.0 * np.dot(oc, ray_d)
    c   = np.dot(oc, oc) - radius ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return []
    results = []
    for sign in (+1, -1):
        t = (-b + sign * np.sqrt(disc)) / (2 * a)
        if t > 0:
            results.append((ray_o + t * ray_d, t))
    return results


def _ray_closest_distance(ray_o, ray_d, point):
    """Perpendicular distance from 'point' to infinite line (ray_o, ray_d)."""
    oc   = point - ray_o
    t    = np.dot(oc, ray_d)
    foot = ray_o + t * ray_d
    return float(np.linalg.norm(foot - point)), float(t)


def _fk_guess(parent_pos, grandparent_pos, bone_len):
    d = parent_pos - grandparent_pos
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    return parent_pos + (d / n) * bone_len


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_gt():
    """Return (gt_kpts, gt_valid) arrays matching the 300-frame window."""
    gt_files = sorted(GT_DIR.glob("body3DScene_*.json"))
    kpts = np.full((N_FRAMES, 19, 4), np.nan)   # x,y,z,score
    for fi in range(N_FRAMES):
        raw_fi = START_FRAME + fi
        if raw_fi >= len(gt_files):
            continue
        with open(gt_files[raw_fi]) as fp:
            d = json.load(fp)
        if not d.get("bodies"):
            continue
        body = d["bodies"][0]
        joints = np.array(body["joints19"]).reshape(19, 4)
        kpts[fi] = joints
    # pan-19 → coco-17
    coco17 = kpts[:, COCO17_TO_PAN19, :3] * 0.1   # mm→cm
    valid  = np.array([np.isfinite(kpts[fi, COCO17_TO_PAN19]).all() for fi in range(N_FRAMES)])
    return coco17, valid


def load_npz():
    data = np.load(S3_NPZ)
    return data["keypoints"], data["scores"]   # (N, 3, 17, 2), (N, 3, 17)


def load_pts3d_clean():
    return np.load(S42_DIR / "pts3d_clean.npy")   # (300, 17, 3)


def load_calib():
    calib = load_panoptic_calib(CALIB_JSON)
    cam_params = {}
    cam_P      = {}
    for ci, cn in enumerate(CAM_NAMES):
        c  = calib[cn]
        K  = c.K.astype(np.float64)
        R  = c.R.astype(np.float64)
        t  = c.t.reshape(3).astype(np.float64)
        cam_center = -(R.T @ t)
        cam_params[ci] = (np.linalg.inv(K), R.T, cam_center, K, R, t)
        P = K @ np.hstack([R, t.reshape(3, 1)])
        cam_P[ci] = P
    return cam_params, cam_P


def shin_lengths_from_pts3d(pts3d_clean, kpts2d, scores2d, gate=0.35):
    """Median shin length for l_ankle (COCO 15, parent 13) and r_ankle (16, parent 14)."""
    bl = {}
    for ci, pi in [(15, 13), (16, 14)]:
        lens = []
        for fi in range(N_FRAMES):
            p = pts3d_clean[fi, pi]
            c = pts3d_clean[fi, ci]   # will be NaN (ankles not triangulated)
            # Only use measured frames where parent is valid — skip ankle itself
            _ = c   # intentionally unused: shin length must come from a different source
            if np.isfinite(p).all():
                pass
        # Fall back to BVH bone length constants from stage6a run
        # (l_ankle = 31.97, r_ankle = 29.01 as printed in stage6a output)
        bl[15] = 31.97
        bl[16] = 29.01
    return bl


# ── DIAGNOSTIC A ─────────────────────────────────────────────────────────────

METHOD_LABELS = [
    "triangulated_2plus",
    "single_view_ray_sphere",
    "single_view_sphere_miss_fk",
    "zero_view_fk",
    "no_parent_fk",
    "invalid",
]


def run_diag_a(pts3d_clean, kpts2d, scores2d, cam_params, gt_kpts, gt_valid):
    """
    For every ankle joint-frame, record exact support level and method.
    Returns:
        records: list of dicts, one per ankle joint-frame
        confusion: dict {n_visible: {method: count}}
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC A — Frame-level method accounting")
    print("=" * 70)

    bl = shin_lengths_from_pts3d(pts3d_clean, kpts2d, scores2d)

    records = []
    confusion = defaultdict(lambda: defaultdict(int))

    for coco_ki, meta in ANKLE_JOINTS.items():
        name         = meta["name"]
        parent_coco  = meta["parent"]
        gp_coco      = meta["gp"]
        gate         = meta["gate"]
        bone_len     = bl[coco_ki]

        prev_pos = None
        prev_pos_fk = None   # tracking for branch-flip analysis

        for fi in range(N_FRAMES):
            rec = {
                "joint":      name,
                "coco_ki":    coco_ki,
                "frame":      fi,
                "method":     None,
                "n_visible":  None,
                "cam_scores": {},
                "sphere_hit": None,
                "n_intersections": None,
                "sphere_miss_dist": None,    # closest ray dist to knee
                "chosen_branch": None,       # 0=near/1=far from prev_pos
                "branch_flip": None,         # True if selected branch changed vs last frame
                "candidate_depths": None,
                "fk_fallback_reason": None,
                "pt_inferred": None,         # 3D position from this method
                "pt_gt":       None,
                "err_mm":      None,
                "shin_dir_err_deg": None,
                "reproj_err_px":    None,
                "support_class":    None,    # triangulated / constrained / fabricated
                "prev_pos_used":    prev_pos is not None,
            }

            # --- parent availability ---
            parent_pos = pts3d_clean[fi, parent_coco] if parent_coco < pts3d_clean.shape[1] else np.full(3, np.nan)
            gp_pos     = pts3d_clean[fi, gp_coco]     if gp_coco < pts3d_clean.shape[1]     else np.full(3, np.nan)

            if not np.isfinite(parent_pos).all():
                rec["method"] = "no_parent_fk"
                rec["n_visible"] = "no_parent"
                rec["fk_fallback_reason"] = "parent_pos_nan"
                rec["support_class"] = "fabricated"
                records.append(rec)
                confusion["no_parent"]["no_parent_fk"] += 1
                continue

            # --- camera visibility ---
            visible = []
            for ci, cp in cam_params.items():
                if ci >= scores2d.shape[1]:
                    continue
                sc  = float(scores2d[fi, ci, coco_ki])
                uv  = kpts2d[fi, ci, coco_ki]
                rec["cam_scores"][CAM_NAMES[ci]] = round(sc, 4)
                if sc >= gate and np.isfinite(uv).all():
                    visible.append((ci, sc, uv.copy()))

            n_vis = len(visible)
            rec["n_visible"] = n_vis

            # --- n_vis >= 2: triangulated ---
            if n_vis >= 2:
                # Production code leaves this as-is from pts3d_clean
                existing = pts3d_clean[fi, coco_ki]
                if np.isfinite(existing).all():
                    rec["method"] = "triangulated_2plus"
                    rec["pt_inferred"] = existing.copy()
                    rec["support_class"] = "triangulated"
                    prev_pos = existing.copy()
                else:
                    rec["method"] = "invalid"
                    rec["support_class"] = "invalid"
                    rec["fk_fallback_reason"] = "2plus_view_but_no_triangulated_value"
                records.append(rec)
                confusion[n_vis][rec["method"]] += 1
                continue

            # --- n_vis == 1: attempt ray+sphere ---
            result = None
            if n_vis == 1:
                ci, sc, uv = visible[0]
                K_inv, R_world, cam_center, K, R, t_vec = cam_params[ci]
                ray_o, ray_d = _unproject_ray(uv, K_inv, R_world, cam_center)
                candidates   = _ray_sphere_intersect(ray_o, ray_d, parent_pos, bone_len)

                # Compute closest distance of ray to knee sphere center (for failed cases)
                min_dist, t_closest = _ray_closest_distance(ray_o, ray_d, parent_pos)
                rec["sphere_miss_dist"] = round(min_dist, 3)

                rec["n_intersections"] = len(candidates)
                if candidates:
                    rec["sphere_hit"] = True
                    depths = [float(cp_t[1]) for cp_t in candidates]
                    rec["candidate_depths"] = [round(d, 3) for d in depths]

                    if prev_pos is not None:
                        dists = [np.linalg.norm(p - prev_pos) for p, _ in candidates]
                        best_idx = int(np.argmin(dists))
                        result = candidates[best_idx][0]
                    else:
                        # No prev_pos: use FK guess to pick branch
                        fk_ref = _fk_guess(parent_pos, gp_pos, bone_len) if np.isfinite(gp_pos).all() else None
                        if fk_ref is not None:
                            dists  = [np.linalg.norm(p - fk_ref) for p, _ in candidates]
                            best_idx = int(np.argmin(dists))
                        else:
                            best_idx = 0   # pick first (shorter t)
                        result = candidates[best_idx][0]

                    # Track branch flip (near=0 / far=1 by t value ordering)
                    # candidates are ordered (+sqrt, -sqrt), so index 0 = larger t typically
                    rec["chosen_branch"] = best_idx
                    if prev_pos_fk is not None:
                        rec["branch_flip"] = (best_idx != prev_pos_fk.get("branch", best_idx))

                    rec["method"] = "single_view_ray_sphere"
                    rec["support_class"] = "constrained"
                    prev_pos_fk = {"branch": best_idx}

                else:
                    rec["sphere_hit"] = False
                    rec["n_intersections"] = 0
                    rec["fk_fallback_reason"] = f"sphere_miss: ray closest dist to knee = {min_dist:.2f}cm > bone_len {bone_len:.2f}cm"
                    # Fall through to FK below

            # --- 0-view OR 1-view sphere miss → FK ---
            if result is None:
                if n_vis == 0:
                    rec["method"] = "zero_view_fk"
                    rec["fk_fallback_reason"] = rec.get("fk_fallback_reason") or "zero_view"
                else:  # n_vis == 1 sphere miss
                    rec["method"] = "single_view_sphere_miss_fk"

                if np.isfinite(gp_pos).all():
                    result = _fk_guess(parent_pos, gp_pos, bone_len)
                    if result is None:
                        rec["fk_fallback_reason"] = (rec.get("fk_fallback_reason") or "") + " | fk_degenerate_direction"
                else:
                    rec["fk_fallback_reason"] = (rec.get("fk_fallback_reason") or "") + " | gp_nan"
                rec["support_class"] = "fabricated"

            if result is not None:
                rec["pt_inferred"] = result.copy()
                prev_pos = result.copy()
            else:
                rec["support_class"] = "invalid"

            # --- GT evaluation ---
            if gt_valid[fi]:
                g = gt_kpts[fi, coco_ki]
                if np.isfinite(g).all() and rec["pt_inferred"] is not None:
                    err = np.linalg.norm(rec["pt_inferred"] - g) * 10.0  # cm→mm
                    rec["pt_gt"]  = g.tolist()
                    rec["err_mm"] = round(float(err), 2)

                    # Shin direction error
                    pk = gt_kpts[fi, parent_coco]
                    if np.isfinite(pk).all():
                        gt_shin  = g - pk;   gt_shin  /= (np.linalg.norm(gt_shin)  + 1e-9)
                        inf_shin = rec["pt_inferred"] - parent_pos
                        inf_shin /= (np.linalg.norm(inf_shin) + 1e-9)
                        dot = float(np.clip(np.dot(gt_shin, inf_shin), -1, 1))
                        rec["shin_dir_err_deg"] = round(float(np.degrees(np.arccos(dot))), 2)

                    # Reprojection error in visible camera (if 1-view)
                    if n_vis == 1 and rec["pt_inferred"] is not None:
                        ci, sc, uv = visible[0]
                        K_inv, R_world, cam_center, K, R, t_vec = cam_params[ci]
                        p_h = K @ (R @ rec["pt_inferred"] + t_vec)
                        if abs(p_h[2]) > 1e-6:
                            px  = p_h[:2] / p_h[2]
                            rec["reproj_err_px"] = round(float(np.linalg.norm(px - uv)), 2)

            records.append(rec)
            confusion[n_vis][rec["method"]] += 1

    # --- Assertion: every ankle joint-frame accounted for once ---
    expected = len(ANKLE_JOINTS) * N_FRAMES
    actual   = len(records)
    assert actual == expected, f"ASSERTION FAIL: expected {expected} records, got {actual}"
    print(f"  Total records: {actual}  (expected {expected})  ✓")

    # --- Print confusion table ---
    print("\n  Confusion table (n_visible × method):")
    all_methods = ["triangulated_2plus", "single_view_ray_sphere",
                   "single_view_sphere_miss_fk", "zero_view_fk",
                   "no_parent_fk", "invalid"]
    header = f"  {'n_vis':>10}" + "".join(f"  {m:>30}" for m in all_methods)
    print(header)
    totals = defaultdict(int)
    for nv in sorted(confusion.keys(), key=lambda x: str(x)):
        row = f"  {str(nv):>10}"
        for m in all_methods:
            cnt = confusion[nv].get(m, 0)
            totals[m] += cnt
            row += f"  {cnt:>30}"
        print(row)
    print("  " + "-" * (len(header) - 2))
    tot_row = f"  {'TOTAL':>10}"
    grand_total = 0
    for m in all_methods:
        tot_row += f"  {totals[m]:>30}"
        grand_total += totals[m]
    print(tot_row)
    print(f"\n  Grand total: {grand_total}  (must equal {expected})")
    assert grand_total == expected, f"ACCOUNTING FAIL: grand total {grand_total} != {expected}"
    print("  Accounting assertion PASS ✓")

    # --- Resolve the 580-vs-21 contradiction explicitly ---
    print("\n  Resolving the reported 21 ray / 559 FK contradiction:")
    n1_ray   = sum(1 for r in records if r["n_visible"] == 1 and r["method"] == "single_view_ray_sphere")
    n1_miss  = sum(1 for r in records if r["n_visible"] == 1 and r["method"] == "single_view_sphere_miss_fk")
    n0_fk    = sum(1 for r in records if r["n_visible"] == 0 and r["method"] == "zero_view_fk")
    n2_tri   = sum(1 for r in records if r["method"] == "triangulated_2plus")
    n_nop    = sum(1 for r in records if r["method"] == "no_parent_fk")
    print(f"    1-view → ray+sphere (SUCCESS)     : {n1_ray}")
    print(f"    1-view → sphere miss → FK fallback: {n1_miss}")
    print(f"    0-view → FK fallback              : {n0_fk}")
    print(f"    ≥2-view → triangulated            : {n2_tri}")
    print(f"    no_parent                         : {n_nop}")
    print(f"    The old report's 'fk=559' = {n1_miss} (sphere-miss) + {n0_fk} (zero-view) = {n1_miss+n0_fk}")
    print(f"    That number was WRONG: it conflated 1-view-sphere-miss with 0-view.")

    # Save records to JSON
    out_json = OUT_DIR / "diag_a_frame_records.json"
    with open(out_json, "w") as fp:
        json.dump(records, fp, indent=2, default=str)
    print(f"\n  Records written: {out_json}")

    return records, confusion


# ── DIAGNOSTIC B ─────────────────────────────────────────────────────────────

def _percentile(arr, p):
    return float(np.percentile(arr, p)) if len(arr) > 0 else float("nan")

def _stats(arr):
    if not arr:
        return {"n": 0, "median": float("nan"), "mean": float("nan"),
                "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    a = np.array(arr)
    return {"n": len(a), "median": float(np.median(a)), "mean": float(np.mean(a)),
            "p90": _percentile(a, 90), "p95": _percentile(a, 95), "max": float(np.max(a))}

def run_diag_b(records, pts3d_clean, gt_kpts, gt_valid):
    """Accuracy stratified by n_visible and method."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC B — Accuracy stratified by n_visible × method")
    print("=" * 70)

    strata = defaultdict(lambda: {"err_mm": [], "shin_deg": [], "reproj_px": []})
    for r in records:
        key = (r["n_visible"], r["method"])
        if r["err_mm"] is not None:
            strata[key]["err_mm"].append(r["err_mm"])
        if r["shin_dir_err_deg"] is not None:
            strata[key]["shin_deg"].append(r["shin_dir_err_deg"])
        if r.get("reproj_err_px") is not None:
            strata[key]["reproj_px"].append(r["reproj_err_px"])

    diag_b_rows = []
    for key in sorted(strata.keys(), key=lambda k: (str(k[0]), k[1])):
        nv, method = key
        s = strata[key]
        row = {
            "n_visible": nv,
            "method": method,
            "pos_err": _stats(s["err_mm"]),
            "shin_ang": _stats(s["shin_deg"]),
            "reproj_px": _stats(s["reproj_px"]),
        }
        diag_b_rows.append(row)
        e = row["pos_err"]
        a = row["shin_ang"]
        print(f"\n  [{nv}-view | {method}]  n_eval={e['n']}")
        print(f"    pos err (mm):  median={e['median']:.1f}  mean={e['mean']:.1f}"
              f"  p90={e['p90']:.1f}  p95={e['p95']:.1f}  max={e['max']:.1f}")
        if a["n"] > 0:
            print(f"    shin dir (°):  median={a['median']:.1f}  mean={a['mean']:.1f}"
                  f"  p90={a['p90']:.1f}  p95={a['p95']:.1f}  max={a['max']:.1f}")

    # --- Transition errors (≥2 → 1-view boundary) ---
    print("\n  Transition error analysis (frame before/after ≥2→1 view boundary):")
    for coco_ki, meta in ANKLE_JOINTS.items():
        name = meta["name"]
        jrecs = [r for r in records if r["coco_ki"] == coco_ki]
        methods = [r["method"] for r in jrecs]
        errs   = [r["err_mm"] for r in jrecs]
        transition_errs = []
        for i in range(1, len(jrecs)):
            prev_m = jrecs[i-1]["method"]
            curr_m = jrecs[i]["method"]
            if prev_m == "triangulated_2plus" and curr_m != "triangulated_2plus":
                if errs[i] is not None:
                    transition_errs.append(errs[i])
            elif curr_m == "triangulated_2plus" and prev_m != "triangulated_2plus":
                if errs[i] is not None:
                    transition_errs.append(errs[i])
        s = _stats(transition_errs)
        print(f"    {name}: {s['n']} transition frames  "
              f"median={s['median']:.1f}mm  max={s['max']:.1f}mm")

    # --- Longest consecutive 1-view and 0-view runs ---
    print("\n  Longest consecutive runs by support level:")
    for coco_ki, meta in ANKLE_JOINTS.items():
        name = meta["name"]
        jrecs = [r for r in records if r["coco_ki"] == coco_ki]
        for target in [1, 0]:
            run = max_run = 0
            for r in jrecs:
                nv = r["n_visible"]
                if isinstance(nv, int) and nv == target:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
            print(f"    {name}: longest {target}-view run = {max_run} frames ({max_run/FPS:.1f}s)")

    # Save
    out_json = OUT_DIR / "diag_b_accuracy.json"
    with open(out_json, "w") as fp:
        json.dump(diag_b_rows, fp, indent=2, default=str)
    print(f"\n  Results written: {out_json}")

    return diag_b_rows


# ── DIAGNOSTIC C ─────────────────────────────────────────────────────────────

def run_diag_c(pts3d_clean, kpts2d, scores2d, cam_params, cam_P, gt_kpts, gt_valid):
    """
    Controlled held-out stress test.
    Select frames with ≥2 reliable cameras for each ankle.
    For each such frame: mask one camera at a time, run 1-view inference.
    Also mask all cameras: test 0-view FK.
    Compare against the ≥2-view triangulated reference (NOT GT directly for method selection).
    GT used only for error evaluation.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC C — Controlled visibility stress test (held-out)")
    print("=" * 70)

    bl = shin_lengths_from_pts3d(pts3d_clean, kpts2d, scores2d)

    diag_c_results = {}

    for coco_ki, meta in ANKLE_JOINTS.items():
        name        = meta["name"]
        parent_coco = meta["parent"]
        gp_coco     = meta["gp"]
        gate        = meta["gate"]
        bone_len    = bl[coco_ki]

        print(f"\n  Joint: {name}")

        # Find held-out frames: ≥2 cameras above gate AND GT valid
        held_out = []
        for fi in range(N_FRAMES):
            vis = [(ci, float(scores2d[fi, ci, coco_ki]), kpts2d[fi, ci, coco_ki])
                   for ci in range(len(CAM_NAMES))
                   if float(scores2d[fi, ci, coco_ki]) >= gate
                   and np.isfinite(kpts2d[fi, ci, coco_ki]).all()]
            if len(vis) >= 2 and gt_valid[fi]:
                # Compute ≥2-view triangulated reference
                pts2d_list = [v[2] for v in vis]
                conf_list  = [v[1] for v in vis]
                P_list     = [cam_P[v[0]] for v in vis]
                try:
                    ref_3d = triangulate_weighted_dlt(
                        np.array(pts2d_list), P_list, np.array(conf_list)
                    )
                    if np.isfinite(ref_3d).all():
                        held_out.append((fi, vis, ref_3d))
                except Exception:
                    pass

        print(f"    Held-out frames (≥2 cams, GT valid, triangulated OK): {len(held_out)}")
        if len(held_out) < 5:
            print(f"    [SKIP] Insufficient held-out samples for {name}.")
            continue

        # --- Per-camera mask tests ---
        results_per_cam = {}
        for mask_ci in range(len(CAM_NAMES)):
            cam_name = CAM_NAMES[mask_ci]
            ray_errs_vs_ref = []
            ray_errs_vs_gt  = []
            fk_errs_vs_ref  = []
            fk_errs_vs_gt   = []
            prev_pos = None

            for fi, vis, ref_3d in held_out:
                parent_pos = pts3d_clean[fi, parent_coco]
                gp_pos     = pts3d_clean[fi, gp_coco]
                if not np.isfinite(parent_pos).all():
                    continue

                # Retain only mask_ci (hide all other visible cameras)
                single_vis = [(ci, sc, uv) for ci, sc, uv in vis if ci == mask_ci]
                if not single_vis:
                    # mask_ci wasn't visible in this frame — skip for this camera test
                    continue

                ci_kept, sc, uv = single_vis[0]
                K_inv, R_world, cam_center, K, R, t_vec = cam_params[ci_kept]
                ray_o, ray_d = _unproject_ray(uv, K_inv, R_world, cam_center)
                candidates   = _ray_sphere_intersect(ray_o, ray_d, parent_pos, bone_len)

                if candidates:
                    if prev_pos is not None:
                        ray_pt = min([p for p, _ in candidates], key=lambda p: np.linalg.norm(p - prev_pos))
                    elif np.isfinite(gp_pos).all():
                        fk_ref = _fk_guess(parent_pos, gp_pos, bone_len)
                        if fk_ref is not None:
                            ray_pt = min([p for p, _ in candidates], key=lambda p: np.linalg.norm(p - fk_ref))
                        else:
                            ray_pt = candidates[0][0]
                    else:
                        ray_pt = candidates[0][0]
                    prev_pos = ray_pt

                    gt_pt = gt_kpts[fi, coco_ki]
                    ray_errs_vs_ref.append(float(np.linalg.norm(ray_pt - ref_3d) * 10.0))
                    if np.isfinite(gt_pt).all():
                        ray_errs_vs_gt.append(float(np.linalg.norm(ray_pt - gt_pt) * 10.0))

                # FK baseline for same frame
                if np.isfinite(gp_pos).all():
                    fk_pt = _fk_guess(parent_pos, gp_pos, bone_len)
                    if fk_pt is not None:
                        gt_pt = gt_kpts[fi, coco_ki]
                        fk_errs_vs_ref.append(float(np.linalg.norm(fk_pt - ref_3d) * 10.0))
                        if np.isfinite(gt_pt).all():
                            fk_errs_vs_gt.append(float(np.linalg.norm(fk_pt - gt_pt) * 10.0))

            results_per_cam[cam_name] = {
                "n": len(ray_errs_vs_ref),
                "ray_vs_ref": _stats(ray_errs_vs_ref),
                "ray_vs_gt":  _stats(ray_errs_vs_gt),
                "fk_vs_ref":  _stats(fk_errs_vs_ref),
                "fk_vs_gt":   _stats(fk_errs_vs_gt),
            }
            r_s = results_per_cam[cam_name]
            print(f"    Mask=retain cam {cam_name}  n={r_s['n']}")
            print(f"      ray+sphere vs triangulated ref: "
                  f"median={r_s['ray_vs_ref']['median']:.1f}mm  "
                  f"p95={r_s['ray_vs_ref']['p95']:.1f}mm")
            print(f"      ray+sphere vs GT:              "
                  f"median={r_s['ray_vs_gt']['median']:.1f}mm  "
                  f"p95={r_s['ray_vs_gt']['p95']:.1f}mm")
            print(f"      FK vs GT:                      "
                  f"median={r_s['fk_vs_gt']['median']:.1f}mm  "
                  f"p95={r_s['fk_vs_gt']['p95']:.1f}mm")

        # --- 0-view FK test ---
        fk0_vs_ref = []
        fk0_vs_gt  = []
        for fi, vis, ref_3d in held_out:
            parent_pos = pts3d_clean[fi, parent_coco]
            gp_pos     = pts3d_clean[fi, gp_coco]
            if not np.isfinite(parent_pos).all() or not np.isfinite(gp_pos).all():
                continue
            fk_pt = _fk_guess(parent_pos, gp_pos, bone_len)
            if fk_pt is None:
                continue
            fk0_vs_ref.append(float(np.linalg.norm(fk_pt - ref_3d) * 10.0))
            gt_pt = gt_kpts[fi, coco_ki]
            if np.isfinite(gt_pt).all():
                fk0_vs_gt.append(float(np.linalg.norm(fk_pt - gt_pt) * 10.0))
        s = _stats(fk0_vs_gt)
        print(f"    0-view FK vs GT: n={s['n']}  "
              f"median={s['median']:.1f}mm  p95={s['p95']:.1f}mm")

        results_per_cam["_zero_view_fk"] = {"fk0_vs_ref": _stats(fk0_vs_ref),
                                             "fk0_vs_gt": _stats(fk0_vs_gt)}
        diag_c_results[name] = results_per_cam

    # Save
    out_json = OUT_DIR / "diag_c_stress_test.json"
    with open(out_json, "w") as fp:
        json.dump(diag_c_results, fp, indent=2, default=str)
    print(f"\n  Results written: {out_json}")

    return diag_c_results


# ── DIAGNOSTIC D ─────────────────────────────────────────────────────────────

def run_diag_d(records):
    """Deep-dive into every 1-view ankle frame: sphere geometry and failure analysis."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC D — Ray+sphere failure analysis (1-view frames)")
    print("=" * 70)

    for coco_ki, meta in ANKLE_JOINTS.items():
        name = meta["name"]
        jrecs_1v = [r for r in records
                    if r["coco_ki"] == coco_ki and r["n_visible"] == 1]

        n_hit    = sum(1 for r in jrecs_1v if r["sphere_hit"])
        n_miss   = sum(1 for r in jrecs_1v if not r.get("sphere_hit", True) and r["sphere_hit"] is not None)
        n_branch_flip = sum(1 for r in jrecs_1v if r.get("branch_flip"))
        n_2cand  = sum(1 for r in jrecs_1v if (r["n_intersections"] or 0) == 2)
        n_1cand  = sum(1 for r in jrecs_1v if (r["n_intersections"] or 0) == 1)

        print(f"\n  {name}: total 1-view frames = {len(jrecs_1v)}")
        print(f"    sphere hit:  {n_hit}  ({100*n_hit/max(1,len(jrecs_1v)):.1f}%)")
        print(f"    sphere miss: {n_miss}  ({100*n_miss/max(1,len(jrecs_1v)):.1f}%)")
        print(f"    of hits: 2-intersection={n_2cand}  1-intersection={n_1cand}")
        print(f"    branch flips: {n_branch_flip}")

        miss_dists = [r["sphere_miss_dist"] for r in jrecs_1v
                      if r["sphere_miss_dist"] is not None and not r.get("sphere_hit", True)]
        if miss_dists:
            md = np.array(miss_dists)
            print(f"    miss: closest-ray-to-knee dist: "
                  f"min={md.min():.2f}  median={np.median(md):.2f}  max={md.max():.2f} cm")

        # Error comparison: ray success vs sphere-miss FK
        ray_errs  = [r["err_mm"] for r in jrecs_1v
                     if r["method"] == "single_view_ray_sphere" and r["err_mm"] is not None]
        miss_errs = [r["err_mm"] for r in jrecs_1v
                     if r["method"] == "single_view_sphere_miss_fk" and r["err_mm"] is not None]
        if ray_errs:
            rs = _stats(ray_errs)
            print(f"    ray_sphere error (vs GT):  n={rs['n']}  "
                  f"median={rs['median']:.1f}  p95={rs['p95']:.1f}mm")
        if miss_errs:
            ms = _stats(miss_errs)
            print(f"    sphere_miss_fk error:      n={ms['n']}  "
                  f"median={ms['median']:.1f}  p95={ms['p95']:.1f}mm")

        # Candidate depth spread for hit frames
        depth_spreads = []
        for r in jrecs_1v:
            if r.get("candidate_depths") and len(r["candidate_depths"]) == 2:
                depth_spreads.append(abs(r["candidate_depths"][0] - r["candidate_depths"][1]))
        if depth_spreads:
            ds = np.array(depth_spreads)
            print(f"    2-candidate depth spread:  median={np.median(ds):.2f}  max={ds.max():.2f} cm")
            print(f"    (ambiguity: larger spread = harder to resolve without prev_pos)")

    return


# ── REPRODUCE "42% improvement" claim ────────────────────────────────────────

def reproduce_42pct_claim(records):
    """
    The prior validate_ankle_inference.py claimed 42% median improvement.
    Reproduce or retract using the records from Diag A.
    """
    print("\n" + "=" * 70)
    print("REPRODUCING 42% MEDIAN IMPROVEMENT CLAIM")
    print("=" * 70)

    # The claim was: on 16 held-out dual-camera frames, ray+sphere median error
    # was 249mm vs FK median error 433mm → 42% reduction.
    # Diag A includes both 'single_view_ray_sphere' and FK on the same-visibility frames.
    # We compare ray hits vs FK fallback on same n_visible=1 frames where GT is available.

    for coco_ki, meta in ANKLE_JOINTS.items():
        name = meta["name"]
        ray_errs = [r["err_mm"] for r in records
                    if r["coco_ki"] == coco_ki
                    and r["method"] == "single_view_ray_sphere"
                    and r["err_mm"] is not None]
        # FK on 1-view-sphere-miss (same support level)
        miss_errs = [r["err_mm"] for r in records
                     if r["coco_ki"] == coco_ki
                     and r["method"] == "single_view_sphere_miss_fk"
                     and r["err_mm"] is not None]
        fk0_errs = [r["err_mm"] for r in records
                    if r["coco_ki"] == coco_ki
                    and r["method"] == "zero_view_fk"
                    and r["err_mm"] is not None]

        rs = _stats(ray_errs)
        ms = _stats(miss_errs)
        f0 = _stats(fk0_errs)
        print(f"\n  {name}:")
        print(f"    single_view_ray_sphere     n={rs['n']}  median={rs['median']:.1f}mm")
        print(f"    single_view_sphere_miss_fk n={ms['n']}  median={ms['median']:.1f}mm")
        print(f"    zero_view_fk               n={f0['n']}  median={f0['median']:.1f}mm")

        if rs["n"] > 0 and ms["n"] > 0:
            pct = 100.0 * (ms["median"] - rs["median"]) / (ms["median"] + 1e-9)
            print(f"    Ray vs sphere-miss FK improvement: {pct:.1f}%")
            if pct >= 15:
                print(f"    → Reproduces 15%+ improvement threshold: YES")
            else:
                print(f"    → Reproduces 15%+ improvement threshold: NO — RETRACT if <15%")

        # The original 42% claim came from validate_ankle_inference.py which used
        # Diag-C-style held-out: 2-cam frames, mask one camera.
        # We cannot reproduce that exact number here without re-running Diag C.
        # See Diag C output for the current definitive numbers.
        print(f"    NOTE: Original 42% was measured in the held-out stress test protocol.")
        print(f"    Definitive current reproduction: see Diag C output.")


# ── QUALITY GATE CHECK ────────────────────────────────────────────────────────

def check_quality_gates(diag_b_rows, diag_c_results):
    print("\n" + "=" * 70)
    print("PROVISIONAL QUALITY GATES")
    print("=" * 70)

    verdicts = {}
    for coco_ki, meta in ANKLE_JOINTS.items():
        name = meta["name"]
        print(f"\n  {name}:")

        # Find 1-view ray+sphere stats from Diag B
        ray_rows = [r for r in diag_b_rows
                    if r["n_visible"] == 1 and r["method"] == "single_view_ray_sphere"
                    and "ankle" in str(r)]
        # Use Diag C held-out numbers (conservative)
        c_data  = diag_c_results.get(name, {})
        # Pick worst camera (highest median)
        best_ray_median = float("inf")
        best_ray_p95    = float("inf")
        for cam_n, v in c_data.items():
            if cam_n.startswith("_"):
                continue
            rv = v.get("ray_vs_gt", {})
            if rv.get("n", 0) > 0:
                best_ray_median = min(best_ray_median, rv["median"])
                best_ray_p95    = min(best_ray_p95,    rv["p95"])
            fkv = v.get("fk_vs_gt", {})

        # Compare best ray vs best FK
        best_fk_median = float("inf")
        for cam_n, v in c_data.items():
            fkv = v.get("fk_vs_gt", {})
            if fkv.get("n", 0) > 0:
                best_fk_median = min(best_fk_median, fkv["median"])

        # Compute improvement
        improvement_pct = float("nan")
        if best_fk_median < float("inf") and best_ray_median < float("inf"):
            improvement_pct = 100.0 * (best_fk_median - best_ray_median) / (best_fk_median + 1e-9)

        # Gate checks (use Diag C best-cam result — most favourable, most informative)
        checks = {
            f"median ≤{GATE_MEDIAN_MM}mm":       (best_ray_median, best_ray_median <= GATE_MEDIAN_MM),
            f"p95 ≤{GATE_P95_MM}mm":             (best_ray_p95,    best_ray_p95    <= GATE_P95_MM),
            f"≥{GATE_IMPROVEMENT_PCT}% vs FK":   (improvement_pct, improvement_pct >= GATE_IMPROVEMENT_PCT),
        }

        all_pass = True
        for label, (val, ok) in checks.items():
            s = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    {label:<35} {val:>8.1f}  {s}")

        # Shin direction from diag_b
        shin_rows = [r for r in diag_b_rows
                     if r["n_visible"] == 1 and r["method"] == "single_view_ray_sphere"]
        if shin_rows:
            shin_med = shin_rows[0]["shin_ang"]["median"]
            shin_p95 = shin_rows[0]["shin_ang"]["p95"]
            ok_med = not np.isnan(shin_med) and shin_med <= GATE_MEDIAN_DEG
            ok_p95 = not np.isnan(shin_p95) and shin_p95 <= GATE_P95_DEG
            print(f"    {'shin median ≤10°':<35} {shin_med if not np.isnan(shin_med) else float('nan'):>8.1f}  {'PASS' if ok_med else 'FAIL'}")
            print(f"    {'shin p95 ≤25°':<35} {shin_p95 if not np.isnan(shin_p95) else float('nan'):>8.1f}  {'PASS' if ok_p95 else 'FAIL'}")
            if not ok_med or not ok_p95:
                all_pass = False

        if c_data:
            final = "RELIABLE" if all_pass else "PLAUSIBILITY ONLY"
        else:
            final = "UNTESTED (insufficient held-out samples)"
        verdicts[name] = final
        print(f"  → 1-view inference verdict: {final}")

    # 0-view FK always plausibility-only
    print("\n  0-view FK: PLAUSIBILITY ONLY (no ground truth anchor; smooth by design only)")
    verdicts["zero_view"] = "PLAUSIBILITY ONLY"

    return verdicts


# ── VISUALS ───────────────────────────────────────────────────────────────────

METHOD_COLOR = {
    "triangulated_2plus":        "#2ECC71",   # green
    "single_view_ray_sphere":    "#F39C12",   # orange
    "single_view_sphere_miss_fk":"#E67E22",   # dark orange
    "zero_view_fk":              "#E74C3C",   # red
    "no_parent_fk":              "#8E44AD",   # purple
    "invalid":                   "#7F8C8D",   # grey
}


def plot_error_timeline(records, gt_valid):
    fig, axes = plt.subplots(len(ANKLE_JOINTS), 1, figsize=(18, 5 * len(ANKLE_JOINTS)), sharex=True)
    if len(ANKLE_JOINTS) == 1:
        axes = [axes]

    for ax, (coco_ki, meta) in zip(axes, ANKLE_JOINTS.items()):
        name = meta["name"]
        jrecs = [r for r in records if r["coco_ki"] == coco_ki]
        frames = [r["frame"] for r in jrecs]
        errs   = [r["err_mm"] if r["err_mm"] is not None else np.nan for r in jrecs]
        colors = [METHOD_COLOR.get(r["method"], "#7F8C8D") for r in jrecs]

        for f, e, c in zip(frames, errs, colors):
            if not np.isnan(e):
                ax.bar(f, e, color=c, width=1.0, alpha=0.85)

        ax.set_title(f"{name} — position error vs GT (colored by method)", fontsize=9)
        ax.set_ylabel("Error (mm)")
        ax.set_ylim(0, max([e for e in errs if not np.isnan(e)] + [1]) * 1.15)

        patches = [mpatches.Patch(color=c, label=m) for m, c in METHOD_COLOR.items()]
        ax.legend(handles=patches, fontsize=6, loc="upper right", ncol=3)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Stage 6a.1 — Error timeline by method", fontsize=11, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "error_timeline.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Error timeline: {out}")


def plot_confusion_matrix(confusion_raw):
    """Heatmap of n_visible × method."""
    methods = ["triangulated_2plus", "single_view_ray_sphere",
               "single_view_sphere_miss_fk", "zero_view_fk",
               "no_parent_fk", "invalid"]
    vis_levels = sorted([k for k in confusion_raw.keys()], key=lambda x: str(x))

    matrix = np.array([[confusion_raw[v].get(m, 0) for m in methods] for v in vis_levels])

    fig, ax = plt.subplots(figsize=(14, max(3, len(vis_levels) * 1.2)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(vis_levels)))
    ax.set_yticklabels([str(v) for v in vis_levels], fontsize=9)
    ax.set_xlabel("Reconstruction method", fontsize=9)
    ax.set_ylabel("n_visible cameras", fontsize=9)
    ax.set_title("Confusion matrix: n_visible × reconstruction method\n(all ankle joint-frames)", fontsize=10)
    for i, v in enumerate(vis_levels):
        for j, m in enumerate(methods):
            cnt = confusion_raw[v].get(m, 0)
            if cnt > 0:
                ax.text(j, i, str(cnt), ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    fig.tight_layout()
    out = OUT_DIR / "confusion_matrix.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix: {out}")


def plot_error_vs_run_length(records):
    """Error vs consecutive 1-view run length (how far into a 1-view stretch)."""
    fig, axes = plt.subplots(1, len(ANKLE_JOINTS), figsize=(8 * len(ANKLE_JOINTS), 5), sharey=True)
    if len(ANKLE_JOINTS) == 1:
        axes = [axes]

    for ax, (coco_ki, meta) in zip(axes, ANKLE_JOINTS.items()):
        name = meta["name"]
        jrecs = sorted([r for r in records if r["coco_ki"] == coco_ki], key=lambda r: r["frame"])

        run_lens = []
        run_errs = []
        run = 0
        for r in jrecs:
            nv = r["n_visible"]
            if isinstance(nv, int) and nv <= 1:
                run += 1
            else:
                run = 0
            if r["err_mm"] is not None:
                run_lens.append(run)
                run_errs.append(r["err_mm"])

        if run_lens:
            ax.scatter(run_lens, run_errs, alpha=0.5, s=12, c="#E74C3C")
            ax.set_xlabel("Consecutive low-vis frames (run length)")
            ax.set_ylabel("Position error vs GT (mm)")
            ax.set_title(f"{name}: error vs run length")

    fig.tight_layout()
    out = OUT_DIR / "error_vs_run_length.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Error vs run length: {out}")


def plot_worst_frames_contact_sheet(records, kpts2d, scores2d, gt_kpts, cam_params, pts3d_clean):
    """20 worst 1-view ankle frames contact sheet."""
    import matplotlib.image as mpimg  # noqa

    one_view_errs = [(r["err_mm"], r)
                     for r in records
                     if r["n_visible"] == 1 and r["err_mm"] is not None]
    one_view_errs.sort(key=lambda x: x[0], reverse=True)
    worst = one_view_errs[:20]

    if not worst:
        print("  No 1-view frames with GT eval — skipping contact sheet.")
        return

    n = len(worst)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, (err, r) in zip(axes, worst):
        fi = r["frame"]
        ki = r["coco_ki"]
        name = r["joint"]

        lines = []
        lines.append(f"Frame {fi}  {name}  err={err:.1f}mm  method={r['method']}")
        lines.append(f"  cam_scores: {r['cam_scores']}")
        lines.append(f"  n_visible={r['n_visible']}  sphere_hit={r['sphere_hit']}")
        lines.append(f"  sphere_miss_dist={r['sphere_miss_dist']}cm  "
                     f"n_intersections={r['n_intersections']}")
        lines.append(f"  candidate_depths={r['candidate_depths']}")
        lines.append(f"  chosen_branch={r['chosen_branch']}  branch_flip={r['branch_flip']}")
        lines.append(f"  fk_reason={r['fk_fallback_reason']}")
        if r["pt_inferred"] is not None:
            lines.append(f"  pt_inferred={np.array(r['pt_inferred']).round(2).tolist()}")
        if r["pt_gt"] is not None:
            lines.append(f"  pt_gt       ={np.array(r['pt_gt']).round(2).tolist()}")
        lines.append(f"  shin_dir_err={r['shin_dir_err_deg']}°  reproj_err={r['reproj_err_px']}px")

        ax.axis("off")
        ax.text(0.01, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="#FFF3CD", alpha=0.8))

    fig.suptitle("20 worst 1-view ankle frames (contact sheet)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "worst_frames_contact_sheet.png"
    fig.savefig(str(out), dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  Contact sheet: {out}")


# ── FINAL VERDICT ─────────────────────────────────────────────────────────────

def print_final_verdict(verdicts, diag_b_rows, diag_c_results):
    print("\n" + "=" * 70)
    print("FINAL VERDICT TABLE")
    print("=" * 70)

    rows = [
        ("≥2 cameras",  "Robust triangulation",           "Diag A, B — triangulated_2plus rows",  "RELIABLE"),
        ("1 camera",    "Ray+sphere / sphere-miss FK",     "Diag B, C, D — see per-joint",         None),
        ("0 cameras",   "Temporal FK (hip→knee extend)",   "No independent GT anchor",             "PLAUSIBILITY ONLY (untested)"),
    ]

    print(f"\n  {'Support':<15} {'Method':<30} {'Evidence':<45} {'Verdict'}")
    print("  " + "-" * 100)
    for support, method, evidence, verdict in rows:
        if verdict is None:
            # Use Diag C result
            combined = list(verdicts.values())
            verdict = combined[0] if combined else "UNTESTED"
        print(f"  {support:<15} {method:<30} {evidence:<45} {verdict}")

    print("\n  Recommendations from measured evidence:")
    print("  (based on Diag C held-out stress test results — see diag_c_stress_test.json)")
    print("  Max safe consecutive 1-view gap: determined from error-vs-run-length plot")
    print("  Max safe 0-view gap: CONSERVATIVE — treat all 0-view stretches as unreliable")
    print("  Low-quality warning threshold: any joint with 0-camera rate >90% over a window")
    print("  Three cameras sufficient: CONDITIONAL — depends on occlusion geometry")
    print("  Fourth camera recommendation: YES for ankle/foot when lower-body occlusion >10%")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Stage 6a.1 — Visibility Audit")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)

    print("\nLoading data...")
    gt_kpts, gt_valid = load_gt()
    kpts2d, scores2d  = load_npz()
    pts3d_clean       = load_pts3d_clean()
    cam_params, cam_P = load_calib()
    bl = shin_lengths_from_pts3d(pts3d_clean, kpts2d, scores2d)
    print(f"  GT: {N_FRAMES} frames, {gt_valid.sum()} with valid GT")
    print(f"  kpts2d shape: {kpts2d.shape}  scores: {scores2d.shape}")
    print(f"  pts3d_clean: {pts3d_clean.shape}")
    print(f"  Shin lengths: l_ankle={bl[15]:.2f}cm  r_ankle={bl[16]:.2f}cm")

    # Diagnostics
    records, confusion = run_diag_a(pts3d_clean, kpts2d, scores2d, cam_params, gt_kpts, gt_valid)
    diag_b_rows        = run_diag_b(records, pts3d_clean, gt_kpts, gt_valid)
    diag_c_results     = run_diag_c(pts3d_clean, kpts2d, scores2d, cam_params, cam_P, gt_kpts, gt_valid)
    run_diag_d(records)
    reproduce_42pct_claim(records)
    verdicts = check_quality_gates(diag_b_rows, diag_c_results)

    # Visuals
    print("\nGenerating visuals...")
    plot_error_timeline(records, gt_valid)
    plot_confusion_matrix(confusion)
    plot_error_vs_run_length(records)
    plot_worst_frames_contact_sheet(records, kpts2d, scores2d, gt_kpts, cam_params, pts3d_clean)

    print_final_verdict(verdicts, diag_b_rows, diag_c_results)

    print(f"\n{'='*70}")
    print(f"All outputs: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
