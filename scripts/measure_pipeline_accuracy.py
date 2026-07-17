# scripts/measure_pipeline_accuracy.py
"""Canonical pipeline accuracy measurement.

Single reproducible source of truth for MPJPE across pipeline stages,
replacing the non-reproducible 31.8 mm (audit_report.md) and the mislabeled
viz numbers. All inputs use the canonical detector observations
(canonical_detector_pose_observations.npz), cameras 00_11/00_12/00_23,
GT hdPose3d_stage1_coco19, OFFSET=150.

Conventions: pred arrays are cm internal (Y-up, Z-back); gt_mm is mm internal.
Errors are reported in mm. BODY_J = range(5,17) (12 body joints), matching
audit_23 / diagnosis.md so numbers are directly comparable.
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(r"E:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

OFFSET = 150
BODY_J = list(range(5, 17))          # 12 body joints (shoulders..ankles)
PELVIS_J = [11, 12]
TORSO_J = [5, 6, 11, 12]


def _finite_mask(pts, joints):
    return np.all(np.isfinite(pts[:, joints, :]), axis=-1)   # (F, len(joints))


def coverage(pred, joints):
    """Fraction of finite joint-frames across all frames for the given joints."""
    m = _finite_mask(pred, joints)
    return {"finite_fraction": float(m.mean()),
            "n": int(m.sum()), "n_possible": int(m.size)}


def _align(pred, gt_mm, mode, joints):
    pred_a = pred.copy().astype(np.float64)
    gt_a = (gt_mm.copy().astype(np.float64) / 10.0)          # mm -> cm
    if mode == "none":
        return pred_a, gt_a
    if mode == "pelvis":
        anch = PELVIS_J
    elif mode == "torso":
        anch = TORSO_J
    else:
        raise ValueError(f"unknown alignment {mode}")
    for f in range(len(pred_a)):
        m = _finite_mask(pred[f:f+1], anch)[0] & _finite_mask(gt_mm[f:f+1], anch)[0]
        if m.sum() >= 1:
            shift = (np.nanmean(gt_a[f, anch], axis=0)
                     - np.nanmean(pred_a[f, anch], axis=0))
            pred_a[f] = pred_a[f] + shift
    return pred_a, gt_a


def mpjpe(pred, gt_mm, joints=BODY_J, alignment="none"):
    """MPJPE in mm over given joints. pred in cm internal, gt_mm in mm internal.

    Returns {mean_mm, median_mm, p95_mm, max_mm, n, excluded_fraction}.
    """
    pred_a, gt_a = _align(pred, gt_mm, alignment, joints)
    pj = pred_a[:, joints, :]
    gj = gt_a[:, joints, :]
    finite = np.all(np.isfinite(pj), axis=-1) & np.all(np.isfinite(gj), axis=-1)
    err_cm = np.linalg.norm(pj - gj, axis=-1)            # cm
    err_mm = err_cm[finite] * 10.0                       # cm -> mm
    n_possible = int(finite.size)
    n = int(finite.sum())
    if n == 0:
        return {"mean_mm": float("nan"), "median_mm": float("nan"),
                "p95_mm": float("nan"), "max_mm": float("nan"),
                "n": 0, "excluded_fraction": 1.0}
    return {"mean_mm": float(err_mm.mean()),
            "median_mm": float(np.median(err_mm)),
            "p95_mm": float(np.percentile(err_mm, 95)),
            "max_mm": float(err_mm.max()),
            "n": n, "excluded_fraction": 1.0 - n / n_possible}


def mpjpe_per_frame(pred, gt_mm, joints=BODY_J, alignment="none"):
    """Per-frame mean MPJPE (mm). Returns (F,) array — one mean error per frame.

    Unlike mpjpe which flattens to a scalar, this keeps the frame axis so you
    can see WHICH frames spike. Joints with NaN pred or NaN gt are excluded
    per-frame (the mean is over finite joints only). Frames with zero finite
    joints get 0.0 (not NaN) so plots don't break.
    """
    pred_a, gt_a = _align(pred, gt_mm, alignment, joints)
    pj = pred_a[:, joints, :]          # (F, J, 3) cm
    gj = gt_a[:, joints, :]            # _align already converts mm -> cm
    err_cm = np.linalg.norm(pj - gj, axis=-1)    # (F, J)
    finite = np.isfinite(err_cm)
    err_sum = np.where(finite, err_cm, 0.0).sum(axis=1)
    n_per_frame = finite.sum(axis=1)
    err_mean_cm = np.where(n_per_frame > 0, err_sum / n_per_frame, 0.0)
    return err_mean_cm * 10.0     # cm -> mm, (F,) array


# --- Canonical data loader + raw triangulation baseline (Task 2) ---
import json
from aimocap.data.panoptic import load_calibration
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

SEQ = "171204_pose1"
CAMS = ["00_11", "00_12", "00_23"]
NFRAMES = 1800
FPS = 30.0
CONF_GATE = 0.4
REPROJ_THR = 100.0
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_canonical_data():
    """Load canonical 2D detections, calibration, GT (mm internal), bone lengths."""
    npz_path = ROOT / "outputs/canonical_dataset/canonical_detector_pose_observations.npz"
    npz = np.load(npz_path)
    kpts = npz["kpts"].astype(np.float32)
    scores = npz["scores"].astype(np.float32)
    calib = load_calibration(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    K = [calib[cn].K.astype(np.float64) for cn in CAMS]
    extr = [(calib[cn].R.astype(np.float64), calib[cn].t.astype(np.float64).reshape(3, 1))
            for cn in CAMS]
    bl = np.array(json.loads(
        (ROOT / "outputs/phase_b_audit/audit_config.json").read_text())["bone_lengths"])

    gt_raw = np.full((NFRAMES, 17, 3), np.nan, np.float64)
    for i in range(NFRAMES):
        fn = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19/body3DScene_{OFFSET+i:08d}.json"
        if not fn.exists():
            continue
        js = json.loads(fn.read_text())
        if not js.get("bodies"):
            continue
        j19 = np.array(js["bodies"][0]["joints19"], np.float64).reshape(19, 4)
        for c17, p19 in enumerate(COCO17_TO_PAN19):
            if j19[p19, 3] >= 0:           # mask untracked markers (conf=-1 → NaN)
                gt_raw[i, c17] = j19[p19, :3]
    gt_mm = gt_raw.copy()
    gt_mm[:, :, 1] *= -1          # Panoptic Y-down -> internal Y-up
    gt_mm[:, :, 2] *= -1          # Z-forward -> Z-back
    gt_mm *= 10.0                 # cm -> mm
    return {"kpts": kpts, "scores": scores, "calib": calib, "K": K, "extr": extr,
            "gt_mm": gt_mm, "bl": bl,
            "provenance": {"npz_sha256": _sha(npz_path), "cams": CAMS,
                           "offset": OFFSET, "nframes": NFRAMES}}


# The retarget only uses body (0-16) + feet (17-22) = 23 joints. Triangulating
# all 133 is ~6x slower and the extra joints (face/hands) are discarded. With
# min_aspect_ratio=0.0 the engine's camera selection is joint-count-independent,
# so body joints 0-16 are bit-identical whether we pass 17, 23, or 133 joints.
_TRIANGULATE_KEEP_KPTS = 23


def triangulate_raw(d):
    """Raw triangulation (T1 raw pixels), returned in cm internal (Y-up, Z-back).

    Params match scratch/audit_23_canonical_baseline.py (the 35.8 mm T1_raw
    anchor): min_conf=0.4, reproj=100, min_aspect_ratio=0.0, f_scale=10.0.
    NOTE: the plan specified f_scale=0.0, but scipy's Huber least_squares
    divides by f_scale**2, so 0.0 raises ValueError (divide-by-zero -> NaN).

    Slices detections to the first 23 joints (body+feet) so the IK gets
    measured foot keypoints (big toe, COCO 17/20) for foot orientation.
    """
    k = _TRIANGULATE_KEEP_KPTS
    kpts = d["kpts"][:, :, :k]
    scores = d["scores"][:, :, :k]
    tri = triangulate_sequence_with_diagnostics(
        kpts, scores, d["K"], d["extr"],
        min_conf=CONF_GATE, reproj_threshold_px=REPROJ_THR,
        min_aspect_ratio=0.0, f_scale=10.0)
    return tri.points3d.copy()    # already internal Y-up, cm (Panoptic t is cm)


# --- Stage-6 waterfall ablation (Task 3) ---
from scripts.experiment_gating_architectures import (
    build_bvh_positions, fit_skeleton_sequence, fit_skeleton_sequence_preserve_finite, infer_by_ray_sphere)
from aimocap.math.filter import fill_gaps_with_logging, filter_skeleton_one_euro

BVH2C = {1: 11, 2: 13, 3: 15, 4: 12, 5: 14, 6: 16,
         8: 5, 9: 7, 10: 9, 11: 6, 12: 8, 13: 10, 14: 0}
ANKLE_J = [15, 16]


def _bvh17_from_fk(fk):
    """Map 15-joint BVH FK positions back to 17-joint COCO order (cm internal)."""
    out = np.full((fk.shape[0], 17, 3), np.nan, np.float32)
    for bj, cj in BVH2C.items():
        out[:, cj] = fk[:, bj]
    out[:, 0] = (out[:, 11] + out[:, 12]) / 2.0   # head placeholder (not in BODY_J)
    return out


def stage6_waterfall(d):
    """Reproduce the gate1 candidate-B Stage-6 path step by step.

    Returns {stage: {"all": mpjpe, "raw_finite_only": mpjpe, "coverage": float}}.
    'all'             = MPJPE over joints finite at THIS stage.
    'raw_finite_only' = MPJPE over the SAME joint-frames that were finite in RAW,
                       so filling hard joints does not inflate the number --
                       this isolates true degradation of already-good joints.
    """
    raw = triangulate_raw(d)
    raw_finite = np.all(np.isfinite(raw), axis=-1)            # (F,17)

    def measure(arr):
        return {
            "all": mpjpe(arr, d["gt_mm"], BODY_J, "none"),
            "raw_finite_only": mpjpe(
                np.where(raw_finite[:, :, None], arr, np.nan), d["gt_mm"], BODY_J, "none"),
            "coverage": coverage(arr, BODY_J)["finite_fraction"],
        }

    out = {"raw": measure(raw)}

    filled, _, _ = fill_gaps_with_logging(raw, [str(i) for i in range(17)], fps=FPS)
    out["gap_filled"] = measure(filled)

    smooth = filter_skeleton_one_euro(filled, fps=FPS)
    out["one_euro"] = measure(smooth)

    bvh = build_bvh_positions(smooth)
    _, fk = fit_skeleton_sequence(bvh, d["bl"])
    fk17 = _bvh17_from_fk(fk)
    out["fk_fit"] = measure(fk17)

    bl_dict = {15: d["bl"][3], 16: d["bl"][6]}
    joint_defs = {15: (13, 11), 16: (14, 12)}
    fkc, _ = infer_by_ray_sphere(fk17, bl_dict, d["calib"], d["kpts"], d["scores"],
                                 {15: 0.35, 16: 0.35}, joint_defs, CAMS)
    out["ray_sphere"] = measure(fkc)

    bvh2 = build_bvh_positions(fkc)
    _, fk2 = fit_skeleton_sequence(bvh2, d["bl"])
    out["refit"] = measure(_bvh17_from_fk(fk2))
    return out


def stage6_waterfall_preserve(d):
    """Same as stage6_waterfall but uses fit_skeleton_sequence_preserve_finite
    for BOTH FK calls (fk_fit and refit). Joints that had a finite triangulated
    measurement are snapped back to it after FK; FK only fills missing joints.
    raw / gap_filled / one_euro are unchanged (they don't touch FK)."""
    raw = triangulate_raw(d)
    raw_finite = np.all(np.isfinite(raw), axis=-1)            # (F,17)

    def measure(arr):
        return {
            "all": mpjpe(arr, d["gt_mm"], BODY_J, "none"),
            "raw_finite_only": mpjpe(
                np.where(raw_finite[:, :, None], arr, np.nan), d["gt_mm"], BODY_J, "none"),
            "coverage": coverage(arr, BODY_J)["finite_fraction"],
        }

    out = {"raw": measure(raw)}

    filled, _, _ = fill_gaps_with_logging(raw, [str(i) for i in range(17)], fps=FPS)
    out["gap_filled"] = measure(filled)

    smooth = filter_skeleton_one_euro(filled, fps=FPS)
    out["one_euro"] = measure(smooth)

    bvh = build_bvh_positions(smooth)
    _, fk = fit_skeleton_sequence_preserve_finite(bvh, d["bl"])
    fk17 = _bvh17_from_fk(fk)
    out["fk_fit"] = measure(fk17)

    bl_dict = {15: d["bl"][3], 16: d["bl"][6]}
    joint_defs = {15: (13, 11), 16: (14, 12)}
    fkc, _ = infer_by_ray_sphere(fk17, bl_dict, d["calib"], d["kpts"], d["scores"],
                                 {15: 0.35, 16: 0.35}, joint_defs, CAMS)
    out["ray_sphere"] = measure(fkc)

    bvh2 = build_bvh_positions(fkc)
    _, fk2 = fit_skeleton_sequence_preserve_finite(bvh2, d["bl"])
    out["refit"] = measure(_bvh17_from_fk(fk2))
    return out


# --- IK + retarget + FK harness, input-quality ablation (Task 4) ---
from scipy.spatial.transform import Rotation
from aimocap.retarget.mocap_skeleton import MocapSkeleton, extract_mocap_points
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.fbx_rig import Skeleton

MANNY_TO_COCO = {"head": 0, "upperarm_l": 5, "upperarm_r": 6, "lowerarm_l": 7,
                 "lowerarm_r": 8, "hand_l": 9, "hand_r": 10, "thigh_l": 11,
                 "thigh_r": 12, "calf_l": 13, "calf_r": 14, "foot_l": 15, "foot_r": 16}


def ik_retarget_to_manny(pts3d_cm_internal):
    """Mirror viz_ik_truth.solve_ik_window's rotation-transfer+FK (no foot-lock,
    no temporal stab) on a (F,17,3) cm-internal array. Returns (F,J,3) Manny cm.
    This is the SAME transform the viz used, so b_stage6 here reproduces the viz.

    Frames whose IK-critical anchors (COCO 5,6,11,12 = shoulders+hips, which
    analytic_init needs to build the root frame) are NaN are forward-filled with
    the last finite target so the IK temporal chain stays intact, and their
    Manny output is set to NaN so mpjpe excludes them. This is a no-op for
    b_stage6 and filtered_raw (fully finite); only raw triangulation has gaps.
    """
    pts = np.zeros_like(pts3d_cm_internal)
    pts[..., 0] = pts3d_cm_internal[..., 0]
    pts[..., 1] = -pts3d_cm_internal[..., 2]      # internal Y-up -> Manny Z-up
    pts[..., 2] = pts3d_cm_internal[..., 1]
    # Track solvability BEFORE nan_to_num: analytic_init crashes on zero anchors.
    anchor_finite = np.all(np.isfinite(pts[:, [5, 6, 11, 12]]), axis=(1, 2))   # (F,)
    pts = np.nan_to_num(pts, nan=0.0)
    w = np.any(np.isfinite(pts3d_cm_internal), axis=-1).astype(np.float32)

    fbx = Skeleton(str(ROOT / "Manny.FBX"))
    skel = MocapSkeleton(pts, w, fbx_skel=fbx)
    ik = MocapIKSolver(skel)
    tgt = extract_mocap_points(pts)

    n = len(pts)
    if not anchor_finite.any():
        return np.full((n, fbx.num_joints, 3), np.nan)
    # Forward-fill measured targets on gap frames (hold last finite target) so
    # the IK solve runs on every frame; output for gap frames is NaN-marked.
    first_finite = int(np.argmax(anchor_finite))
    for k in tgt:
        a = tgt[k]
        last = a[first_finite].copy()
        for f in range(n):
            if anchor_finite[f]:
                last = a[f].copy()
            else:
                a[f] = last

    root_seq, q_seq, prev = [], [], None
    for f in range(n):
        m = {k: v[f] for k, v in tgt.items()}
        x = ik.solve_frame(m, prev_x=prev, temporal_weight=0.03)
        prev = x
        rt, lq = ik._state_to_local_rotations(x)
        root_seq.append(rt)
        q_seq.append(lq)
    root_seq = np.array(root_seq)
    q_seq = np.array(q_seq)

    rest_g, rest_gr = fbx.get_forward_kinematics()
    rest_global = Rotation.from_quat(rest_gr)
    rest_local = Rotation.from_quat(np.array(fbx.rest_rotations))
    m2f = {mi: fbx.name_to_idx[nm] for mi, nm in skel.fbx_mapping.items()}
    f2m = {fi: mi for mi, fi in m2f.items()}
    root_i = [i for i, p in enumerate(fbx.parents) if p == -1][0]
    pelvis_i = fbx.name_to_idx["pelvis"]
    manny = np.zeros((n, fbx.num_joints, 3))

    for f in range(n):
        _, gq = ik.forward_kinematics(root_seq[f], q_seq[f])
        mg = Rotation.from_quat(gq)
        fg, fl = [None] * fbx.num_joints, [None] * fbx.num_joints
        for fi in range(fbx.num_joints):
            p = fbx.parents[fi]
            if fi in f2m:
                Rgt = mg[f2m[fi]] * rest_global[fi]
            elif p == -1:
                Rgt = rest_global[fi]
            else:
                Rgt = fg[p] * rest_local[fi]
            fl[fi] = Rgt if p == -1 else fg[p].inv() * Rgt
            fg[fi] = Rgt
        root_world = root_seq[f] - fg[root_i].apply(fbx.rest_translations[pelvis_i])
        rt = root_world - fbx.rest_translations[root_i]
        lq = np.array([r.as_quat() for r in fl])
        lq = Rotation.from_euler("xyz", Rotation.from_quat(lq).as_euler("xyz")).as_quat()
        pos, _ = fbx.get_forward_kinematics(lq, root_translation=rt)
        manny[f] = pos
        if not anchor_finite[f]:
            manny[f] = np.nan     # exclude gap frames from MPJPE
    return manny


def _manny_to_internal_coco(manny, fbx):
    """Build a (F,17,3) internal-cm array from manny joints named in MANNY_TO_COCO.
    Manny Z-up -> internal Y-up: (X, Z, -Y)."""
    arr = np.full((manny.shape[0], 17, 3), np.nan, np.float64)
    for mn, ci in MANNY_TO_COCO.items():
        mi = fbx.name_to_idx[mn]
        p = manny[:, mi]                      # Manny Z-up cm
        arr[:, ci, 0] = p[:, 0]
        arr[:, ci, 1] = p[:, 2]               # Manny -> internal Y-up: (X,Z,-Y)
        arr[:, ci, 2] = -p[:, 1]
    return arr


def _most_moved_window(gt_mm, window=450):
    diff = np.diff(gt_mm, axis=0)
    disp = np.linalg.norm(np.nan_to_num(diff), axis=-1)
    disp[~np.all(np.isfinite(gt_mm[1:]), axis=-1)] = 0
    motion = np.convolve(disp.sum(axis=-1), np.ones(window), mode="valid")
    s = int(np.argmax(motion))
    return slice(s, s + window)


def input_quality_ablation(d, window=450):
    """Measure end-to-end Manny-FK MPJPE for three IK inputs on the most-moved window:
    raw triangulation, filtered_raw (filter_skeleton3d), and b_stage6 (audit path).

    Reports pelvis-aligned MPJPE on the MANNY_TO_COCO joints (matches the viz).
    """
    from aimocap.math.filter import filter_skeleton3d
    fbx = Skeleton(str(ROOT / "Manny.FBX"))
    seg = _most_moved_window(d["gt_mm"], window)
    gt_w = d["gt_mm"][seg]

    # triangulate ONCE, reuse for raw and filtered_raw
    raw_full = triangulate_raw(d)
    inputs = {
        "raw": raw_full[seg],
        "filtered_raw": filter_skeleton3d(raw_full)[seg],
        "b_stage6": np.load(ROOT / "outputs/phase_b_gate1/gate1_arrays.npz",
                            allow_pickle=True)["b_stage6"][seg],
    }
    res = {}
    for name, pts in inputs.items():
        manny = ik_retarget_to_manny(pts)
        solved = _manny_to_internal_coco(manny, fbx)
        res[name] = mpjpe(solved, gt_w, joints=list(MANNY_TO_COCO.values()),
                         alignment="pelvis")
    return res


def main():
    import subprocess
    import sys

    # Allow --window N to control ablation window size (default 450 = the viz's
    # WINDOW_FRAMES). Smaller windows run faster: 120 frames ~11 min, 450 ~2 h.
    window = 450
    if "--window" in sys.argv:
        i = sys.argv.index("--window")
        window = int(sys.argv[i + 1])

    d = load_canonical_data()
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()

    print("=== Stage-6 waterfall (3D points vs GT, BODY_J, absolute) ===")
    wf = stage6_waterfall(d)
    for stage, m in wf.items():
        print(f"  {stage:11s} all={m['all']['mean_mm']:6.2f}mm "
              f"(raw_finite_only={m['raw_finite_only']['mean_mm']:6.2f}mm, "
              f"cov={m['coverage']*100:5.1f}%)")

    print("\n=== Stage-6 waterfall with preserve_finite fix (BODY_J, absolute) ===")
    wfp = stage6_waterfall_preserve(d)
    for stage, m in wfp.items():
        print(f"  {stage:11s} all={m['all']['mean_mm']:6.2f}mm "
              f"(raw_finite_only={m['raw_finite_only']['mean_mm']:6.2f}mm, "
              f"cov={m['coverage']*100:5.1f}%)")

    print("\n=== Waterfall delta (preserve - plain, raw_finite_only) ===")
    for stage in wfp:
        delta = wfp[stage]["raw_finite_only"]["mean_mm"] - wf[stage]["raw_finite_only"]["mean_mm"]
        print(f"  {stage:11s} {delta:+6.2f}mm")

    print(f"\n=== End-to-end input-quality ablation (Manny FK, pelvis-aligned, window={window}) ===")
    abl = input_quality_ablation(d, window=window)
    for name, m in abl.items():
        print(f"  {name:13s} mean={m['mean_mm']:6.2f}mm median={m['median_mm']:6.2f}mm "
              f"p95={m['p95_mm']:6.2f}mm n={m['n']}")

    report = {"commit": commit, "provenance": d["provenance"],
              "waterfall": wf, "waterfall_preserve": wfp,
              "input_ablation": abl}
    out = ROOT / "outputs/accuracy/accuracy_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
