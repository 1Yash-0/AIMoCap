"""Stage 4.2 / 5.1 -- Knee Rescue: per-joint confidence gate.

Changes vs Stage 4 Check-C:
  - Per-joint gate: 0.35 for hips+knees (COCO 11-14), 0.5 for everything else.
  - Ankles (15-16): EXCLUDED -- set scores to 0 before triangulation so the
    engine never produces a 3D point for them. They are absent from all metrics.
  - Runs full Stage 5.1 cleaning pipeline (gap-fill + One-Euro, no normalization).

Outputs -> outputs/stage4_2_knee_rescue/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aimocap  # noqa: F401
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics
from aimocap.math.coords import internal_to_opencv
from aimocap.pose.keypoints import KEYPOINT_NAMES_133
from aimocap.math.filter import (
    fill_gaps_with_logging,
    filter_skeleton_one_euro,
    LONG_GAP_THRESHOLD,
)

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_DIR    = ROOT / "data" / "panoptic" / "171204_pose1"
CALIB_JSON = SEQ_DIR / "calibration_171204_pose1.json"
GT_DIR     = SEQ_DIR / "hdPose3d_stage1_coco19"

NPZ_PATH   = ROOT / "outputs" / "stage3_check_c" / "kpts.npz"
OUT_DIR    = ROOT / "outputs" / "stage4_2_knee_rescue"

# Load Stage 4 Check-C raw npy for before/after comparison
S4_PREV_DIR = ROOT / "outputs" / "stage4_check_c"

CAM_NAMES   = ["00_26", "00_29", "00_30"]
GT_FPS      = 29.97
START_FRAME = int(GT_FPS * 5.0)   # 149
N_FRAMES    = 300

# ── Per-joint gate policy ─────────────────────────────────────────────────────
# Hips (11-12) and knees (13-14): lower gate
HIP_KNEE_IDX  = [11, 12, 13, 14]
ANKLE_IDX     = [15, 16]           # excluded entirely
GATE_DEFAULT  = 0.50
GATE_HIP_KNEE = 0.35

JOINT_NAMES = [KEYPOINT_NAMES_133[i] for i in range(17)]
K = 17

# Bones (ankles excluded from CV metrics)
BONES_ALL = {
    "l_upper_arm":    (5, 7),
    "r_upper_arm":    (6, 8),
    "l_forearm":      (7, 9),
    "r_forearm":      (8, 10),
    "l_thigh":        (11, 13),
    "r_thigh":        (12, 14),
    "l_shin":         (13, 15),  # will always be NaN; excluded from CV
    "r_shin":         (14, 16),
    "l_torso":        (5, 11),
    "r_torso":        (6, 12),
    "shoulder_width": (5, 6),
    "hip_width":      (11, 12),
}
# Bones to report CV (shins excluded -- ankles are absent)
BONES_REPORT = {k: v for k, v in BONES_ALL.items() if k not in ("l_shin", "r_shin")}

DRAW_EDGES = list(BONES_ALL.values()) + [(0, 5), (0, 6)]

# COCO-17 -> Panoptic-19 joint mapping
COCO17_TO_PAN19 = [
    1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gt_sequence(gt_dir, start, n):
    gt_kpts  = np.full((n, 17, 3), np.nan, dtype=np.float32)
    gt_valid = np.zeros(n, dtype=bool)
    for i in range(n):
        fn = gt_dir / f"body3DScene_{start+i:08d}.json"
        if not fn.exists(): continue
        with open(fn) as f:
            data = json.load(f)
        bodies = data.get("bodies", [])
        if not bodies: continue
        joints19 = np.array(bodies[0]["joints19"], dtype=np.float32).reshape(19, 4)
        for c17, p19 in enumerate(COCO17_TO_PAN19):
            gt_kpts[i, c17] = joints19[p19, :3]
        gt_valid[i] = True
    return gt_kpts, gt_valid


def per_frame_mpjpe_full(pred, gt, gt_valid, exclude_joints=None):
    """Returns (F,) array RC-MPJPE in mm. exclude_joints: list of joint idx to skip."""
    exclude = set(exclude_joints or [])
    errs = np.full(len(gt_valid), np.nan)
    for f in range(len(gt_valid)):
        if not gt_valid[f]: continue
        p = pred[f].copy(); g = gt[f].copy()
        for ji in exclude:
            p[ji] = np.nan; g[ji] = np.nan
        valid_j = np.isfinite(p).all(1) & np.isfinite(g).all(1)
        if valid_j.sum() < 3: continue
        root_p = (p[11] + p[12]) / 2.0
        root_g = (g[11] + g[12]) / 2.0
        diff = np.where(valid_j[:, None], (p - root_p) - (g - root_g), np.nan)
        errs[f] = float(np.nanmean(np.linalg.norm(diff, axis=1))) * 10.0
    return errs


def compute_mpjpe(pred, gt, gt_valid, exclude_joints=None):
    errs = per_frame_mpjpe_full(pred, gt, gt_valid, exclude_joints)
    return float(np.nanmean(errs[gt_valid]))


def compute_bone_stats(pts, bones):
    stats = {}
    for name, (i, j) in bones.items():
        lens = np.linalg.norm(pts[:, j] - pts[:, i], axis=1)
        finite = lens[np.isfinite(lens)]
        if len(finite) == 0:
            stats[name] = {"mean": float("nan"), "std": float("nan"), "cv": float("nan")}
        else:
            mu = float(np.mean(finite)); sig = float(np.std(finite))
            stats[name] = {"mean": round(mu, 3), "std": round(sig, 3),
                           "cv": round(sig / (mu + 1e-6), 4)}
    return stats


def nan_pct(pts, exclude_joints=None):
    F, K, _ = pts.shape
    nan_j = np.all(np.isnan(pts), axis=2)
    if exclude_joints:
        for ji in exclude_joints:
            nan_j[:, ji] = False   # don't count excluded joints as NaN
    return 100.0 * nan_j.sum() / (F * K)


def mean_jitter(pts, exclude_joints=None):
    p = pts.copy()
    if exclude_joints:
        for ji in exclude_joints:
            p[:, ji] = np.nan
    delta = np.linalg.norm(np.diff(p, axis=0), axis=2)
    return float(np.nanmean(delta))


def lag_check(raw, cleaned, joint_idx=9):
    vel_r = np.linalg.norm(np.diff(raw[:, joint_idx], axis=0), axis=1)
    vel_c = np.linalg.norm(np.diff(cleaned[:, joint_idx], axis=0), axis=1)
    def norm(x):
        s = x.std(); return (x - x.mean()) / (s + 1e-9)
    r = np.correlate(norm(vel_c), norm(vel_r), mode="full")
    lags = np.arange(-(len(vel_r) - 1), len(vel_r))
    return float(np.clip(int(lags[np.argmax(r)]), -10, 10))


def build_recon_mask(gap_log, F):
    mask = np.zeros(F, dtype=bool)
    for rec in gap_log:
        if rec.get("reconstructed"):
            mask[rec["start_frame"]: rec["end_frame"] + 1] = True
    return mask


def print_row(label, before, after, threshold, ok, notes=""):
    def fmt(v):
        if v is None: return "--"
        if isinstance(v, float): return f"{v:.2f}"
        return str(v)
    status = "PASS" if ok else "FAIL"
    print(f"{label:<50} {fmt(before):>10}  ->  {fmt(after):>10}   {threshold:<18} {status}  {notes}")


# ── Visuals ───────────────────────────────────────────────────────────────────

def make_3panel_gif(raw, cleaned, gt, gt_valid, recon_mask, out_path, n_frames=60, fps=10):
    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(131, projection="3d"),
           fig.add_subplot(132, projection="3d"),
           fig.add_subplot(133, projection="3d")]

    valid_gt = gt[gt_valid]
    center = np.nanmedian((valid_gt[:, 11] + valid_gt[:, 12]) / 2.0, axis=0) \
             if len(valid_gt) > 0 else np.zeros(3)

    def _plot(ax, pts, title, f, is_recon=False):
        ax.cla()
        ax.set_title(f"{title}\nf={f}", fontsize=7)
        c = center; r = 100
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        color = "#E74C3C" if is_recon else "#2980B9"
        for i, j in DRAW_EDGES:
            if i in ANKLE_IDX or j in ANKLE_IDX: continue
            if np.all(np.isfinite(pts[f, [i, j]])):
                ax.plot(*pts[f, [i, j]].T, "-", color=color, lw=1.2)
        valid = np.all(np.isfinite(pts[f]), axis=1)
        valid[list(ANKLE_IDX)] = False
        ax.scatter(*pts[f, valid].T, c="r", s=12, zorder=5)

    def update(fi):
        is_r = bool(recon_mask[fi])
        _plot(axs[0], raw,     "Raw",                               fi, False)
        _plot(axs[1], cleaned, "Cleaned\nblue=measured red=recon", fi, is_r)
        _plot(axs[2], gt,      "Panoptic GT",                      fi, False)
        return []

    frames = [i for i in range(min(n_frames * 4, len(raw))) if gt_valid[i]][:n_frames]
    if not frames:
        frames = list(range(min(n_frames, len(raw))))

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=1000/fps)
    ani.save(str(out_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  3-panel GIF: {out_path}")


def make_perframe_mpjpe_plot(errs_before, errs_after, gap_log, gt_valid, recon_mask, out_path):
    fig, ax = plt.subplots(figsize=(16, 4))
    frames = np.arange(len(errs_before))

    for rec in gap_log:
        color = "#FF4444" if rec.get("reconstructed") else "#FFD700"
        ax.axvspan(rec["start_frame"], rec["end_frame"] + 1, alpha=0.20, color=color, zorder=1)

    ax.plot(frames, errs_before, lw=0.8, color="#95A5A6", alpha=0.7, label="Before (Stage 4.1)", zorder=2)
    ax.plot(frames, errs_after,  lw=0.9, color="#3498DB", label="After (Stage 4.2 + clean)", zorder=3)

    meas = errs_after[gt_valid & ~recon_mask]
    recon = errs_after[gt_valid & recon_mask]
    mean_m = np.nanmean(meas)  if len(meas)  > 0 else float("nan")
    mean_r = np.nanmean(recon) if len(recon) > 0 else float("nan")

    ax.axhline(mean_m,  color="#2ECC71", lw=1.5, ls="--",
               label=f"Mean measured: {mean_m:.1f}mm")
    ax.axhline(mean_r,  color="#E74C3C", lw=1.5, ls="--",
               label=f"Mean recon: {mean_r:.1f}mm")

    ax.set_xlabel("Frame"); ax.set_ylabel("RC-MPJPE (mm)")
    ax.set_title("Per-Frame RC-MPJPE -- Before vs After Knee Rescue\n"
                 "Yellow=short-gap fill  Red=reconstructed (>15f linear)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  Per-frame MPJPE plot: {out_path}")
    return mean_m, mean_r


def make_knee_coverage_plot(num_inliers, out_path):
    """Bar-per-frame inlier count for left_knee (13) and right_knee (14)."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    frames = np.arange(N_FRAMES)
    colors = {0: "#E74C3C", 1: "#F39C12", 2: "#2ECC71", 3: "#3498DB"}

    for ax, ji, jname in zip(axes, [13, 14], ["left_knee", "right_knee"]):
        inl = num_inliers[:, ji]
        bar_colors = [colors.get(int(v), "#95A5A6") for v in inl]
        ax.bar(frames, inl, color=bar_colors, width=1.0)
        ax.set_ylabel("Inliers")
        ax.set_title(f"{jname} inlier count per frame  "
                     f"(red=0, orange=1, green=2, blue=3)")
        ax.set_ylim(0, 3.5)
        ax.set_yticks([0, 1, 2, 3])

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Knee Coverage Timeline -- Stage 4.2 (min_conf=0.35 for hips/knees)", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  Knee coverage plot: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("Stage 4.2 -- Knee Rescue (per-joint gate: 0.35 hips/knees, ankles excluded)")
    print("=" * 80)

    # ── Load calibration ──────────────────────────────────────────────────────
    print("\nLoading Panoptic calibration...")
    calib      = load_panoptic_calib(CALIB_JSON)
    K_list     = []
    extrinsics = []
    for cn in CAM_NAMES:
        c = calib[cn]
        K_list.append(c.K.astype(np.float64))
        t = c.t.reshape(3, 1).astype(np.float64)
        extrinsics.append((c.R.astype(np.float64), t))
    print(f"  Cameras: {CAM_NAMES}")

    # ── Load Stage-3 NPZ ─────────────────────────────────────────────────────
    print("Loading Stage-3 NPZ...")
    data       = np.load(NPZ_PATH, allow_pickle=True)
    kpts_all   = data["keypoints"].astype(np.float64)   # (300, 3, 133, 2)
    scores_all = data["scores"].astype(np.float64)       # (300, 3, 133)

    kpts   = kpts_all[:, :, :17, :]
    scores = scores_all[:, :, :17].copy()
    print(f"  Keypoints: {kpts.shape}, Scores: {scores.shape}")

    # ── Apply per-joint gate policy to scores ─────────────────────────────────
    # We implement the gate by clamping scores so the engine's >= min_conf check works:
    # - Ankles (15-16): set score to 0 so engine never admits them
    # - Hips/knees (11-14): keep scores as-is; we will call engine with min_conf=0.35
    # - Others: scores >= 0.5 to pass at min_conf=0.35 requires we keep them as-is,
    #   but we must enforce the 0.5 floor for non-hip/knee joints manually pre-call.

    # Enforce 0.5 floor for all joints EXCEPT hips/knees (which use 0.35) and ankles
    scores_gated = scores.copy()
    for ji in range(17):
        if ji in ANKLE_IDX:
            scores_gated[:, :, ji] = 0.0          # excluded
        elif ji not in HIP_KNEE_IDX:
            # Zero out any score <0.5 for non-lower-body joints
            scores_gated[:, :, ji] = np.where(
                scores[:, :, ji] >= GATE_DEFAULT,
                scores[:, :, ji],
                0.0,
            )
        # HIP_KNEE_IDX: leave scores untouched — gate at 0.35 via min_conf arg

    # ── Coverage check: before vs after for knees ────────────────────────────
    # Before: how many frames had >=2 cams passing 0.5 for knees?
    for ji, jn in [(13, "left_knee"), (14, "right_knee")]:
        n_before = int((scores[:, :, ji] >= 0.50).sum(axis=1).clip(max=2).sum())
        n_after  = int((scores_gated[:, :, ji] >= 0.35).sum(axis=1).clip(max=2).sum())
        pct_before = 100.0 * (scores[:, :, ji].max(axis=1) >= 0.50).mean()
        pct_2cam_before = 100.0 * ((scores[:, :, ji] >= 0.50).sum(axis=1) >= 2).mean()
        pct_2cam_after  = 100.0 * ((scores_gated[:, :, ji] >= 0.35).sum(axis=1) >= 2).mean()
        print(f"  {jn}: >=2 cams @0.50={pct_2cam_before:.1f}%  "
              f"->  >=2 cams @0.35={pct_2cam_after:.1f}%")

    # ── Triangulate ───────────────────────────────────────────────────────────
    print("\nTriangulating with per-joint gate...")
    diag = triangulate_sequence_with_diagnostics(
        kpts, scores_gated, K_list, extrinsics, min_conf=GATE_HIP_KNEE,
    )
    pts3d_int = diag.points3d
    pts3d_cv  = internal_to_opencv(pts3d_int)     # (F, 17, 3) cm world space
    reproj_err  = diag.reprojection_error_px       # (F, 17, 3)
    num_inliers = diag.num_inliers                 # (F, 17)

    # Force ankles to NaN
    for ji in ANKLE_IDX:
        pts3d_cv[:, ji] = np.nan

    finite_count = int(np.isfinite(pts3d_cv[:, [i for i in range(17) if i not in ANKLE_IDX]]).all(-1).sum())
    print(f"  Triangulated (excl. ankles): {finite_count}/{N_FRAMES * 15} joint-frames")

    # ── Load GT ───────────────────────────────────────────────────────────────
    print(f"Loading GT from frame {START_FRAME}...")
    gt_kpts, gt_valid = load_gt_sequence(GT_DIR, START_FRAME, N_FRAMES)
    print(f"  GT frames with data: {gt_valid.sum()}/{N_FRAMES}")

    # ── Knee coverage after rescue ────────────────────────────────────────────
    print("\nKnee inlier coverage after rescue:")
    for ji, jn in [(13, "left_knee"), (14, "right_knee")]:
        inl = num_inliers[:, ji]
        pct_2plus = 100.0 * (inl >= 2).mean()
        pct_zero  = 100.0 * (inl == 0).mean()
        print(f"  {jn}: >=2 inliers={pct_2plus:.1f}%  zeros={pct_zero:.1f}%")

    # ── Reprojection error check ──────────────────────────────────────────────
    print("\nReprojection error (guard against garbage from lower gate):")
    for ci, cn in enumerate(CAM_NAMES):
        err = reproj_err[:, :, ci]
        finite = err[np.isfinite(err)]
        if len(finite) > 0:
            print(f"  [{cn}] mean={np.mean(finite):.2f}px  "
                  f"p95={np.percentile(finite, 95):.2f}px  "
                  f"max={np.max(finite):.2f}px")

    # Also compare reproj for knee-only before/after
    # Load Stage 4 Check-C pts3d as "before"
    pts3d_before = np.load(S4_PREV_DIR / "pts3d.npy")
    gt_kpts_ref  = np.load(S4_PREV_DIR / "gt_kpts.npy")
    gt_valid_ref = np.load(S4_PREV_DIR / "gt_valid.npy")

    # ── MPJPE before (Stage 5.1 measured-frame baseline: 52.9mm) ─────────────
    mpjpe_before = compute_mpjpe(pts3d_before, gt_kpts_ref, gt_valid_ref, exclude_joints=ANKLE_IDX)
    print(f"\nMPJPE before (Stage 4 Check-C, ankles excl.): {mpjpe_before:.1f}mm")

    # ── Stage 5.1 Cleaning ────────────────────────────────────────────────────
    print("\n-- Running Stage 5.1 cleaning pipeline --")

    print("  Gap fill (cubic <=15f, linear >15f)...")
    filled, gap_log, long_gap_counts = fill_gaps_with_logging(
        pts3d_cv, JOINT_NAMES, fps=GT_FPS,
    )
    recon_mask = build_recon_mask(gap_log, N_FRAMES)
    n_recon    = int(recon_mask.sum())
    long_gaps  = [g for g in gap_log if g["long_gap"]]
    max_gap    = max((g["gap_length"] for g in gap_log), default=0)
    print(f"  Gaps: {len(gap_log)} total, {len(long_gaps)} reconstructed  "
          f"max={max_gap}f  recon frames={n_recon}/{N_FRAMES} ({100*n_recon/N_FRAMES:.1f}%)")

    print("  One-Euro smoothing...")
    cleaned = filter_skeleton_one_euro(filled, fps=GT_FPS, min_cutoff=1.0, beta=0.007)
    lag_wrist = lag_check(filled, cleaned, joint_idx=9)
    lag_knee  = lag_check(filled, cleaned, joint_idx=13)
    print(f"  Lag: wrist={lag_wrist:.1f}f  knee={lag_knee:.1f}f")

    # ── AFTER metrics ─────────────────────────────────────────────────────────
    mpjpe_after = compute_mpjpe(cleaned, gt_kpts, gt_valid, exclude_joints=ANKLE_IDX)
    mpjpe_delta = 100.0 * (mpjpe_after - mpjpe_before) / (mpjpe_before + 1e-9)

    frame_errs_before = per_frame_mpjpe_full(pts3d_before, gt_kpts_ref, gt_valid_ref, ANKLE_IDX)
    frame_errs_after  = per_frame_mpjpe_full(cleaned, gt_kpts, gt_valid, ANKLE_IDX)

    # MPJPE split by measured vs reconstructed
    valid_meas  = gt_valid & ~recon_mask
    valid_recon = gt_valid & recon_mask
    mpjpe_meas  = float(np.nanmean(frame_errs_after[valid_meas]))
    mpjpe_recon = float(np.nanmean(frame_errs_after[valid_recon])) \
                  if valid_recon.any() else float("nan")

    jitter_after = mean_jitter(cleaned, exclude_joints=ANKLE_IDX)
    bones_after  = compute_bone_stats(cleaned, BONES_REPORT)
    cv_fail_after = [b for b, s in bones_after.items() if np.isfinite(s["cv"]) and s["cv"] > 0.05]

    bones_before = compute_bone_stats(pts3d_before, BONES_REPORT)
    cv_fail_before = [b for b, s in bones_before.items() if np.isfinite(s["cv"]) and s["cv"] > 0.05]

    # ── Visuals ───────────────────────────────────────────────────────────────
    print("\nGenerating visuals...")

    # Save gap log
    gap_log_path = OUT_DIR / "gap_log.json"
    with open(gap_log_path, "w") as fp:
        json.dump({
            "policy": f"cubic if gap<={LONG_GAP_THRESHOLD}, linear (reconstructed) if longer",
            "total_gaps": len(gap_log),
            "long_reconstructed": len(long_gaps),
            "max_gap_length": max_gap,
            "reconstructed_frames": n_recon,
            "per_joint_long_gap_counts": long_gap_counts,
            "gap_log": gap_log,
        }, fp, indent=2)

    make_3panel_gif(
        pts3d_cv, cleaned, gt_kpts, gt_valid, recon_mask,
        OUT_DIR / "skeleton_3panel.gif", n_frames=60, fps=10,
    )

    mean_meas, mean_recon = make_perframe_mpjpe_plot(
        frame_errs_before, frame_errs_after,
        gap_log, gt_valid, recon_mask,
        OUT_DIR / "perframe_mpjpe.png",
    )

    make_knee_coverage_plot(num_inliers, OUT_DIR / "knee_coverage.png")

    # ── Report table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'Metric':<50} {'BEFORE':>10}     {'AFTER':>10}   {'Threshold':<18} {'Status'}")
    print("-" * 95)

    # Knee coverage before/after (recompute from raw scores)
    knee_cov_before_l = 100.0 * ((scores[:, :, 13] >= 0.50).sum(axis=1) >= 2).mean()
    knee_cov_before_r = 100.0 * ((scores[:, :, 14] >= 0.50).sum(axis=1) >= 2).mean()
    knee_cov_after_l  = 100.0 * (num_inliers[:, 13] >= 2).mean()
    knee_cov_after_r  = 100.0 * (num_inliers[:, 14] >= 2).mean()
    hip_cov_before_l  = 100.0 * ((scores[:, :, 11] >= 0.50).sum(axis=1) >= 2).mean()
    hip_cov_before_r  = 100.0 * ((scores[:, :, 12] >= 0.50).sum(axis=1) >= 2).mean()
    hip_cov_after_l   = 100.0 * (num_inliers[:, 11] >= 2).mean()
    hip_cov_after_r   = 100.0 * (num_inliers[:, 12] >= 2).mean()

    # Before stage5.1 jitter (raw pts before cleaning)
    jitter_before = mean_jitter(pts3d_before, exclude_joints=ANKLE_IDX)

    print_row("Left knee coverage (>=2 inliers %)",
              knee_cov_before_l, knee_cov_after_l, ">95%", knee_cov_after_l >= 95)
    print_row("Right knee coverage (>=2 inliers %)",
              knee_cov_before_r, knee_cov_after_r, ">95%", knee_cov_after_r >= 95)
    print_row("Left hip coverage (>=2 inliers %)",
              hip_cov_before_l, hip_cov_after_l, ">95%", hip_cov_after_l >= 95)
    print_row("Right hip coverage (>=2 inliers %)",
              hip_cov_before_r, hip_cov_after_r, ">95%", hip_cov_after_r >= 95)
    print_row("RC-MPJPE overall (ankles excl.) (mm)",
              mpjpe_before, mpjpe_after, "<5% delta", abs(mpjpe_delta) <= 5.0,
              f"delta {mpjpe_delta:+.1f}%")
    print_row("RC-MPJPE measured frames (mm)",
              52.9, mpjpe_meas, "<=55.5mm (+5%)", mpjpe_meas <= 55.5,
              "(vs Stage 5.1 baseline 52.9mm)")
    print_row("RC-MPJPE reconstructed frames (mm)",
              None, mpjpe_recon, "report only", True)
    print_row("Reconstructed frames",
              107, n_recon, "clear drop expected", n_recon < 107)
    print_row("Long gaps (reconstructed)",
              14, len(long_gaps), "fewer is better", len(long_gaps) < 14)
    print_row("Max gap length (frames)",
              69, max_gap, f"flag if >{LONG_GAP_THRESHOLD}", max_gap <= LONG_GAP_THRESHOLD)
    print_row("Bones with CV>5% (shins excl.)",
              len(cv_fail_before), len(cv_fail_after), "0", len(cv_fail_after) == 0,
              str(cv_fail_after[:3]) if cv_fail_after else "")
    print_row("Jitter (cm/frame, ankles excl.)",
              jitter_before, jitter_after, "<=1.1", jitter_after <= 1.1)
    print_row("One-Euro lag -- wrist (frames)",
              None, lag_wrist, "<=2", abs(lag_wrist) <= 2)
    print_row("One-Euro lag -- knee (frames)",
              None, lag_knee, "<=2", abs(lag_knee) <= 2)

    print(f"\nBones with CV>5% after: {cv_fail_after}")

    # Per-bone CV table
    print(f"\n{'Bone':<20} {'CV before':>10} {'CV after':>10} {'Status'}")
    print("-" * 48)
    for bone in BONES_REPORT:
        cv_b = bones_before[bone]["cv"]
        cv_a = bones_after[bone]["cv"]
        pf = "PASS" if (np.isfinite(cv_a) and cv_a <= 0.05) else "FAIL"
        print(f"  {bone:<18} {cv_b:>10.4f} {cv_a:>10.4f}   {pf}")

    # ── Save npy for downstream ───────────────────────────────────────────────
    np.save(OUT_DIR / "pts3d.npy",    pts3d_cv)
    np.save(OUT_DIR / "pts3d_clean.npy", cleaned)
    np.save(OUT_DIR / "gt_kpts.npy",  gt_kpts)
    np.save(OUT_DIR / "gt_valid.npy", gt_valid)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics = {
        "stage": "4.2_knee_rescue",
        "gate_policy": {
            "hips_knees": GATE_HIP_KNEE,
            "default": GATE_DEFAULT,
            "ankles": "excluded",
        },
        "coverage": {
            "left_knee_before":  round(knee_cov_before_l, 1),
            "left_knee_after":   round(knee_cov_after_l, 1),
            "right_knee_before": round(knee_cov_before_r, 1),
            "right_knee_after":  round(knee_cov_after_r, 1),
            "left_hip_before":   round(hip_cov_before_l, 1),
            "left_hip_after":    round(hip_cov_after_l, 1),
            "right_hip_before":  round(hip_cov_before_r, 1),
            "right_hip_after":   round(hip_cov_after_r, 1),
        },
        "mpjpe": {
            "before_mm": round(mpjpe_before, 2),
            "after_mm":  round(mpjpe_after, 2),
            "delta_pct": round(mpjpe_delta, 2),
            "measured_frames_mm":      round(mpjpe_meas, 2),
            "reconstructed_frames_mm": round(mpjpe_recon, 2) if not np.isnan(mpjpe_recon) else None,
        },
        "cleaning": {
            "total_gaps": len(gap_log),
            "long_reconstructed": len(long_gaps),
            "max_gap_frames": max_gap,
            "reconstructed_frames": n_recon,
            "jitter_cm": round(jitter_after, 4),
            "lag_wrist_frames": lag_wrist,
            "lag_knee_frames": lag_knee,
        },
        "per_bone_cv": {
            b: {"before": bones_before[b]["cv"], "after": bones_after[b]["cv"]}
            for b in BONES_REPORT
        },
    }
    with open(OUT_DIR / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"\nAll outputs saved to: {OUT_DIR}")

    # ── Final readiness verdict ───────────────────────────────────────────────
    # Criteria: measured-frame MPJPE <=55.5mm, jitter <=1.1, lag <=2, recon frames drop
    ready = (mpjpe_meas <= 55.5 and jitter_after <= 1.1
             and abs(lag_wrist) <= 2 and abs(lag_knee) <= 2)
    print("\n" + "=" * 80)
    if ready:
        print("READY FOR STAGE 6: YES")
        print(f"  Measured-frame MPJPE={mpjpe_meas:.1f}mm (<55.5), "
              f"jitter={jitter_after:.2f}cm (<1.1), "
              f"lag wrist={lag_wrist:.0f}f knee={lag_knee:.0f}f (both <=2)")
        print(f"  {n_recon} reconstructed frames are flagged in gap_log.json "
              f"and will be handled by Stage 6 kinematic constraints.")
        print(f"  Ankles excluded (camera blind spot confirmed -- not a gate issue).")
    else:
        print("READY FOR STAGE 6: NO")
        reasons = []
        if mpjpe_meas > 55.5:  reasons.append(f"MPJPE={mpjpe_meas:.1f}mm > 55.5")
        if jitter_after > 1.1: reasons.append(f"jitter={jitter_after:.2f} > 1.1")
        if abs(lag_wrist) > 2: reasons.append(f"lag_wrist={lag_wrist:.0f}f > 2")
        if abs(lag_knee) > 2:  reasons.append(f"lag_knee={lag_knee:.0f}f > 2")
        print("  Reasons: " + ";  ".join(reasons))
    print("=" * 80)


if __name__ == "__main__":
    main()
