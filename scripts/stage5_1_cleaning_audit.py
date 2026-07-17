"""Stage 5.1 -- Temporal Cleaning Audit (no bone normalization).

Changes from Stage 5.0:
    - Bone-length normalization REMOVED (deferred to Stage 6 rotation-space).
    - Gap-fill policy: <=15 frames cubic spline, >15 frames linear + reconstructed flag.
    - Diagnostic A: per-frame MPJPE vs GT with gap spans shaded; measured vs
      reconstructed MPJPE reported separately; GIF color-codes reconstructed spans.
    - Diagnostic B: 2D confidence distribution for knees/ankles per camera;
      tests min_conf=0.35 for lower-body joints.

Pipeline: gap-fill -> One-Euro smooth  (no normalization)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aimocap  # noqa: F401
from aimocap.math.filter import (
    fill_gaps_with_logging,
    filter_skeleton_one_euro,
    LONG_GAP_THRESHOLD,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
S4_DIR  = ROOT / "outputs" / "stage4_check_c"
S3_NPZ  = ROOT / "outputs" / "stage3_check_c" / "kpts.npz"
OUT_DIR = ROOT / "outputs" / "stage5_1_cleaning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 29.97

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
K = len(JOINT_NAMES)

BONES = {
    "l_upper_arm":    (5, 7),
    "r_upper_arm":    (6, 8),
    "l_forearm":      (7, 9),
    "r_forearm":      (8, 10),
    "l_thigh":        (11, 13),
    "r_thigh":        (12, 14),
    "l_shin":         (13, 15),
    "r_shin":         (14, 16),
    "l_torso":        (5, 11),
    "r_torso":        (6, 12),
    "shoulder_width": (5, 6),
    "hip_width":      (11, 12),
}

DRAW_EDGES = list(BONES.values()) + [(0, 5), (0, 6)]

# Lower-body joints for Diagnostic B
LOWER_JOINTS = {
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_bone_stats(pts):
    stats = {}
    for name, (i, j) in BONES.items():
        lens = np.linalg.norm(pts[:, j] - pts[:, i], axis=1)
        finite = lens[np.isfinite(lens)]
        if len(finite) == 0:
            stats[name] = {"mean": float("nan"), "std": float("nan"), "cv": float("nan")}
        else:
            mu  = float(np.mean(finite))
            sig = float(np.std(finite))
            stats[name] = {"mean": round(mu, 3), "std": round(sig, 3),
                           "cv": round(sig / (mu + 1e-6), 4)}
    return stats


def nan_pct(pts):
    F, K, _ = pts.shape
    nan_joints = np.all(np.isnan(pts), axis=2)
    return 100.0 * nan_joints.sum() / (F * K)


def per_frame_mpjpe(pred, gt, gt_valid):
    """Returns (F,) array of per-frame RC-MPJPE in mm; NaN where gt_valid=False."""
    errs = np.full(len(gt_valid), np.nan)
    for f in range(len(gt_valid)):
        if not gt_valid[f]:
            continue
        p = pred[f]; g = gt[f]
        valid_j = np.isfinite(p).all(1) & np.isfinite(g).all(1)
        if valid_j.sum() < 3:
            continue
        root_p = (p[11] + p[12]) / 2.0
        root_g = (g[11] + g[12]) / 2.0
        diff = np.where(valid_j[:, None], (p - root_p) - (g - root_g), np.nan)
        errs[f] = float(np.nanmean(np.linalg.norm(diff, axis=1))) * 10.0
    return errs


def compute_mpjpe(pred, gt, gt_valid):
    errs = per_frame_mpjpe(pred, gt, gt_valid)
    return float(np.nanmean(errs[gt_valid]))


def mean_jitter(pts):
    delta = np.linalg.norm(np.diff(pts, axis=0), axis=2)
    return float(np.nanmean(delta))


def lag_check(raw, cleaned, joint_idx=9, max_lag=10):
    vel_raw     = np.linalg.norm(np.diff(raw[:, joint_idx], axis=0), axis=1)
    vel_cleaned = np.linalg.norm(np.diff(cleaned[:, joint_idx], axis=0), axis=1)
    def norm(x):
        s = x.std(); return (x - x.mean()) / (s + 1e-9)
    r = np.correlate(norm(vel_cleaned), norm(vel_raw), mode="full")
    lags = np.arange(-(len(vel_raw) - 1), len(vel_raw))
    return float(np.clip(int(lags[np.argmax(r)]), -max_lag, max_lag))


def build_reconstructed_mask(gap_log, F):
    """Returns (F,) bool mask: True where frame is inside a reconstructed span."""
    mask = np.zeros(F, dtype=bool)
    for rec in gap_log:
        if rec.get("reconstructed", False):
            mask[rec["start_frame"]: rec["end_frame"] + 1] = True
    return mask


# ── Diagnostic A: per-frame MPJPE plot ───────────────────────────────────────

def plot_perframe_mpjpe(frame_errs, gap_log, gt_valid, reconstructed_mask, out_path):
    fig, ax = plt.subplots(figsize=(16, 4))
    frames = np.arange(len(frame_errs))

    # Shade gap spans
    for rec in gap_log:
        color = "#FF4444" if rec.get("reconstructed") else "#FFD700"
        ax.axvspan(rec["start_frame"], rec["end_frame"] + 1,
                   alpha=0.25, color=color, zorder=1)

    ax.plot(frames, frame_errs, lw=0.8, color="#3498DB", label="Per-frame RC-MPJPE", zorder=2)

    # Means
    measured_errs = frame_errs[gt_valid & ~reconstructed_mask]
    recon_errs    = frame_errs[gt_valid & reconstructed_mask]
    mean_meas = np.nanmean(measured_errs) if len(measured_errs) > 0 else float("nan")
    mean_recon = np.nanmean(recon_errs)   if len(recon_errs)    > 0 else float("nan")

    ax.axhline(mean_meas,  color="#2ECC71", lw=1.5, ls="--",
               label=f"Mean (measured): {mean_meas:.1f}mm")
    ax.axhline(mean_recon, color="#E74C3C", lw=1.5, ls="--",
               label=f"Mean (reconstructed): {mean_recon:.1f}mm")

    ax.set_xlabel("Frame"); ax.set_ylabel("RC-MPJPE (mm)")
    ax.set_title("Diagnostic A -- Per-Frame RC-MPJPE vs GT\n"
                 "Yellow=short-gap fill, Red=reconstructed (>15f linear fill)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  Per-frame MPJPE plot: {out_path}")
    return mean_meas, mean_recon


# ── Diagnostic A: color-coded animation ──────────────────────────────────────

def make_3panel_anim_colored(raw, cleaned, gt, gt_valid, reconstructed_mask,
                              out_path, n_frames=60, fps=10):
    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(131, projection="3d"),
           fig.add_subplot(132, projection="3d"),
           fig.add_subplot(133, projection="3d")]
    titles = ["Raw", "Cleaned (gap+smooth)\nblue=measured red=reconstructed", "Panoptic GT"]
    skels  = [raw, cleaned, gt]

    # Center on median GT mid-hip
    valid_gt = gt[gt_valid]
    if len(valid_gt) > 0:
        root_pts = (valid_gt[:, 11] + valid_gt[:, 12]) / 2.0
        center = np.nanmedian(root_pts, axis=0)
    else:
        center = np.zeros(3)

    def _plot(ax, pts, title, f, is_recon):
        ax.cla()
        ax.set_title(f"{title}\nf={f}", fontsize=7)
        c = center; r = 100
        ax.set_xlim(c[0]-r, c[0]+r)
        ax.set_ylim(c[1]-r, c[1]+r)
        ax.set_zlim(c[2]-r, c[2]+r)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        bone_color = "#E74C3C" if is_recon else "#2980B9"
        for i, j in DRAW_EDGES:
            if np.all(np.isfinite(pts[f, [i, j]])):
                ax.plot(*pts[f, [i, j]].T, "-", color=bone_color, lw=1.2)
        valid = np.all(np.isfinite(pts[f]), axis=1)
        ax.scatter(*pts[f, valid].T, c="r", s=12, zorder=5)

    def update(frame_idx):
        is_recon = bool(reconstructed_mask[frame_idx])
        for ax, skel, title in zip(axs, skels, titles):
            _plot(ax, skel, title, frame_idx, is_recon if ax == axs[1] else False)
        return []

    frames = [i for i in range(min(n_frames * 3, len(raw))) if gt_valid[i]][:n_frames]
    if not frames:
        frames = list(range(min(n_frames, len(raw))))

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=1000/fps)
    ani.save(str(out_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Animation saved: {out_path}")


# ── Diagnostic B: 2D confidence analysis for lower-body joints ────────────────

def diagnostic_b(s3_npz_path):
    """
    Load Stage-3 NPZ, report 2D confidence distribution for knees/ankles.
    Returns dict of findings.
    """
    print("\nDiagnostic B -- 2D confidence analysis for lower-body joints...")
    if not s3_npz_path.exists():
        print("  [SKIP] Stage-3 NPZ not found:", s3_npz_path)
        return {}

    data   = np.load(s3_npz_path)
    scores = data["scores"]   # (F, n_cams, K)
    F, n_cams, K_ = scores.shape
    print(f"  Scores shape: {scores.shape}  (frames={F}, cams={n_cams}, joints={K_})")

    results = {}
    cam_labels = [f"cam{i}" for i in range(n_cams)]

    thresholds = [0.50, 0.35]

    for jname, ji in LOWER_JOINTS.items():
        print(f"\n  Joint: {jname} (idx {ji})")
        joint_scores = scores[:, :, ji]  # (F, n_cams)
        results[jname] = {}

        for c in range(n_cams):
            s = joint_scores[:, c]
            finite = s[np.isfinite(s)]
            if len(finite) == 0:
                print(f"    [{cam_labels[c]}] all NaN/missing")
                results[jname][cam_labels[c]] = None
                continue
            above_50  = 100.0 * (finite >= 0.50).sum() / len(finite)
            above_35  = 100.0 * (finite >= 0.35).sum() / len(finite)
            print(f"    [{cam_labels[c]}] mean={np.nanmean(s):.3f}  "
                  f"median={np.nanmedian(s):.3f}  "
                  f">=0.50: {above_50:.1f}%  >=0.35: {above_35:.1f}%")
            results[jname][cam_labels[c]] = {
                "mean": round(float(np.nanmean(s)), 4),
                "median": round(float(np.nanmedian(s)), 4),
                "pct_above_50": round(above_50, 1),
                "pct_above_35": round(above_35, 1),
            }

    # Key diagnostic: do lower-body joints HAVE 2D detections but fail the gate?
    print("\n  -- Gate diagnosis --")
    for jname, ji in LOWER_JOINTS.items():
        joint_scores = scores[:, :, ji]  # (F, n_cams)
        # Frames where >=2 cameras have score >= 0.50 vs 0.35
        cams_above_50 = (joint_scores >= 0.50).sum(axis=1)
        cams_above_35 = (joint_scores >= 0.35).sum(axis=1)
        pct_2cams_50 = 100.0 * (cams_above_50 >= 2).mean()
        pct_2cams_35 = 100.0 * (cams_above_35 >= 2).mean()
        print(f"  {jname:<14}  >=2 cams passing 0.50: {pct_2cams_50:.1f}%   "
              f">=2 cams passing 0.35: {pct_2cams_35:.1f}%  "
              f"{'-> GATED OUT by 0.5, rescued by 0.35' if pct_2cams_35 > pct_2cams_50 + 5 else '-> genuinely missing'}")
        results[jname]["gate_pct_2cams_50"] = round(pct_2cams_50, 1)
        results[jname]["gate_pct_2cams_35"] = round(pct_2cams_35, 1)

    return results


# ── Report helpers ────────────────────────────────────────────────────────────

def print_row(label, before, after, threshold, pass_fail, notes=""):
    def fmt(v):
        if v is None: return "--"
        if isinstance(v, float): return f"{v:.2f}"
        return str(v)
    status = "PASS" if pass_fail else "FAIL"
    print(f"{label:<45} {fmt(before):>10}  ->  {fmt(after):>10}   {threshold:<15} {status}  {notes}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Stage 5.1 -- Temporal Cleaning (gap-fill + One-Euro, NO bone normalization)")
    print("=" * 80)

    print("\nLoading Stage-4 raw outputs...")
    raw_pts  = np.load(S4_DIR / "pts3d.npy")
    gt_kpts  = np.load(S4_DIR / "gt_kpts.npy")
    gt_valid = np.load(S4_DIR / "gt_valid.npy")
    F = raw_pts.shape[0]
    print(f"  Shape: {raw_pts.shape}, GT frames valid: {gt_valid.sum()}/{F}")

    # ── BEFORE metrics ────────────────────────────────────────────────────────
    before_mpjpe  = compute_mpjpe(raw_pts, gt_kpts, gt_valid)
    before_bones  = compute_bone_stats(raw_pts)
    before_nan    = nan_pct(raw_pts)
    before_jitter = mean_jitter(raw_pts)
    before_cv_fail = [b for b, s in before_bones.items() if s["cv"] > 0.05]
    print(f"\nBEFORE: RC-MPJPE={before_mpjpe:.1f}mm  NaN={before_nan:.1f}%  "
          f"Jitter={before_jitter:.4f}cm/f  Bones CV>5%: {len(before_cv_fail)}")

    # ── Step 1: Gap fill (new policy) ─────────────────────────────────────────
    print("\nStep 1: Gap fill (cubic <=15f, linear >15f)...")
    filled, gap_log, long_gap_counts = fill_gaps_with_logging(
        raw_pts, JOINT_NAMES, fps=FPS
    )

    reconstructed_mask = build_reconstructed_mask(gap_log, F)
    n_reconstructed_frames = int(reconstructed_mask.sum())
    total_gaps  = len(gap_log)
    long_gaps   = [g for g in gap_log if g["long_gap"]]
    short_gaps  = [g for g in gap_log if not g["long_gap"]]
    max_gap_len = max((g["gap_length"] for g in gap_log), default=0)
    after_nan   = nan_pct(filled)

    print(f"  Total gaps: {total_gaps}  short: {len(short_gaps)}  long/reconstructed: {len(long_gaps)}")
    print(f"  Max gap: {max_gap_len}f  NaN after fill: {after_nan:.1f}%")
    print(f"  Reconstructed frames: {n_reconstructed_frames}/{F} "
          f"({100*n_reconstructed_frames/F:.1f}%)")
    flagged = {j: c for j, c in long_gap_counts.items() if c > 0}
    if flagged:
        print(f"  [UPSTREAM FLAG] joints with long gaps: {flagged}")

    # Save gap log
    gap_log_path = OUT_DIR / "gap_log.json"
    with open(gap_log_path, "w") as fp:
        json.dump({
            "policy": f"cubic if gap_len<={LONG_GAP_THRESHOLD}, linear (reconstructed) if longer",
            "long_gap_threshold": LONG_GAP_THRESHOLD,
            "total_gaps": total_gaps,
            "short_gaps": len(short_gaps),
            "long_gaps_reconstructed": len(long_gaps),
            "max_gap_length": max_gap_len,
            "reconstructed_frames": n_reconstructed_frames,
            "per_joint_long_gap_counts": long_gap_counts,
            "flagged_joints_upstream": flagged,
            "gap_log": gap_log,
        }, fp, indent=2)
    print(f"  Gap log saved: {gap_log_path}")

    # ── Step 2: One-Euro smooth ───────────────────────────────────────────────
    print("\nStep 2: One-Euro smoothing...")
    cleaned = filter_skeleton_one_euro(filled, fps=FPS, min_cutoff=1.0, beta=0.007)
    lag_wrist = lag_check(filled, cleaned, joint_idx=9)
    lag_knee  = lag_check(filled, cleaned, joint_idx=13)
    print(f"  Lag left_wrist: {lag_wrist:.1f}f   left_knee: {lag_knee:.1f}f")

    # ── AFTER metrics ─────────────────────────────────────────────────────────
    print("\nComputing AFTER metrics...")
    after_mpjpe   = compute_mpjpe(cleaned, gt_kpts, gt_valid)
    after_bones   = compute_bone_stats(cleaned)
    after_jitter  = mean_jitter(cleaned)
    after_nan     = nan_pct(cleaned)
    after_cv_fail = [b for b, s in after_bones.items() if s["cv"] > 0.05]
    mpjpe_delta   = 100.0 * (after_mpjpe - before_mpjpe) / (before_mpjpe + 1e-9)
    print(f"  RC-MPJPE: {after_mpjpe:.1f}mm (delta {mpjpe_delta:+.1f}%)  "
          f"Jitter: {after_jitter:.4f}cm/f  CV>5%: {len(after_cv_fail)}")

    # ── Diagnostic A ──────────────────────────────────────────────────────────
    print("\nDiagnostic A: per-frame MPJPE analysis...")
    frame_errs = per_frame_mpjpe(cleaned, gt_kpts, gt_valid)

    mean_meas, mean_recon = plot_perframe_mpjpe(
        frame_errs, gap_log, gt_valid, reconstructed_mask,
        OUT_DIR / "perframe_mpjpe.png",
    )
    print(f"  Mean MPJPE -- measured frames:      {mean_meas:.1f}mm")
    print(f"  Mean MPJPE -- reconstructed frames: {mean_recon:.1f}mm")

    # ── Color-coded animation ─────────────────────────────────────────────────
    print("\nGenerating color-coded animation...")
    make_3panel_anim_colored(
        raw_pts, cleaned, gt_kpts, gt_valid, reconstructed_mask,
        OUT_DIR / "skeleton_3panel.gif", n_frames=60, fps=10,
    )

    # ── Diagnostic B ──────────────────────────────────────────────────────────
    diag_b = diagnostic_b(S3_NPZ)
    with open(OUT_DIR / "diag_b_shin_confidence.json", "w") as fp:
        json.dump(diag_b, fp, indent=2)
    print(f"\n  Diag-B saved: {OUT_DIR / 'diag_b_shin_confidence.json'}")

    # ── Report table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Metric':<45} {'BEFORE':>10}     {'AFTER':>10}   {'Threshold':<15} {'Status'}")
    print("-" * 90)

    print_row("RC-MPJPE (mm)", before_mpjpe, after_mpjpe,
              "<5% delta", mpjpe_delta <= 5.0,
              f"delta {mpjpe_delta:+.1f}%")
    print_row("NaN joint-frames (%)", before_nan, after_nan,
              "~0%", after_nan < 1.0)
    print_row("Bones with CV>5%", len(before_cv_fail), len(after_cv_fail),
              "0", len(after_cv_fail) == 0)
    print_row("Jitter (cm/frame)", before_jitter, after_jitter,
              "lower", after_jitter < before_jitter)
    print_row("Max gap filled (frames)", None, max_gap_len,
              f"flag if >{LONG_GAP_THRESHOLD}", max_gap_len <= LONG_GAP_THRESHOLD)
    print_row("Long/reconstructed gaps", None, len(long_gaps),
              "0 preferred", len(long_gaps) == 0)
    print_row("Reconstructed frames", None, n_reconstructed_frames,
              "0 preferred", n_reconstructed_frames == 0)
    print_row("One-Euro lag -- wrist (frames)", None, lag_wrist,
              "<=2", abs(lag_wrist) <= 2)
    print_row("One-Euro lag -- knee (frames)", None, lag_knee,
              "<=2", abs(lag_knee) <= 2)
    print_row("MPJPE on measured frames (mm)", None, mean_meas,
              "~63mm", mean_meas <= 70.0)
    print_row("MPJPE on reconstructed frames (mm)", None, mean_recon,
              "report only", True)

    # Per-bone CV
    print(f"\n{'Bone':<20} {'CV before':>10} {'CV after':>10} {'Status'}")
    print("-" * 48)
    for bone in BONES:
        cv_b = before_bones[bone]["cv"]
        cv_a = after_bones[bone]["cv"]
        pf   = "PASS" if cv_a <= 0.05 else "FAIL"
        print(f"  {bone:<18} {cv_b:>10.4f} {cv_a:>10.4f}   {pf}")

    # Diagnostic B summary
    if diag_b:
        print("\nDiagnostic B Summary -- Lower-body gate analysis:")
        print(f"  {'Joint':<14}  {'pct>=2cams@0.50':>16}  {'pct>=2cams@0.35':>16}  Verdict")
        print("  " + "-" * 65)
        for jname in LOWER_JOINTS:
            if jname in diag_b:
                p50 = diag_b[jname].get("gate_pct_2cams_50", float("nan"))
                p35 = diag_b[jname].get("gate_pct_2cams_35", float("nan"))
                rescued = p35 - p50 > 5.0
                verdict = "GATED OUT at 0.5, rescued at 0.35" if rescued else "genuinely missing"
                print(f"  {jname:<14}  {p50:>16.1f}  {p35:>16.1f}  {verdict}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics = {
        "stage": "5.1",
        "note": "No bone normalization -- deferred to Stage 6",
        "before": {
            "rc_mpjpe_mm": round(before_mpjpe, 2),
            "nan_pct": round(before_nan, 2),
            "jitter_cm": round(before_jitter, 4),
            "bones_cv_failing": before_cv_fail,
        },
        "after": {
            "rc_mpjpe_mm": round(after_mpjpe, 2),
            "mpjpe_delta_pct": round(mpjpe_delta, 2),
            "nan_pct": round(after_nan, 2),
            "jitter_cm": round(after_jitter, 4),
            "bones_cv_failing": after_cv_fail,
            "lag_wrist_frames": lag_wrist,
            "lag_knee_frames": lag_knee,
        },
        "diag_a": {
            "mean_mpjpe_measured_frames_mm": round(mean_meas, 2) if not np.isnan(mean_meas) else None,
            "mean_mpjpe_reconstructed_frames_mm": round(mean_recon, 2) if not np.isnan(mean_recon) else None,
            "reconstructed_frames": n_reconstructed_frames,
            "total_frames": F,
        },
        "gap_fill": {
            "total_gaps": total_gaps,
            "short_cubic": len(short_gaps),
            "long_reconstructed_linear": len(long_gaps),
            "max_gap_frames": max_gap_len,
            "flagged_joints": flagged,
        },
        "per_bone_cv": {
            b: {"before": before_bones[b]["cv"], "after": after_bones[b]["cv"]}
            for b in BONES
        },
        "diag_b": diag_b,
    }
    with open(OUT_DIR / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"\nMetrics saved: {OUT_DIR / 'metrics.json'}")
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
