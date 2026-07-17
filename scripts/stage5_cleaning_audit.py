"""Stage 5 — Temporal Cleaning Audit.

Pipeline:
    1. Load raw triangulated 3D from outputs/stage4_check_c/pts3d.npy
    2. Gap-fill with structured logging (flags long gaps per joint)
    3. One-Euro smooth
    4. Median bone-length normalization (root-walk)
    5. Report BEFORE vs AFTER on every required metric
    6. Generate visuals → outputs/stage5_cleaning/

No new math — glue code only over aimocap.math.filter.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aimocap  # noqa: F401
from aimocap.math.filter import (
    fill_gaps_with_logging,
    filter_skeleton_one_euro,
    normalize_bone_lengths_median,
    LONG_GAP_THRESHOLD,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
S4_DIR  = ROOT / "outputs" / "stage4_check_c"
OUT_DIR = ROOT / "outputs" / "stage5_cleaning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FPS = 29.97

# COCO-17 joint names (same order as stage4)
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
K = len(JOINT_NAMES)

# Bones used in stage4 audit (parent_idx, child_idx)
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_bone_stats(pts: np.ndarray) -> dict:
    """pts: (F, 17, 3). Returns {bone_name: {mean, std, cv}}."""
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


def nan_pct(pts: np.ndarray) -> float:
    """% of joint-frames that are NaN (all 3 coords NaN → joint is NaN)."""
    F, K, _ = pts.shape
    nan_joints = np.all(np.isnan(pts), axis=2)  # (F, K)
    return 100.0 * nan_joints.sum() / (F * K)


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray) -> float:
    """RC-MPJPE (root-centred, joints 11+12 as root). Returns overall mean mm."""
    errs = []
    for f in range(len(gt_valid)):
        if not gt_valid[f]:
            continue
        p = pred[f]   # (17, 3)
        g = gt[f]     # (17, 3)
        valid_j = np.isfinite(p).all(1) & np.isfinite(g).all(1)
        if valid_j.sum() < 3:
            continue
        root_p = (p[11] + p[12]) / 2.0
        root_g = (g[11] + g[12]) / 2.0
        p_rc   = p - root_p
        g_rc   = g - root_g
        diff   = np.where(valid_j[:, None], p_rc - g_rc, np.nan)
        err    = np.nanmean(np.linalg.norm(diff, axis=1)) * 10.0  # cm→mm
        errs.append(err)
    return float(np.nanmean(errs)) if errs else float("nan")


def mean_jitter(pts: np.ndarray) -> float:
    """Mean frame-to-frame Euclidean displacement per joint (cm)."""
    delta = np.linalg.norm(np.diff(pts, axis=0), axis=2)  # (F-1, K)
    return float(np.nanmean(delta))


def lag_check(raw: np.ndarray, cleaned: np.ndarray, joint_idx: int = 9,
              max_lag: int = 10) -> float:
    """
    Cross-correlate the velocity magnitudes of raw vs cleaned for one joint.
    Returns the lag (in frames) of the cleaned signal relative to raw.
    Positive = cleaned lags behind raw.
    """
    vel_raw     = np.linalg.norm(np.diff(raw[:, joint_idx], axis=0), axis=1)
    vel_cleaned = np.linalg.norm(np.diff(cleaned[:, joint_idx], axis=0), axis=1)

    # Normalise
    def norm(x):
        s = x.std()
        return (x - x.mean()) / (s + 1e-9)

    r = np.correlate(norm(vel_cleaned), norm(vel_raw), mode="full")
    lags = np.arange(-(len(vel_raw) - 1), len(vel_raw))
    best_lag = int(lags[np.argmax(r)])
    # Clamp to search window
    return float(np.clip(best_lag, -max_lag, max_lag))


# ── 3D skeleton animation (raw / cleaned / GT side-by-side) ──────────────────

def make_3panel_anim(raw, cleaned, gt, gt_valid, out_path: Path, n_frames=60, fps=10):
    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(131, projection="3d"),
           fig.add_subplot(132, projection="3d"),
           fig.add_subplot(133, projection="3d")]
    titles = ["Raw", "Cleaned (gap+smooth+norm)", "Panoptic GT"]
    skels  = [raw, cleaned, gt]

    # Calculate a common center from the GT data
    valid_gt = gt[gt_valid]
    if len(valid_gt) > 0:
        # mid-hip is the root
        root_pts = (valid_gt[:, 11] + valid_gt[:, 12]) / 2.0
        center = np.nanmedian(root_pts, axis=0)
    else:
        center = np.array([0, 0, 0])

    def _plot_frame(ax, pts, title, f):
        ax.cla()
        ax.set_title(f"{title}\nf={f}", fontsize=8)
        c = center
        r = 100
        ax.set_xlim(c[0]-r, c[0]+r)
        ax.set_ylim(c[1]-r, c[1]+r)
        ax.set_zlim(c[2]-r, c[2]+r)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        for i, j in DRAW_EDGES:
            if np.all(np.isfinite(pts[f, [i, j]])):
                ax.plot(*pts[f, [i, j]].T, "b-", lw=1)
        valid = np.all(np.isfinite(pts[f]), axis=1)
        ax.scatter(*pts[f, valid].T, c="r", s=10, zorder=5)

    def update(frame_idx):
        for ax, skel, title in zip(axs, skels, titles):
            _plot_frame(ax, skel, title, frame_idx)
        return []

    frames = [i for i in range(min(n_frames, len(raw))) if gt_valid[i]][:n_frames]
    if not frames:
        frames = list(range(min(n_frames, len(raw))))

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=1000/fps)
    ani.save(str(out_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Animation saved: {out_path}")


# ── Per-joint time-series plot ────────────────────────────────────────────────

def make_timeseries_plot(raw, filled, cleaned, gap_log, joint_idxs: list[int],
                          joint_labels: list[str], out_path: Path):
    """X/Y/Z time series for selected joints: raw dots + cleaned line + gap highlights."""
    n_joints = len(joint_idxs)
    fig, axes = plt.subplots(n_joints, 3, figsize=(18, 4 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    frames = np.arange(raw.shape[0])
    colors = {"X": "#E74C3C", "Y": "#2ECC71", "Z": "#3498DB"}
    dims   = ["X", "Y", "Z"]

    # Build per-joint gap index for shading
    gap_idx: dict[int, list] = {}
    for rec in gap_log:
        ji = rec["joint_idx"]
        gap_idx.setdefault(ji, []).append(rec)

    for row, (ji, jlabel) in enumerate(zip(joint_idxs, joint_labels)):
        for col, (dim, dname) in enumerate(zip(range(3), dims)):
            ax = axes[row][col]
            ax.set_title(f"{jlabel} — {dname}", fontsize=9)

            # Shade filled gaps
            for rec in gap_idx.get(ji, []):
                color = "#FF6B6B" if rec["long_gap"] else "#FFD93D"
                ax.axvspan(rec["start_frame"], rec["end_frame"], alpha=0.25,
                           color=color, label="long gap" if rec["long_gap"] else "gap")

            # Raw dots (sparse — show actual NaN structure)
            r = raw[:, ji, dim]
            ax.scatter(frames[np.isfinite(r)], r[np.isfinite(r)],
                       s=4, color=colors[dname], alpha=0.4, label="raw", zorder=2)

            # Cleaned line
            c = cleaned[:, ji, dim]
            ax.plot(frames, c, color=colors[dname], lw=1.0, label="cleaned", zorder=3)

            ax.set_ylabel("cm")
            if row == 0 and col == 0:
                # Add legend once
                from matplotlib.patches import Patch
                handles = [
                    Patch(color="#FF6B6B", alpha=0.4, label=f"long gap (>{LONG_GAP_THRESHOLD}f)"),
                    Patch(color="#FFD93D", alpha=0.4, label="short gap"),
                    plt.Line2D([0], [0], color=colors[dname], lw=1.5, label="cleaned"),
                ]
                ax.legend(handles=handles, fontsize=7, loc="upper right")

    axes[-1][1].set_xlabel("Frame")
    fig.suptitle("Stage 5 — Per-Joint Time Series: Raw vs Cleaned", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Time-series plot saved: {out_path}")


# ── Report table ──────────────────────────────────────────────────────────────

def print_row(label, before, after, threshold, pass_fail, notes=""):
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)
    status = "PASS" if pass_fail else "FAIL"
    print(f"{label:<45} {fmt(before):>10}  ->  {fmt(after):>10}   {threshold:<12} {status}  {notes}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading Stage-4 raw outputs...")
    raw_pts  = np.load(S4_DIR / "pts3d.npy")     # (F, 17, 3) cm
    gt_kpts  = np.load(S4_DIR / "gt_kpts.npy")   # (F, 17, 3) cm
    gt_valid = np.load(S4_DIR / "gt_valid.npy")  # (F,) bool
    F = raw_pts.shape[0]
    print(f"  Shape: {raw_pts.shape}, GT frames valid: {gt_valid.sum()}/{F}")

    # ── BEFORE metrics ────────────────────────────────────────────────────────
    print("\nComputing BEFORE metrics...")
    before_mpjpe = compute_mpjpe(raw_pts, gt_kpts, gt_valid)
    before_bones = compute_bone_stats(raw_pts)
    before_nan   = nan_pct(raw_pts)
    before_jitter = mean_jitter(raw_pts)
    before_cv_fail = [b for b, s in before_bones.items() if s["cv"] > 0.05]

    print(f"  RC-MPJPE:  {before_mpjpe:.1f} mm")
    print(f"  NaN joints: {before_nan:.1f}%")
    print(f"  Jitter:    {before_jitter:.4f} cm/frame")
    print(f"  Bones CV>5%: {before_cv_fail}")

    # ── Step 1: Gap fill ─────────────────────────────────────────────────────
    print("\nStep 1: Filling gaps...")
    filled, gap_log, long_gap_counts = fill_gaps_with_logging(
        raw_pts, JOINT_NAMES, fps=FPS
    )

    total_gaps  = len(gap_log)
    long_gaps   = [g for g in gap_log if g["long_gap"]]
    max_gap_len = max((g["gap_length"] for g in gap_log), default=0)
    after_nan   = nan_pct(filled)

    print(f"  Total gaps found: {total_gaps}")
    print(f"  Long gaps (>{LONG_GAP_THRESHOLD}f): {len(long_gaps)}")
    print(f"  Max gap length: {max_gap_len} frames")
    print(f"  NaN after fill: {after_nan:.1f}%")

    # Identify joints with frequent long gaps (upstream signal)
    flagged_joints = {j: c for j, c in long_gap_counts.items() if c > 0}
    if flagged_joints:
        print(f"  [UPSTREAM FLAG] joints with frequent long gaps: {flagged_joints}")

    # Save gap log as structured JSON
    gap_log_path = OUT_DIR / "gap_log.json"
    with open(gap_log_path, "w") as f:
        json.dump({
            "long_gap_threshold": LONG_GAP_THRESHOLD,
            "total_gaps": total_gaps,
            "total_long_gaps": len(long_gaps),
            "max_gap_length": max_gap_len,
            "per_joint_long_gap_counts": long_gap_counts,
            "flagged_joints_upstream": flagged_joints,
            "gap_log": gap_log,
        }, f, indent=2)
    print(f"  Gap log saved: {gap_log_path}")

    # ── Step 2: One-Euro smooth ───────────────────────────────────────────────
    print("\nStep 2: One-Euro smoothing...")
    smoothed = filter_skeleton_one_euro(filled, fps=FPS, min_cutoff=1.0, beta=0.007)

    # Lag check (left_wrist = joint 9, high-motion joint)
    lag_wrist = lag_check(filled, smoothed, joint_idx=9)
    lag_knee  = lag_check(filled, smoothed, joint_idx=13)
    print(f"  Lag (left_wrist): {lag_wrist:.1f} frames")
    print(f"  Lag (left_knee):  {lag_knee:.1f} frames")

    # ── Step 3: Bone-length normalization ────────────────────────────────────
    print("\nStep 3: Median bone-length normalization...")
    cleaned = normalize_bone_lengths_median(smoothed, BONES)

    # ── AFTER metrics ─────────────────────────────────────────────────────────
    print("\nComputing AFTER metrics...")
    after_mpjpe  = compute_mpjpe(cleaned, gt_kpts, gt_valid)
    after_bones  = compute_bone_stats(cleaned)
    after_jitter = mean_jitter(cleaned)
    after_cv_fail = [b for b, s in after_bones.items() if s["cv"] > 0.05]

    mpjpe_delta_pct = 100.0 * (after_mpjpe - before_mpjpe) / (before_mpjpe + 1e-9)

    print(f"  RC-MPJPE:  {after_mpjpe:.1f} mm  (delta {mpjpe_delta_pct:+.1f}%)")
    print(f"  Jitter:    {after_jitter:.4f} cm/frame")
    print(f"  Bones CV>5%: {after_cv_fail}")

    # ── Visual outputs ────────────────────────────────────────────────────────
    print("\nGenerating visuals...")

    # 3-panel animation
    make_3panel_anim(
        raw_pts, cleaned, gt_kpts, gt_valid,
        out_path=OUT_DIR / "skeleton_3panel.gif",
        n_frames=60, fps=10,
    )

    # Per-joint time series: left_wrist (9) and left_knee (13)
    make_timeseries_plot(
        raw=raw_pts, filled=filled, cleaned=cleaned,
        gap_log=gap_log,
        joint_idxs=[9, 13],
        joint_labels=["left_wrist", "left_knee"],
        out_path=OUT_DIR / "timeseries_wrist_knee.png",
    )

    # ── Report table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Metric':<45} {'BEFORE':>10}     {'AFTER':>10}   {'Threshold':<12} {'Pass/Fail'}")
    print("-" * 90)

    print_row(
        "RC-MPJPE (mm)", before_mpjpe, after_mpjpe,
        "<5% increase",
        mpjpe_delta_pct <= 5.0,
        f"delta {mpjpe_delta_pct:+.1f}%",
    )
    print_row(
        "NaN joint-frames (%)", before_nan, after_nan,
        "~0%",
        after_nan < 1.0,
    )
    print_row(
        "Bones with CV>5%", len(before_cv_fail), len(after_cv_fail),
        "0",
        len(after_cv_fail) == 0,
        f"before={before_cv_fail[:3]}...",
    )
    print_row(
        "Jitter (cm/frame mean)", before_jitter, after_jitter,
        "lower is better",
        after_jitter < before_jitter,
    )
    print_row(
        f"Max gap filled (frames)", None, max_gap_len,
        f"flag if >{LONG_GAP_THRESHOLD}",
        max_gap_len <= LONG_GAP_THRESHOLD,
    )
    print_row(
        "Long gaps total", None, len(long_gaps),
        "0 preferred",
        len(long_gaps) == 0,
        str(flagged_joints) if flagged_joints else "",
    )
    print_row(
        "One-Euro lag — left_wrist (frames)", None, lag_wrist,
        "≤2",
        abs(lag_wrist) <= 2,
    )
    print_row(
        "One-Euro lag — left_knee (frames)", None, lag_knee,
        "≤2",
        abs(lag_knee) <= 2,
    )

    # Per-bone CV table
    print(f"\n{'Bone':<20} {'CV before':>12} {'CV after':>12} {'Pass/Fail'}")
    print("-" * 50)
    for bone in BONES:
        cv_b = before_bones[bone]["cv"]
        cv_a = after_bones[bone]["cv"]
        pf   = "PASS" if cv_a <= 0.05 else "FAIL"
        print(f"  {bone:<18} {cv_b:>12.4f} {cv_a:>12.4f}   {pf}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics = {
        "stage": 5,
        "before": {
            "rc_mpjpe_mm": round(before_mpjpe, 2),
            "nan_pct":     round(before_nan, 2),
            "jitter_cm":   round(before_jitter, 4),
            "bones_cv_failing": before_cv_fail,
        },
        "after": {
            "rc_mpjpe_mm":     round(after_mpjpe, 2),
            "mpjpe_delta_pct": round(mpjpe_delta_pct, 2),
            "nan_pct":         round(after_nan, 2),
            "jitter_cm":       round(after_jitter, 4),
            "bones_cv_failing": after_cv_fail,
            "lag_wrist_frames": lag_wrist,
            "lag_knee_frames":  lag_knee,
        },
        "gap_fill": {
            "total_gaps":     total_gaps,
            "long_gaps":      len(long_gaps),
            "max_gap_frames": max_gap_len,
            "flagged_joints": flagged_joints,
        },
        "per_bone_cv": {
            b: {"before": before_bones[b]["cv"], "after": after_bones[b]["cv"]}
            for b in BONES
        },
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {OUT_DIR / 'metrics.json'}")
    print(f"Gap log saved: {gap_log_path}")


if __name__ == "__main__":
    main()
