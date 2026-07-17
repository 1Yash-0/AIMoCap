"""Stage 4 — Triangulation Audit.

Uses existing triangulate_sequence_with_diagnostics (no new math written).
Calibration: Panoptic GT (do NOT use our own calibration — Stage 1 deferred).
GT comparison: Panoptic body3DScene JSON files (COCO-19, 19 joints).

Frame mapping:
  Stage-3 NPZ was collected starting at t=5s in hd_29.97fps video.
  start_frame = int(29.97 * 5.0) = 149.
  NPZ frame i  <->  GT file body3DScene_{149+i:08d}.json

Outputs -> outputs/stage4_triangulation/
  metrics.json
  stage4_report_table.txt
  skeleton_animation.gif      (pred vs GT, 50 frames at 10fps)
  cam{id}_reproj.mp4          (per-camera reprojection overlay)

Usage:
    python scripts/stage4_triangulation_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
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

import argparse

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_DIR    = ROOT / "data" / "panoptic" / "171204_pose1"
CALIB_JSON = SEQ_DIR / "calibration_171204_pose1.json"
GT_DIR     = SEQ_DIR / "hdPose3d_stage1_coco19"
VIDEO_DIR  = SEQ_DIR / "hdVideos"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs="+", default=["00_00", "00_01", "00_02"])
    parser.add_argument("--npz", default="outputs/stage3_pose/kpts.npz")
    parser.add_argument("--outdir", default="outputs/stage4_triangulation")
    parser.add_argument("--min-conf", type=float, default=0.65)
    return parser.parse_args()

args = parse_args()

NPZ_PATH   = ROOT / args.npz
OUT_DIR    = ROOT / args.outdir
CAM_NAMES  = args.cams
VIDEO_FILES = [VIDEO_DIR / f"hd_{c}.mp4" for c in CAM_NAMES]
MIN_CONF   = args.min_conf

GT_FPS     = 29.97
START_SEC  = 5.0
START_FRAME = int(GT_FPS * START_SEC)   # 149
N_FRAMES   = 300

# COCO-17 body indices (0-16) only — these map to Panoptic-19
BODY17 = list(range(17))
BODY17_NAMES = [KEYPOINT_NAMES_133[i] for i in BODY17]

# COCO-17 → Panoptic-19 joint index mapping
# Panoptic-19: 0=Neck, 1=Nose, 2=BodyCenter, 3=lShoulder, 4=lElbow, 5=lWrist,
#              6=lHip, 7=lKnee, 8=lAnkle, 9=rShoulder, 10=rElbow, 11=rWrist,
#              12=rHip, 13=rKnee, 14=rAnkle, 15=lEye, 16=lEar, 17=rEye, 18=rEar
COCO17_TO_PAN19 = [
    1,   # 0 nose      → Nose(1)
    15,  # 1 left_eye  → lEye(15)
    17,  # 2 right_eye → rEye(17)
    16,  # 3 left_ear  → lEar(16)
    18,  # 4 right_ear → rEar(18)
    3,   # 5 l_shoulder→ lShoulder(3)
    9,   # 6 r_shoulder→ rShoulder(9)
    4,   # 7 l_elbow   → lElbow(4)
    10,  # 8 r_elbow   → rElbow(10)
    5,   # 9 l_wrist   → lWrist(5)
    11,  # 10 r_wrist  → rWrist(11)
    6,   # 11 l_hip    → lHip(6)
    12,  # 12 r_hip    → rHip(12)
    7,   # 13 l_knee   → lKnee(7)
    13,  # 14 r_knee   → rKnee(13)
    8,   # 15 l_ankle  → lAnkle(8)
    14,  # 16 r_ankle  → rAnkle(14)
]

# Bones for body-length analysis (COCO-17 indices)
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

# Skeleton edges to draw
DRAW_EDGES = list(BONES.values()) + [(0, 5), (0, 6), (5, 6), (11, 12)]


# ── Procrustes alignment ──────────────────────────────────────────────────────
def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Rotation+Translation Procrustes (no scale) — standard PA-MPJPE definition.
    Guarantees PA-MPJPE <= MPJPE. Both inputs: (N, 3).
    Falls back to identity alignment if SVD fails or input is degenerate."""
    if len(pred) < 3:
        return pred
    pred_c = pred - pred.mean(0)
    gt_c   = gt   - gt.mean(0)
    H = pred_c.T @ gt_c
    if not np.isfinite(H).all() or np.all(np.abs(H) < 1e-12):
        return pred
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return pred
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    # rotation only — no scale — then shift to GT centroid
    t = gt.mean(0) - (R @ pred.mean(0))
    return (R @ pred.T).T + t


# ── GT loading ────────────────────────────────────────────────────────────────
def load_gt_sequence(gt_dir: Path, start: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Load n GT frames starting at 'start'. Returns (n,17,3) array and (n,) valid mask.
    GT is in cm, Panoptic-world space. Maps Panoptic-19 → COCO-17."""
    gt_kpts  = np.full((n, 17, 3), np.nan, dtype=np.float32)
    gt_valid = np.zeros(n, dtype=bool)

    for i in range(n):
        fn = gt_dir / f"body3DScene_{start+i:08d}.json"
        if not fn.exists():
            continue
        with open(fn) as f:
            data = json.load(f)
        bodies = data.get("bodies", [])
        if not bodies:
            continue
        # Use first body (primary subject)
        joints19 = np.array(bodies[0]["joints19"], dtype=np.float32).reshape(19, 4)
        # joints19[:, 0:3] = xyz in cm; [:, 3] = confidence
        # Map to COCO-17
        for c17, p19 in enumerate(COCO17_TO_PAN19):
            gt_kpts[i, c17] = joints19[p19, :3]
        gt_valid[i] = True

    return gt_kpts, gt_valid


# ── Per-frame MPJPE ───────────────────────────────────────────────────────────
def compute_mpjpe(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray):
    """Root-centered and PA-MPJPE in mm.
    pred, gt: (F, 17, 3) in cm. valid_mask: (F,).
    Returns per_joint_mpjpe (17,), per_joint_pa_mpjpe (17,), per_frame_mpjpe (F,).
    """
    root = 11  # left_hip as root (common convention)
    errs_rc = []
    errs_pa = []

    for f in np.where(valid_mask)[0]:
        p = pred[f]  # (17,3)
        g = gt[f]    # (17,3)
        # both must be finite
        valid_j = np.isfinite(p).all(1) & np.isfinite(g).all(1)
        if valid_j.sum() < 4:
            errs_rc.append(np.full(17, np.nan))
            errs_pa.append(np.full(17, np.nan))
            continue

        # root-center on hip midpoint
        root_p = (p[11] + p[12]) / 2.0
        root_g = (g[11] + g[12]) / 2.0
        p_rc = p - root_p
        g_rc = g - root_g

        # raw error (root-centered)
        diff_rc = np.where(valid_j[:, None], p_rc - g_rc, np.nan)
        err_rc  = np.linalg.norm(diff_rc, axis=1) * 10.0   # cm → mm

        # Procrustes-aligned (use valid joints only for alignment)
        p_sel = p_rc[valid_j]
        g_sel = g_rc[valid_j]
        p_al  = procrustes_align(p_sel, g_sel)
        # reconstruct full aligned array
        p_aligned = p_rc.copy()
        p_aligned[valid_j] = p_al
        diff_pa = np.where(valid_j[:, None], p_aligned - g_rc, np.nan)
        err_pa  = np.linalg.norm(diff_pa, axis=1) * 10.0

        errs_rc.append(err_rc)
        errs_pa.append(err_pa)

    if not errs_rc:
        return np.full(17, np.nan), np.full(17, np.nan), np.array([])

    errs_rc = np.array(errs_rc)   # (F_valid, 17)
    errs_pa = np.array(errs_pa)
    return (
        np.nanmean(errs_rc, axis=0),  # (17,)
        np.nanmean(errs_pa, axis=0),  # (17,)
        np.nanmean(errs_rc, axis=1),  # (F_valid,)  — per-frame MPJPE
    )


# ── Bone lengths ──────────────────────────────────────────────────────────────
def compute_bone_stats(pts3d: np.ndarray) -> dict[str, dict]:
    """pts3d: (F, 17, 3) in cm. Returns per-bone {mean, std, cv} where cv=std/mean."""
    stats = {}
    for name, (i, j) in BONES.items():
        lens = np.linalg.norm(pts3d[:, i] - pts3d[:, j], axis=1)   # (F,)
        finite = lens[np.isfinite(lens)]
        if len(finite) == 0:
            stats[name] = {"mean_cm": float("nan"), "std_cm": float("nan"), "cv": float("nan")}
        else:
            mu  = float(np.mean(finite))
            sig = float(np.std(finite))
            stats[name] = {"mean_cm": round(mu, 2), "std_cm": round(sig, 2),
                           "cv": round(sig / (mu + 1e-6), 4)}
    return stats


# ── 3D skeleton animation ─────────────────────────────────────────────────────
def make_skeleton_animation(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    out_gif: Path,
    n_animate: int = 50,
    step: int = 3,
) -> None:
    """Animate predicted vs GT skeleton, every `step` frames, n_animate total."""
    frame_indices = [i for i in range(0, N_FRAMES, step) if valid_mask[i]][:n_animate]
    if not frame_indices:
        print("No valid GT frames for animation.")
        return

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")

    def _draw(f_idx: int):
        ax.cla()
        p = pred[f_idx]  # (17,3)
        g = gt[f_idx]    # (17,3)
        # root-center both on hip midpoint
        rp = (p[11] + p[12]) / 2.0
        rg = (g[11] + g[12]) / 2.0
        p = p - rp
        g = g - rg

        for i, j in DRAW_EDGES:
            if np.isfinite(p[i]).all() and np.isfinite(p[j]).all():
                ax.plot([p[i,0], p[j,0]], [p[i,2], p[j,2]], [p[i,1], p[j,1]],
                        "b-", lw=1.5, alpha=0.8)
            if np.isfinite(g[i]).all() and np.isfinite(g[j]).all():
                ax.plot([g[i,0], g[j,0]], [g[i,2], g[j,2]], [g[i,1], g[j,1]],
                        "r-", lw=1.5, alpha=0.6)

        valid_p = np.isfinite(p).all(1)
        valid_g = np.isfinite(g).all(1)
        ax.scatter(p[valid_p,0], p[valid_p,2], p[valid_p,1], c="blue", s=20, label="Pred")
        ax.scatter(g[valid_g,0], g[valid_g,2], g[valid_g,1], c="red",  s=20, alpha=0.6, label="GT")

        ax.set_xlabel("X (cm)"); ax.set_ylabel("Z (cm)"); ax.set_zlabel("Y (cm)")
        ax.set_title(f"Frame {f_idx}  (blue=pred, red=GT)")
        ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_zlim(-120, 120)
        ax.legend(loc="upper right", fontsize=8)

    ani = animation.FuncAnimation(fig, _draw, frames=frame_indices, interval=100)
    ani.save(str(out_gif), writer="pillow", fps=10)
    plt.close(fig)
    print(f"Animation saved: {out_gif}")


# ── Per-camera reprojection overlay clip ──────────────────────────────────────
def make_reproj_clip(
    cam_name: str,
    video_path: Path,
    pts3d_cv: np.ndarray,     # (F, 17, 3) in Panoptic world space (OpenCV convention for projection)
    kpts2d: np.ndarray,       # (F, 17, 2) Stage-3 2D detections
    P: np.ndarray,            # (3, 4) projection matrix
    out_mp4: Path,
) -> None:
    """Draw Stage-3 2D (green) and reprojected 3D (magenta) per frame."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for f in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break

        # Draw Stage-3 2D detections (green)
        for kp in kpts2d[f]:
            if np.isfinite(kp).all():
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 255, 0), -1)

        # Reproject 3D → 2D
        for k in range(17):
            pt = pts3d_cv[f, k]
            if not np.isfinite(pt).all():
                continue
            X = np.append(pt, 1.0)
            p = P @ X
            if abs(p[2]) < 1e-9:
                continue
            px, py = int(p[0]/p[2]), int(p[1]/p[2])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px, py), 4, (255, 0, 255), -1)

        # Draw skeleton edges (reprojected, magenta)
        for i, j in DRAW_EDGES:
            pt_i, pt_j = pts3d_cv[f, i], pts3d_cv[f, j]
            if not (np.isfinite(pt_i).all() and np.isfinite(pt_j).all()):
                continue
            def proj(pt):
                X = np.append(pt, 1.0)
                p = P @ X
                if abs(p[2]) < 1e-9: return None
                return int(p[0]/p[2]), int(p[1]/p[2])
            pi, pj = proj(pt_i), proj(pt_j)
            if pi and pj:
                if (0<=pi[0]<w and 0<=pi[1]<h and 0<=pj[0]<w and 0<=pj[1]<h):
                    cv2.line(frame, pi, pj, (255, 0, 255), 1)

        cv2.putText(frame, f"{cam_name} f={f} | green=Stage3 magenta=reproj3D",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Reprojection clip: {out_mp4}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load calibration
    print("Loading Panoptic calibration...")
    calib = load_panoptic_calib(CALIB_JSON)
    K_list = []
    extrinsics = []
    for cn in CAM_NAMES:
        c = calib[cn]
        K_list.append(c.K.astype(np.float64))
        # t from panoptic loader is already (3,1) shaped (list of 3 1-element lists)
        t = c.t.reshape(3, 1).astype(np.float64)
        extrinsics.append((c.R.astype(np.float64), t))
    P_list = [K_list[i] @ np.hstack(extrinsics[i]) for i in range(3)]
    print(f"  Cameras: {CAM_NAMES}")

    # 2. Load Stage-3 NPZ
    print("Loading Stage-3 NPZ...")
    data = np.load(NPZ_PATH, allow_pickle=True)
    kpts_all   = data["keypoints"].astype(np.float64)   # (300, 3, 133, 2)
    scores_all = data["scores"].astype(np.float64)       # (300, 3, 133)

    # Use only body-17 joints
    kpts   = kpts_all[:, :, :17, :]    # (300, 3, 17, 2)
    scores = scores_all[:, :, :17]     # (300, 3, 17)
    print(f"  Keypoints: {kpts.shape}, Scores: {scores.shape}")

    # 3. Triangulate
    print("Triangulating...")
    diag = triangulate_sequence_with_diagnostics(
        kpts, scores, K_list, extrinsics, min_conf=MIN_CONF,
    )
    # Engine returns Y-up internal space → convert back to raw world space for GT comparison
    pts3d_internal = diag.points3d                    # (F, 17, 3)
    pts3d_cv = internal_to_opencv(pts3d_internal)     # (F, 17, 3)  Panoptic world space (cm)
    reproj_err = diag.reprojection_error_px           # (F, 17, 3)  nan where not observed
    num_inliers = diag.num_inliers                    # (F, 17)

    finite_count = np.sum(np.isfinite(pts3d_cv).all(-1))
    print(f"  Triangulated {finite_count}/{N_FRAMES*17} joint-frames successfully.")

    # 4. Load GT
    print(f"Loading GT from frame {START_FRAME} to {START_FRAME+N_FRAMES-1}...")
    gt_kpts, gt_valid = load_gt_sequence(GT_DIR, START_FRAME, N_FRAMES)
    print(f"  GT frames with data: {gt_valid.sum()}/{N_FRAMES}")

    # 5. Coverage: frames where any joint has <2 inliers
    BODY_JOINTS = list(range(5, 17))   # shoulders → ankles (exclude nose/eyes/ears)
    lt2_per_joint      = np.sum(num_inliers < 2, axis=0)   # (17,)
    any_lt2_frames     = int(np.any(num_inliers < 2, axis=1).sum())
    body_lt2_frames    = int(np.any(num_inliers[:, BODY_JOINTS] < 2, axis=1).sum())
    print(f"  Frames with any joint <2-camera  (all 17): {any_lt2_frames}/{N_FRAMES}")
    print(f"  Frames with body joint <2-camera (j5-16) : {body_lt2_frames}/{N_FRAMES}")

    # 6. MPJPE
    print("Computing MPJPE...")
    per_joint_rc, per_joint_pa, per_frame_rc = compute_mpjpe(pts3d_cv, gt_kpts, gt_valid)
    overall_mpjpe    = float(np.nanmean(per_joint_rc))
    overall_pa_mpjpe = float(np.nanmean(per_joint_pa))
    threshold_2x     = 2.0 * overall_mpjpe
    bad_joints_rc = [
        {"joint": BODY17_NAMES[i], "mpjpe_mm": round(float(per_joint_rc[i]), 1)}
        for i in range(17)
        if np.isfinite(per_joint_rc[i]) and per_joint_rc[i] > threshold_2x
    ]

    print(f"  Overall MPJPE     : {overall_mpjpe:.1f} mm")
    print(f"  Overall PA-MPJPE  : {overall_pa_mpjpe:.1f} mm")
    print(f"  Joints >2x mean   : {[b['joint'] for b in bad_joints_rc]}")

    # 7. Reprojection error per camera (ignore NaN)
    reproj_per_cam = []
    for c in range(3):
        err_c = reproj_err[:, :, c]
        mean_c = float(np.nanmean(err_c))
        max_c  = float(np.nanmax(np.where(np.isfinite(err_c), err_c, -np.inf)))
        reproj_per_cam.append({"camera": CAM_NAMES[c],
                                "mean_px": round(mean_c, 2),
                                "max_px":  round(max_c,  2)})
        print(f"  Reprojection [{CAM_NAMES[c]}]: mean={mean_c:.2f}px  max={max_c:.2f}px")

    # Flag if one camera is notably worse than the others (> 2x the best)
    reproj_means = [r["mean_px"] for r in reproj_per_cam]
    best_reproj  = min(reproj_means)
    reproj_flags = [r["camera"] for r in reproj_per_cam if r["mean_px"] > 2.0 * best_reproj]

    # 8. Bone-length consistency
    print("Computing bone lengths...")
    bone_stats = compute_bone_stats(pts3d_cv)
    failing_bones = {k: v for k, v in bone_stats.items() if v["cv"] > 0.05}
    print(f"  Bones with CV>5%: {list(failing_bones.keys())}")

    # L/R symmetry
    sym_pairs = [
        ("l_upper_arm", "r_upper_arm"),
        ("l_forearm",   "r_forearm"),
        ("l_thigh",     "r_thigh"),
        ("l_shin",      "r_shin"),
    ]
    sym_report = []
    for l_bone, r_bone in sym_pairs:
        lm = bone_stats[l_bone]["mean_cm"]
        rm = bone_stats[r_bone]["mean_cm"]
        if np.isfinite(lm) and np.isfinite(rm) and rm > 0:
            ratio = round(lm / rm, 3)
        else:
            ratio = float("nan")
        sym_report.append({"left": l_bone, "right": r_bone, "L/R": ratio})

    # 9. Visual: 3D skeleton animation
    print("Generating 3D skeleton animation...")
    make_skeleton_animation(
        pts3d_cv, gt_kpts, gt_valid,
        OUT_DIR / "skeleton_animation.gif",
    )
    
    # 9.5 Save 3D points for Stage 5
    print("Saving triangulated 3D points to pts3d.npy for Stage 5...")
    np.save(OUT_DIR / "pts3d.npy", pts3d_cv)
    np.save(OUT_DIR / "gt_kpts.npy", gt_kpts)
    np.save(OUT_DIR / "gt_valid.npy", gt_valid)

    # 10. Per-camera reprojection overlay clips
    print("Generating reprojection overlay clips...")
    kpts2d_body17 = kpts_all[:, :, :17, :]  # (F, 3, 17, 2)
    for c_idx, (cn, vf) in enumerate(zip(CAM_NAMES, VIDEO_FILES)):
        if not vf.exists():
            print(f"  [SKIP] {cn}: video not found")
            continue
        make_reproj_clip(
            cam_name=cn,
            video_path=vf,
            pts3d_cv=pts3d_cv,
            kpts2d=kpts2d_body17[:, c_idx],   # (F, 17, 2)
            P=P_list[c_idx],
            out_mp4=OUT_DIR / f"cam{cn}_reproj.mp4",
        )

    # 11. Save metrics
    report = {
        "stage": 4,
        "calibration": "Panoptic GT (not our Stage-1 calibration)",
        "frames_analyzed": int(N_FRAMES),
        "gt_frames_with_data": int(gt_valid.sum()),
        "overall_mpjpe_mm": round(overall_mpjpe, 1),
        "overall_pa_mpjpe_mm": round(overall_pa_mpjpe, 1),
        "per_joint_mpjpe_mm": {BODY17_NAMES[i]: round(float(per_joint_rc[i]), 1) for i in range(17)},
        "per_joint_pa_mpjpe_mm": {BODY17_NAMES[i]: round(float(per_joint_pa[i]), 1) for i in range(17)},
        "joints_over_2x_mean": bad_joints_rc,
        "reprojection_per_camera": reproj_per_cam,
        "reproj_flagged_cameras": reproj_flags,
        "bone_length_stats": bone_stats,
        "failing_bones_cv5pct": list(failing_bones.keys()),
        "lr_symmetry": sym_report,
        "frames_any_joint_lt2_cameras": any_lt2_frames,
        "frames_body_joint_lt2_cameras": body_lt2_frames,
        "per_joint_lt2_camera_frames": {BODY17_NAMES[i]: int(lt2_per_joint[i]) for i in range(17)},
    }

    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nMetrics saved: {metrics_path}")

    # 12. Print final table
    table = "\n-- Stage 4 Report Table " + "-"*50
    table += f"\n{'Metric':<40} {'Value':>15} {'Threshold':>12} {'Pass/Fail':>10} {'Notes'}"
    table += "\n" + "-"*110

    def row(m, v, thr, pf, notes=""):
        return f"\n{m:<40} {v:>15} {thr:>12} {pf:>10}  {notes}"

    table += row("Overall MPJPE (mm)",           f"{overall_mpjpe:.1f}",     "report",   "INFO",   "Root-centred")
    table += row("Overall PA-MPJPE (mm)",         f"{overall_pa_mpjpe:.1f}",  "report",   "INFO",   "Procrustes-aligned")
    table += row("Joints >2x mean MPJPE",         str(len(bad_joints_rc)),    "0",        "PASS" if not bad_joints_rc else "FAIL",
                 str([b['joint'] for b in bad_joints_rc])[:60])
    for r in reproj_per_cam:
        flag = "(flagged)" if r["camera"] in reproj_flags else ""
        table += row(f"Reproj error [{r['camera']}] mean px",
                     f"{r['mean_px']:.2f}",  "compare",
                     "FLAG" if r["camera"] in reproj_flags else "OK", flag)
    table += row("Bones with CV>5% (bone-len)",  str(len(failing_bones)),     "0",
                 "PASS" if not failing_bones else "FAIL", str(list(failing_bones.keys()))[:60])
    for s in sym_report:
        diff_pct = abs(1.0 - s["L/R"]) * 100 if np.isfinite(s["L/R"]) else 999
        pf = "PASS" if diff_pct < 5 else "FAIL"
        table += row(f"L/R symmetry: {s['left'].split('_',1)[1]}",
                     f"{s['L/R']:.3f}", "<5% diff", pf)
    table += row("Frames any joint <2 cameras (all17)", str(any_lt2_frames), "<10%",
                 "INFO", f"{100*any_lt2_frames/N_FRAMES:.1f}% (incl. face kpts)")
    table += row("Frames body joint <2 cameras (j5-16)", str(body_lt2_frames), "<10%",
                 "PASS" if body_lt2_frames/N_FRAMES < 0.1 else "FAIL",
                 f"{100*body_lt2_frames/N_FRAMES:.1f}%")
    table += "\n"

    print(table)
    (OUT_DIR / "stage4_report_table.txt").write_text(table, encoding="utf-8")
    print(f"Report table saved: {OUT_DIR / 'stage4_report_table.txt'}")


if __name__ == "__main__":
    main()
