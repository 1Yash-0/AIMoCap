"""Stage 4 Follow-up: Geometry Verification and Synthetic Sanity Check."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics
from aimocap.math.coords import internal_to_opencv

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_DIR    = ROOT / "data" / "panoptic" / "171204_pose1"
CALIB_JSON = SEQ_DIR / "calibration_171204_pose1.json"
GT_DIR     = SEQ_DIR / "hdPose3d_stage1_coco19"
OUT_DIR    = ROOT / "outputs" / "stage4_followup"

CAM_NAMES  = ["00_00", "00_01", "00_02"]
START_FRAME = 149
N_FRAMES   = 300

# Mapping as used in stage4_triangulation_audit
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def get_cam_pos(R, t):
    """pos = -R.T @ t"""
    return -R.T @ t

def get_angle(pos1, pos2, center):
    v1 = pos1 - center
    v2 = pos2 - center
    v1_n = v1 / np.linalg.norm(v1)
    v2_n = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_n.T, v2_n), -1.0, 1.0)))[0,0]

def load_gt_sequence(gt_dir: Path, start: int, n: int) -> tuple[np.ndarray, np.ndarray]:
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
        joints19 = np.array(bodies[0]["joints19"], dtype=np.float32).reshape(19, 4)
        for c17, p19 in enumerate(COCO17_TO_PAN19):
            gt_kpts[i, c17] = joints19[p19, :3]
        gt_valid[i] = True
    return gt_kpts, gt_valid

def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Rotation+Translation Procrustes (no scale)"""
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
    t = gt.mean(0) - (R @ pred.mean(0))
    return (R @ pred.T).T + t

def compute_mpjpe(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray):
    """Root-centered and PA-MPJPE in mm."""
    errs_rc = []
    errs_pa = []
    for f in np.where(valid_mask)[0]:
        p = pred[f]
        g = gt[f]
        valid_j = np.isfinite(p).all(1) & np.isfinite(g).all(1)
        if valid_j.sum() < 4:
            continue
        root_p = (p[11] + p[12]) / 2.0
        root_g = (g[11] + g[12]) / 2.0
        p_rc = p - root_p
        g_rc = g - root_g

        diff_rc = np.where(valid_j[:, None], p_rc - g_rc, np.nan)
        err_rc  = np.linalg.norm(diff_rc, axis=1) * 10.0

        p_sel = p_rc[valid_j]
        g_sel = g_rc[valid_j]
        p_al  = procrustes_align(p_sel, g_sel)
        p_aligned = p_rc.copy()
        p_aligned[valid_j] = p_al
        diff_pa = np.where(valid_j[:, None], p_aligned - g_rc, np.nan)
        err_pa  = np.linalg.norm(diff_pa, axis=1) * 10.0

        errs_rc.append(err_rc)
        errs_pa.append(err_pa)

    if not errs_rc:
        return float('nan'), float('nan')
    return float(np.nanmean(errs_rc)), float(np.nanmean(errs_pa))

def check_a():
    print("=== Check A: Camera Geometry Frustums ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calib = load_calibration(CALIB_JSON)
    gt_kpts, gt_valid = load_gt_sequence(GT_DIR, START_FRAME, N_FRAMES)
    
    # Subject mean root (hips)
    valid_gt = gt_kpts[gt_valid]
    roots = (valid_gt[:, 11, :] + valid_gt[:, 12, :]) / 2.0
    mean_root = roots.mean(0).reshape(3, 1)

    positions = {}
    directions = {}
    for cn in CAM_NAMES:
        c = calib[cn]
        t = c.t.reshape(3, 1)
        positions[cn] = get_cam_pos(c.R, t)
        # Z axis in camera space is [0,0,1], in world space it's R.T @ [0,0,1]
        directions[cn] = c.R.T @ np.array([[0], [0], [1]])

    a1 = get_angle(positions["00_00"], positions["00_01"], mean_root)
    a2 = get_angle(positions["00_01"], positions["00_02"], mean_root)
    a3 = get_angle(positions["00_02"], positions["00_00"], mean_root)
    
    print(f"Angular separation (relative to subject):")
    print(f"  00_00 vs 00_01: {a1:.1f}°")
    print(f"  00_01 vs 00_02: {a2:.1f}°")
    print(f"  00_02 vs 00_00: {a3:.1f}°")

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(mean_root[0], mean_root[2], mean_root[1], c='red', s=100, label='Subject Mean Root')
    colors = ['blue', 'green', 'magenta']
    for idx, cn in enumerate(CAM_NAMES):
        p = positions[cn]
        d = directions[cn]
        ax.scatter(p[0], p[2], p[1], c=colors[idx], s=50, label=f'Cam {cn}')
        # Draw frustum line
        end_pt = p + d * 150 # length 150cm
        ax.plot([p[0,0], end_pt[0,0]], [p[2,0], end_pt[2,0]], [p[1,0], end_pt[1,0]], c=colors[idx], linestyle='-')
        
    ax.set_xlabel('X (cm)'); ax.set_ylabel('Z (cm)'); ax.set_zlabel('Y (cm)')
    ax.set_title('Camera Geometry Check A (World Space)')
    ax.legend()
    plt.savefig(OUT_DIR / "frustums_original.png")
    plt.close()
    print(f"Saved {OUT_DIR / 'frustums_original.png'}")

def check_b():
    print("\n=== Check B: Synthetic Sanity Test ===")
    calib = load_calibration(CALIB_JSON)
    gt_kpts, gt_valid = load_gt_sequence(GT_DIR, START_FRAME, N_FRAMES)
    
    K_list = []
    extrinsics = []
    P_list = []
    for cn in CAM_NAMES:
        c = calib[cn]
        K = c.K.astype(np.float64)
        t = c.t.reshape(3, 1).astype(np.float64)
        R = c.R.astype(np.float64)
        K_list.append(K)
        extrinsics.append((R, t))
        P_list.append(K @ np.hstack((R, t)))

    # Project GT points to create synthetic 2D keypoints
    synth_kpts2d = np.zeros((N_FRAMES, 3, 17, 2), dtype=np.float64)
    synth_scores = np.zeros((N_FRAMES, 3, 17), dtype=np.float64)

    for f in range(N_FRAMES):
        if not gt_valid[f]:
            continue
        for j in range(17):
            pt = gt_kpts[f, j]
            if not np.isfinite(pt).all():
                continue
            X = np.append(pt, 1.0)
            for c_idx in range(3):
                p = P_list[c_idx] @ X
                if p[2] > 0: # In front of camera
                    px, py = p[0]/p[2], p[1]/p[2]
                    synth_kpts2d[f, c_idx, j] = [px, py]
                    synth_scores[f, c_idx, j] = 1.0

    print("Triangulating perfect synthetic 2D points...")
    diag = triangulate_sequence_with_diagnostics(
        synth_kpts2d, synth_scores, K_list, extrinsics, min_conf=0.9
    )
    pts3d_cv = internal_to_opencv(diag.points3d)
    
    rc, pa = compute_mpjpe(pts3d_cv, gt_kpts, gt_valid)
    print(f"Synthetic RC-MPJPE: {rc:.2f} mm")
    print(f"Synthetic PA-MPJPE: {pa:.2f} mm")
    if rc < 5.0:
        print("-> Synthetic sanity test PASSED.")
    else:
        print("-> Synthetic sanity test FAILED. There is a bug in the code path.")

if __name__ == "__main__":
    check_a()
    check_b()
