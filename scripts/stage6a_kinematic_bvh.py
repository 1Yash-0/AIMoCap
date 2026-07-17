"""Stage 6a -- Kinematic Solve + BVH Export (retargeting is 6b, NOT this task).

Fixes vs first attempt:
  1. Bone lengths: computed directly from bvh_pos for ALL joints (including virtual).
     Shins override with 0.98 * thigh prior (ankles are NaN in measured data).
  2. Rotation solve: built sequentially. Pelvis and spine define root/spine rotations.
     Other joints use arc rotation from parent. This ensures no multiple-children
     overwrite bugs and correct relative orientations.
  3. BVH OFFSET: defined using canonical parent-local rest directions.
     Round-trip FK then exactly matches.
  4. All MPJPE comparisons: measured frames only (non-reconstructed).
  5. Rotation smoothness check uses angular velocity matrix magnitude, not Euler deltas.

Pipeline:
    Step 0   - verify Stage 4.2 PASS (43.59mm < 55.55mm)
    Outlier  - drop reproj > 3x p95 (file not saved; noted)
    BoneLens - median dist per BVH joint-pair, measured frames only
    Ankles   - cam 00_30 ray + bone-length prior
    FK solve - sequential root→leaf, utilizing proper local resting offsets
    BVH      - export + self-consistent round-trip FK check (<1mm)
    Visuals  - 3-panel GIF, rotation time-series, sidecar JSON
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
from aimocap.math.coords import internal_to_opencv
from aimocap.pose.keypoints import KEYPOINT_NAMES_133

# ── Paths ─────────────────────────────────────────────────────────────────────
S42_DIR    = ROOT / "outputs" / "stage4_2_knee_rescue"
S3_NPZ     = ROOT / "outputs" / "stage3_check_c" / "kpts.npz"
SEQ_DIR    = ROOT / "data" / "panoptic" / "171204_pose1"
CALIB_JSON = SEQ_DIR / "calibration_171204_pose1.json"
GT_DIR     = SEQ_DIR / "hdPose3d_stage1_coco19"
OUT_DIR    = ROOT / "outputs" / "stage6a_bvh"

CAM_NAMES   = ["00_26", "00_29", "00_30"]
ANKLE_CAM   = "00_30"
FPS         = 29.97
START_FRAME = int(FPS * 5.0)   # 149
N_FRAMES    = 300
ANKLE_IDX   = [15, 16]

COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

from types import MappingProxyType

# === B-vs-C Configuration ===
# Candidate B is the explicit default configuration for ankle inference.
# See diagnosis.md for the engineering rationale.
DEFAULT_PIPELINE_CONFIG = MappingProxyType({
    "ankle_strategy": "ray_sphere_with_fk_fallback",
    "boundary_rejection_gate": False,
    "boundary_margin_px": 40,
    "image_width_px": 1920,
    "image_height_px": 1080
})

def experimental_candidate_c_config(image_width_px=1920, image_height_px=1080, boundary_margin_px=40):
    return {
        "ankle_strategy": "ray_sphere_with_fk_fallback",
        "boundary_rejection_gate": True,
        "boundary_margin_px": boundary_margin_px,
        "image_width_px": image_width_px,
        "image_height_px": image_height_px,
    }

def resolve_pipeline_config(config):
    required = {"ankle_strategy", "boundary_rejection_gate", "boundary_margin_px", "image_width_px", "image_height_px"}
    missing = required - config.keys()
    if missing:
        raise KeyError(f"Missing pipeline config keys: {sorted(missing)}")

    strategy = config["ankle_strategy"]
    gate = config["boundary_rejection_gate"]

    if strategy != "ray_sphere_with_fk_fallback":
        raise ValueError(f"Unsupported ankle strategy: {strategy}")

    if not isinstance(gate, bool):
        raise TypeError("boundary_rejection_gate must be bool")
        
    width = config["image_width_px"]
    height = config["image_height_px"]
    margin = config["boundary_margin_px"]
    
    if not (isinstance(width, int) and width > 0): raise ValueError("image_width_px must be positive int")
    if not (isinstance(height, int) and height > 0): raise ValueError("image_height_px must be positive int")
    if not (isinstance(margin, int) and margin >= 0): raise ValueError("boundary_margin_px must be nonnegative int")
    if 2 * margin >= width or 2 * margin >= height: raise ValueError("margin too large")

    return {
        "ankle_strategy": strategy,
        "boundary_rejection_gate": gate,
        "boundary_margin_px": margin,
        "image_width_px": width,
        "image_height_px": height,
    }

def apply_boundary_rejection_gate(scores2d, kpts2d, image_height_px, image_width_px, margin_px):
    limit_y = image_height_px - margin_px
    limit_x_min = 0
    limit_x_max = image_width_px
    print(f"  [EXPERIMENTAL] Applying Candidate C boundary rejection gate ({margin_px}px)")
    for f in range(kpts2d.shape[0]):
        for c in range(kpts2d.shape[1]):
            for ji in (15, 16):
                x = kpts2d[f, c, ji, 0]
                y = kpts2d[f, c, ji, 1]
                if y > limit_y or x < limit_x_min or x > limit_x_max:
                    scores2d[f, c, ji] = 0.0

def preprocess_observations_for_config(scores2d, kpts2d, config, boundary_gate_fn=apply_boundary_rejection_gate):
    resolved = resolve_pipeline_config(config)
    if resolved["boundary_rejection_gate"]:
        boundary_gate_fn(
            scores2d,
            kpts2d,
            image_height_px=resolved["image_height_px"],
            image_width_px=resolved["image_width_px"],
            margin_px=resolved["boundary_margin_px"]
        )
    return scores2d, kpts2d

def solve_ankles_for_config(pts3d_clean, ankle_bl, calib, kpts2d, scores2d, ankle_gates, ankle_joint_defs, config):
    resolved = resolve_pipeline_config(config)
    strategy = resolved["ankle_strategy"]

    if strategy == "ray_sphere_with_fk_fallback":
        return infer_by_ray_sphere(pts3d_clean, ankle_bl, calib, kpts2d, scores2d, ankle_gates, ankle_joint_defs)

    raise ValueError(f"Unsupported ankle strategy: {strategy}")
# ============================

REPROJ_P95 = {"00_26": 50.74, "00_29": 43.35, "00_30": 62.28}

# ── BVH skeleton definition ───────────────────────────────────────────────────
# 15 joints (removed duplicate neck=spine):
# 0: root      (mid-hip, virtual)
# 1: l_hip     (COCO 11)
# 2: l_knee    (COCO 13)
# 3: l_ankle   (COCO 15, inferred)
# 4: r_hip     (COCO 12)
# 5: r_knee    (COCO 14)
# 6: r_ankle   (COCO 16, inferred)
# 7: spine     (mid-shoulder, virtual)
# 8: l_shoulder(COCO 5)
# 9: l_elbow   (COCO 7)
# 10: l_wrist  (COCO 9)
# 11: r_shoulder(COCO 6)
# 12: r_elbow  (COCO 8)
# 13: r_wrist  (COCO 10)
# 14: head     (COCO 0, nose proxy)

BVH_NAMES   = [
    "root", "l_hip", "l_knee", "l_ankle",
    "r_hip", "r_knee", "r_ankle",
    "spine", "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
    "head",
]
J = 15
BVH_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12, 7]
# COCO-17 index for each BVH joint (-1 = virtual, computed from others)
BVH_COCO    = [-1, 11, 13, 15, 12, 14, 16, -1, 5, 7, 9, 6, 8, 10, 0]

DRAW_EDGES_BVH = [(BVH_PARENTS[j], j) for j in range(1, J)]
DRAW_EDGES_17  = [(11,13),(13,15),(12,14),(14,16),(5,7),(7,9),
                  (6,8),(8,10),(5,11),(6,12),(5,6),(11,12),(0,5),(0,6)]

# Bone direction vectors in canonical rest pose (parent-local coordinates)
REST_DIR = np.zeros((J, 3))
REST_DIR[1]  = [ 1,  0,  0]  # l_hip points +X
REST_DIR[2]  = [ 0, -1,  0]  # l_knee points -Y
REST_DIR[3]  = [ 0, -1,  0]  # l_ankle points -Y
REST_DIR[4]  = [-1,  0,  0]  # r_hip points -X
REST_DIR[5]  = [ 0, -1,  0]  # r_knee points -Y
REST_DIR[6]  = [ 0, -1,  0]  # r_ankle points -Y
REST_DIR[7]  = [ 0,  1,  0]  # spine points +Y
REST_DIR[8]  = [ 1,  0,  0]  # l_shoulder points +X
REST_DIR[9]  = [ 1,  0,  0]  # l_elbow points +X
REST_DIR[10] = [ 1,  0,  0]  # l_wrist points +X
REST_DIR[11] = [-1,  0,  0]  # r_shoulder points -X
REST_DIR[12] = [-1,  0,  0]  # r_elbow points -X
REST_DIR[13] = [-1,  0,  0]  # r_wrist points -X
REST_DIR[14] = [ 0,  1,  0]  # head points +Y


# ── GT loading ────────────────────────────────────────────────────────────────

def load_gt_sequence():
    gt_kpts  = np.full((N_FRAMES, 17, 3), np.nan, dtype=np.float32)
    gt_valid = np.zeros(N_FRAMES, dtype=bool)
    for i in range(N_FRAMES):
        fn = GT_DIR / f"body3DScene_{START_FRAME+i:08d}.json"
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


# ── MPJPE helpers ─────────────────────────────────────────────────────────────

def mpjpe_excl(pred, gt, valid_mask, excl=()):
    """RC-MPJPE in mm, on frames where valid_mask=True, joints in excl skipped."""
    errs = []
    for f in range(len(valid_mask)):
        if not valid_mask[f]: continue
        p = pred[f].copy(); g = gt[f].copy()
        for ji in excl: p[ji] = np.nan; g[ji] = np.nan
        vj = np.isfinite(p).all(1) & np.isfinite(g).all(1)
        if vj.sum() < 3: continue
        rp = (p[11]+p[12])/2; rg = (g[11]+g[12])/2
        diff = np.where(vj[:,None], (p-rp)-(g-rg), np.nan)
        errs.append(float(np.nanmean(np.linalg.norm(diff, axis=1))*10.0))
    return float(np.nanmean(errs)) if errs else float("nan")


# ── BVH world positions from COCO-17 keypoints ───────────────────────────────

def build_bvh_positions(pts3d):
    """(F, 15, 3) world positions for all BVH joints."""
    pos = np.full((N_FRAMES, J, 3), np.nan)
    for ji in range(J):
        cj = BVH_COCO[ji]
        if cj >= 0:
            pos[:, ji] = pts3d[:, cj]
    pos[:, 0] = (pts3d[:, 11] + pts3d[:, 12]) / 2.0   # root = mid-hip
    pos[:, 7] = (pts3d[:, 5]  + pts3d[:, 6])  / 2.0   # spine = mid-shoulder
    return pos


# ── Bone lengths from measured frames directly ────────────────────────────────

def compute_bvh_bone_lengths(bvh_pos, recon_mask):
    """
    Median ||child - parent|| per BVH joint, measured frames only.
    Shins (ji=3, 6) override with 0.98 * thigh (ankles are NaN in measured data).
    Returns (J,) array.
    """
    meas = ~recon_mask
    bl = np.zeros(J)
    for ji in range(1, J):
        p = BVH_PARENTS[ji]
        dists = np.linalg.norm(bvh_pos[meas, ji] - bvh_pos[meas, p], axis=1)
        finite = dists[np.isfinite(dists)]
        bl[ji] = float(np.median(finite)) if len(finite) >= 3 else 10.0

    # Shin prior: ankles were not triangulated so bl[3] and bl[6] will be ~0 or NaN
    l_thigh = bl[2] if bl[2] > 1.0 else 30.0   # l_hip -> l_knee
    r_thigh = bl[5] if bl[5] > 1.0 else 30.0
    bl[3] = 0.98 * l_thigh   # l_ankle
    bl[6] = 0.98 * r_thigh   # r_ankle

    return bl


# ── Rotation solve ────────────────────────────────────────────────────────────

def _arc_rotation(src, tgt):
    """Shortest-arc Rotation mapping unit vector src to unit vector tgt."""
    src = src / (np.linalg.norm(src) + 1e-9)
    tgt = tgt / (np.linalg.norm(tgt) + 1e-9)
    cross = np.cross(src, tgt)
    dot   = float(np.clip(np.dot(src, tgt), -1.0, 1.0))
    cl    = np.linalg.norm(cross)
    if cl < 1e-9:
        if dot > 0: return Rotation.identity()
        perp = np.array([1,0,0]) if abs(src[0]) < 0.9 else np.array([0,1,0])
        ax = np.cross(src, perp); ax /= np.linalg.norm(ax)
        return Rotation.from_rotvec(ax * np.pi)
    return Rotation.from_rotvec((cross / cl) * np.arctan2(cl, dot))


def get_pelvis_rotation(pos, bl=None):
    """Calculates global rotation for root, based on hips and spine."""
    x = pos[1] - pos[4] # r_hip (4) to l_hip (1)
    if not np.isfinite(x).all():
        x = np.array([1.0, 0.0, 0.0])
        
    y = pos[7] - pos[0] # root (0) to spine (7)
    if not np.isfinite(y).all():
        y = np.array([0.0, 1.0, 0.0])
        
    x = x / (np.linalg.norm(x) + 1e-9)
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-9)
    # Ensure orthogonality, keeping Y (spine) exact
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    
    return Rotation.from_matrix(np.column_stack((x, y, z)))


def get_spine_rotation(pos, parent_rot, bl=None):
    """Calculates global rotation for spine, based on shoulders and head."""
    x = pos[8] - pos[11] # r_shoulder (11) to l_shoulder (8)
    if not np.isfinite(x).all():
        x = parent_rot.apply([1, 0, 0])
        
    y = pos[14] - pos[7] # spine (7) to head (14)
    if not np.isfinite(y).all():
        y = parent_rot.apply([0, 1, 0])
        
    x = x / (np.linalg.norm(x) + 1e-9)
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-9)
    # Ensure orthogonality, keeping X (shoulders) exact
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    
    return Rotation.from_matrix(np.column_stack((x, y, z)))


def fit_skeleton_frame(bvh_pos_f, bl):
    """Sequential IK/FK solve for a single frame."""
    global_rot = [Rotation.identity()] * J
    local_rot  = [Rotation.identity()] * J
    fk_pos     = np.full((J, 3), np.nan)

    root = bvh_pos_f[0]
    if not np.isfinite(root).all():
        return local_rot, fk_pos
    fk_pos[0] = root

    global_rot[0] = get_pelvis_rotation(bvh_pos_f, bl)
    local_rot[0]  = global_rot[0]
    
    global_rot[7] = get_spine_rotation(bvh_pos_f, global_rot[0], bl)

    for ji in range(1, J):
        if ji == 7:
            p = BVH_PARENTS[ji]
            local_rot[ji] = global_rot[p].inv() * global_rot[ji]
            continue
            
        p = BVH_PARENTS[ji]
        children = [c for c in range(J) if BVH_PARENTS[c] == ji]
        
        if len(children) == 1:
            child = children[0]
            target_vec = bvh_pos_f[child] - bvh_pos_f[ji]
            if np.isfinite(target_vec).all() and np.linalg.norm(target_vec) > 1e-6:
                target_dir = target_vec / np.linalg.norm(target_vec)
            else:
                target_dir = global_rot[p].apply(REST_DIR[child])
            
            default_dir = global_rot[p].apply(REST_DIR[child])
            R_twist = _arc_rotation(default_dir, target_dir)
            global_rot[ji] = R_twist * global_rot[p]
            local_rot[ji] = global_rot[p].inv() * global_rot[ji]
        else:
            # Leaf joints
            global_rot[ji] = global_rot[p]
            local_rot[ji] = Rotation.identity()

    for ji in range(1, J):
        p = BVH_PARENTS[ji]
        offset_world = global_rot[p].apply(REST_DIR[ji] * bl[ji])
        fk_pos[ji] = fk_pos[p] + offset_world

    return local_rot, fk_pos


def fit_skeleton_sequence(bvh_pos, bl):
    all_rots = []
    all_fk   = np.full((N_FRAMES, J, 3), np.nan)
    for f in range(N_FRAMES):
        lr, fp = fit_skeleton_frame(bvh_pos[f], bl)
        all_rots.append(lr)
        all_fk[f] = fp
    return all_rots, all_fk


# ── Generalized ray+sphere joint inference ────────────────────────────────────

def _unproject_ray(uv, K_inv, R_world, cam_center):
    """Returns (origin, unit_direction) of the 3D ray through pixel uv."""
    ray_cam   = K_inv @ np.array([uv[0], uv[1], 1.0])
    ray_world = R_world @ ray_cam           # R_world = R.T for OpenCV R
    ray_world /= np.linalg.norm(ray_world)
    return cam_center.copy(), ray_world


def _ray_sphere_intersect(ray_o, ray_d, sphere_center, radius):
    """Returns list of 3D intersection points with t > 0."""
    oc = ray_o - sphere_center
    a  = np.dot(ray_d, ray_d)
    b  = 2.0 * np.dot(oc, ray_d)
    c  = np.dot(oc, oc) - radius**2
    disc = b**2 - 4*a*c
    if disc < 0:
        return []
    return [ray_o + t * ray_d
            for t in [(-b + np.sqrt(disc)) / (2*a), (-b - np.sqrt(disc)) / (2*a)]
            if t > 0]


def _fk_guess(parent_pos, grandparent_pos, bone_len):
    """Extend parent→grandparent direction from parent by bone_len."""
    d = parent_pos - grandparent_pos
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    return parent_pos + (d / n) * bone_len


def infer_by_ray_sphere(pts3d, bl_dict, calib, kpts2d, scores2d,
                        conf_gate_dict, joint_defs):
    """
    Generalized single-camera ray+sphere joint inference.

    For each joint, each frame:
      - Count cameras where score >= conf_gate.
      - 0 cameras visible: FK/IK from parent direction.
      - 1 camera visible: ray+sphere. Previous-frame disambiguation.
        Falls back to FK/IK only if sphere intersection fails.
      - (≥2 cameras: already handled by triangulation; no action.)

    Args:
        pts3d:        (N, J_coco, 3) world positions (NaN where not triangulated).
        bl_dict:      {coco_child_idx: bone_length_cm}.
        calib:        calibration dict {cam_name: cam_object}.
        kpts2d:       (N, n_cams, n_joints, 2) 2D detections.
        scores2d:     (N, n_cams, n_joints) detection confidence scores.
        conf_gate_dict: {coco_idx: confidence_threshold}.
        joint_defs:   {coco_child_idx: (coco_parent_idx, coco_grandparent_idx)}.

    Returns:
        pts_out:      pts3d copy with inferred joints filled in.
        stats:        {coco_idx: {'n_ray': int, 'n_fk': int, 'n_noparent': int}}.
    """
    pts_out = pts3d.copy()
    stats = {ki: {"n_ray": 0, "n_fk": 0, "n_noparent": 0} for ki in joint_defs}

    # Pre-compute camera params
    cam_params = {}
    for ci, cn in enumerate(CAM_NAMES):
        if cn not in calib:
            continue
        c = calib[cn]
        K     = c.K.astype(np.float64)
        R     = c.R.astype(np.float64)
        t     = c.t.reshape(3,).astype(np.float64)
        K_inv = np.linalg.inv(K)
        cam_center = -(R.T @ t)
        cam_params[ci] = (K_inv, R.T, cam_center)   # R.T = R_world

    N = pts3d.shape[0]
    n_coco = kpts2d.shape[2]

    for coco_ki, (parent_coco, grandparent_coco) in joint_defs.items():
        if coco_ki >= n_coco:
            continue
        gate  = conf_gate_dict.get(coco_ki, 0.50)
        bl    = bl_dict.get(coco_ki, None)
        if bl is None or bl < 1.0:
            continue

        prev_pos = None   # rolling previous valid position

        for fi in range(N):
            parent_pos = pts_out[fi, parent_coco] if parent_coco < pts_out.shape[1] else np.full(3, np.nan)
            if not np.isfinite(parent_pos).all():
                stats[coco_ki]["n_noparent"] += 1
                continue

            # Count visible cameras above gate
            visible = []
            for ci in cam_params:
                if ci >= scores2d.shape[1]:
                    continue
                sc = float(scores2d[fi, ci, coco_ki])
                if sc >= gate:
                    uv = kpts2d[fi, ci, coco_ki]
                    if np.isfinite(uv).all():
                        visible.append((ci, sc, uv))

            n_vis = len(visible)

            if n_vis >= 2:
                # Already triangulated by Stage 4.2 — leave as-is.
                if np.isfinite(pts_out[fi, coco_ki]).all():
                    prev_pos = pts_out[fi, coco_ki]
                continue

            result = None

            if n_vis == 1:
                ci, sc, uv = visible[0]
                K_inv, R_world, cam_center = cam_params[ci]
                ray_o, ray_d = _unproject_ray(uv, K_inv, R_world, cam_center)
                candidates = _ray_sphere_intersect(ray_o, ray_d, parent_pos, bl)
                if candidates:
                    if prev_pos is not None:
                        result = min(candidates, key=lambda p: np.linalg.norm(p - prev_pos))
                    else:
                        gp = pts_out[fi, grandparent_coco] if grandparent_coco < pts_out.shape[1] else np.full(3, np.nan)
                        if np.isfinite(gp).all():
                            fk = _fk_guess(parent_pos, gp, bl)
                            if fk is not None:
                                result = min(candidates, key=lambda p: np.linalg.norm(p - fk))
                            else:
                                result = min(candidates, key=lambda p: p[1])
                        else:
                            result = min(candidates, key=lambda p: p[1])
                    if result is not None:
                        stats[coco_ki]["n_ray"] += 1

            if result is None:
                # 0 cameras or sphere miss → FK/IK
                gp = pts_out[fi, grandparent_coco] if grandparent_coco < pts_out.shape[1] else np.full(3, np.nan)
                if np.isfinite(gp).all():
                    result = _fk_guess(parent_pos, gp, bl)
                    if result is not None:
                        stats[coco_ki]["n_fk"] += 1

            if result is not None:
                pts_out[fi, coco_ki] = result
                prev_pos = result

    return pts_out, stats


# ── BVH writer ────────────────────────────────────────────────────────────────

def write_bvh_coco(out_path, rotations, fk_pos, bl):
    children = [[] for _ in range(J)]
    for ji in range(1, J):
        children[BVH_PARENTS[ji]].append(ji)

    with open(out_path, "w") as f:
        f.write("HIERARCHY\n")

        def write_node(ji, indent):
            ind = "  " * indent
            kw = "ROOT" if BVH_PARENTS[ji] == -1 else "JOINT"
            f.write(f"{ind}{kw} {BVH_NAMES[ji]}\n{ind}{{\n")
            if BVH_PARENTS[ji] == -1:
                f.write(f"{ind}  OFFSET 0.000000 0.000000 0.000000\n")
                f.write(f"{ind}  CHANNELS 6 Xposition Yposition Zposition "
                        f"Xrotation Yrotation Zrotation\n")
            else:
                offset = REST_DIR[ji] * bl[ji]
                f.write(f"{ind}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
                f.write(f"{ind}  CHANNELS 3 Xrotation Yrotation Zrotation\n")
            if not children[ji]:
                f.write(f"{ind}  End Site\n{ind}  {{\n")
                f.write(f"{ind}    OFFSET 0.000000 0.000000 0.000000\n")
                f.write(f"{ind}  }}\n")
            else:
                for ch in children[ji]:
                    write_node(ch, indent + 1)
            f.write(f"{ind}}}\n")

        write_node(0, 0)
        f.write(f"MOTION\nFrames: {N_FRAMES}\nFrame Time: {1.0/FPS:.6f}\n")

        for fi in range(N_FRAMES):
            root_t = fk_pos[fi, 0]
            if not np.isfinite(root_t).all():
                root_t = np.zeros(3)
            row = list(root_t)
            for ji in range(J):
                euler = rotations[fi][ji].as_euler("XYZ", degrees=True)
                row.extend(euler.tolist())
            f.write(" ".join(f"{v:.4f}" for v in row) + "\n")

    print(f"  BVH written: {out_path}  ({N_FRAMES} frames, {J} joints)")


# ── BVH round-trip: self-contained FK from rotations ─────────────────────────

def bvh_roundtrip_fk(rotations, bl):
    fk_verify = np.full((N_FRAMES, J, 3), np.nan)
    for fi in range(N_FRAMES):
        global_rot = [Rotation.identity()] * J
        pos = np.full((J, 3), np.nan)

        for ji in range(J):
            p = BVH_PARENTS[ji]
            if p == -1:
                global_rot[ji] = rotations[fi][ji]
            else:
                global_rot[ji] = global_rot[p] * rotations[fi][ji]

        pos[0] = np.array([0.0, 0.0, 0.0])  # relative mode: root at origin
        for ji in range(1, J):
            p = BVH_PARENTS[ji]
            if not np.isfinite(pos[p]).all(): continue
            pos[ji] = pos[p] + global_rot[p].apply(REST_DIR[ji] * bl[ji])

        fk_verify[fi] = pos

    return fk_verify


def roundtrip_error(fk_pos, fk_verify):
    diffs = []
    for fi in range(N_FRAMES):
        for ji in range(1, J):
            a = fk_pos[fi, ji] - fk_pos[fi, 0]
            b = fk_verify[fi, ji]
            if np.isfinite(a).all() and np.isfinite(b).all():
                diffs.append(np.linalg.norm(a - b) * 10.0)
    return float(np.max(diffs)) if diffs else float("nan")


# ── Visuals ───────────────────────────────────────────────────────────────────

def make_3panel_gif(pts_clean, fk_pos, gt, valid, recon_mask, out_path,
                    n_frames=60, fps=10):
    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(131, projection="3d"),
           fig.add_subplot(132, projection="3d"),
           fig.add_subplot(133, projection="3d")]

    valid_gt = gt[valid]
    center = np.nanmedian((valid_gt[:,11]+valid_gt[:,12])/2.0, axis=0) \
             if len(valid_gt) > 0 else np.zeros(3)

    def _plot17(ax, pts, title, f, color="#2980B9"):
        ax.cla(); c=center; r=100
        ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
        ax.set_title(f"{title}\nf={f}", fontsize=7)
        for i,j in DRAW_EDGES_17:
            if np.all(np.isfinite(pts[[i,j]])):
                ax.plot(*pts[[i,j]].T, "-", color=color, lw=1.2)
        vld = np.isfinite(pts).all(1)
        if vld.any(): ax.scatter(*pts[vld].T, c="r", s=10)

    def _plotbvh(ax, pos, title, f, color="#2ECC71"):
        ax.cla(); c=center; r=100
        ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
        ax.set_title(f"{title}\nf={f}", fontsize=7)
        for (pi, ci) in DRAW_EDGES_BVH:
            if np.all(np.isfinite(pos[[pi,ci]])):
                ax.plot(*pos[[pi,ci]].T, "-", color=color, lw=1.2)
        vld = np.isfinite(pos).all(1)
        if vld.any(): ax.scatter(*pos[vld].T, c="r", s=10)

    def update(fi):
        _plot17(axs[0], pts_clean[fi], "Cleaned 3D", fi)
        col = "#E74C3C" if recon_mask[fi] else "#2ECC71"
        _plotbvh(axs[1], fk_pos[fi], "Fitted FK\ngreen=meas red=recon", fi, col)
        g = gt[fi] if valid[fi] else np.full((17,3), np.nan)
        _plot17(axs[2], g, "GT", fi, color="#E67E22")
        return []

    # Span the entire sequence (sample every 3rd frame, so 10fps playback is 1x speed)
    frames = list(range(0, N_FRAMES, 3))
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, interval=1000/fps)
    ani.save(str(out_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  3-panel GIF: {out_path}")


def plot_rotation_timeseries(rotations, out_path):
    JOINTS = [("root", 0), ("l_hip", 1), ("l_knee", 2), ("l_shoulder", 8)]
    fig, axes = plt.subplots(len(JOINTS), 3, figsize=(18, 10), sharex=True)
    for row, (jname, ji) in enumerate(JOINTS):
        eulers = np.array([rotations[f][ji].as_euler("XYZ", degrees=True)
                           for f in range(N_FRAMES)])
        for col, axlabel in enumerate(["X", "Y", "Z"]):
            axes[row, col].plot(eulers[:, col], lw=0.8, color="#3498DB")
            axes[row, col].set_ylabel(f"{jname} {axlabel} (deg)", fontsize=7)
            deltas = np.abs(np.diff(eulers[:, col]))
            for s in np.where(deltas > 30)[0]:
                axes[row, col].axvline(s, color="red", alpha=0.4, lw=0.5)
    axes[-1, 1].set_xlabel("Frame")
    fig.suptitle("Rotation Time-series (raw solve, no extra smoothing)\n"
                 "Red = >30 deg/frame delta", fontsize=10)
    fig.tight_layout(); fig.savefig(str(out_path), dpi=120); plt.close(fig)
    print(f"  Rotation time-series: {out_path}")


# ── Report helpers ────────────────────────────────────────────────────────────

def print_row(label, val, thr, ok, note=""):
    def fmt(v):
        if v is None: return "--"
        if isinstance(v, float): return f"{v:.2f}"
        return str(v)
    s = "PASS" if ok else "FAIL"
    print(f"{label:<52} {fmt(val):>10}  {thr:<22} {s}  {note}")


# ── Joint gate coverage check ────────────────────────────────────────────────

ZERO_CAM_FLAG_THRESH = 0.90   # flag a joint if it spends >90% of frames below gate

def compute_joint_gate_coverage(kpts2d, scores2d, conf_gate_dict, joint_defs, n_frames):
    """
    For each joint in joint_defs, count how many frames have 0 / 1 / 2+ cameras
    above the confidence gate.
    Returns:
        coverage: {coco_idx: {'n0': int, 'n1': int, 'n2plus': int, 'pct_zero': float}}
        flagged:  list of (coco_idx, name, pct_zero) for joints above ZERO_CAM_FLAG_THRESH
    """
    n_cams = scores2d.shape[1]
    coverage = {}
    flagged  = []
    joint_names = {15: "l_ankle", 16: "r_ankle"}   # extend as needed

    for coco_ki, (parent_coco, gp_coco) in joint_defs.items():
        if coco_ki >= scores2d.shape[2]:
            continue
        gate  = conf_gate_dict.get(coco_ki, 0.50)
        n0 = n1 = n2p = 0
        for fi in range(n_frames):
            if fi >= scores2d.shape[0]:
                break
            n_above = sum(1 for ci in range(n_cams)
                         if float(scores2d[fi, ci, coco_ki]) >= gate)
            if   n_above == 0: n0  += 1
            elif n_above == 1: n1  += 1
            else:              n2p += 1
        pct_zero = 100.0 * n0 / n_frames
        coverage[coco_ki] = {"n0": n0, "n1": n1, "n2plus": n2p,
                              "pct_zero": pct_zero, "gate": gate}
        name = joint_names.get(coco_ki, f"coco_{coco_ki}")
        if pct_zero > ZERO_CAM_FLAG_THRESH * 100:
            flagged.append((coco_ki, name, pct_zero))

    return coverage, flagged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("Stage 6a -- Kinematic Solve + BVH Export (v3, true IK structure)")
    print("=" * 80)

    pts3d    = np.load(S42_DIR / "pts3d_clean.npy")
    gt_kpts, gt_valid = load_gt_sequence()
    gap_raw  = json.loads((S42_DIR / "gap_log.json").read_text())
    recon_mask = np.zeros(N_FRAMES, dtype=bool)
    for rec in gap_raw["gap_log"]:
        if rec.get("reconstructed"):
            recon_mask[rec["start_frame"]: rec["end_frame"] + 1] = True
    valid_measured = gt_valid & ~recon_mask

    print(f"\nInput: {pts3d.shape}  GT valid={gt_valid.sum()}  "
          f"Measured={valid_measured.sum()}  Reconstructed={recon_mask.sum()}")

    print("\nStep 0: Stage 4.2 verification...")
    s0_mpjpe = mpjpe_excl(pts3d, gt_kpts, valid_measured, [13, 14, 15, 16])
    s0_thr   = 52.9 * 1.05
    s0_pass  = s0_mpjpe <= s0_thr
    print(f"  RC-MPJPE (excl. knees+ankles, measured): {s0_mpjpe:.2f}mm  "
          f"threshold={s0_thr:.2f}mm  {'PASS' if s0_pass else 'FAIL -- STOP'}")
    if not s0_pass:
        return

    print("\nOutlier rejection...")
    reproj_path = S42_DIR / "reproj_err.npy"
    if not reproj_path.exists():
        print("  reproj_err.npy not saved by Stage 4.2 -- skipped.")
        print("  Stage 4.2 max reproj: cam 00_29 = 338px (1-2 frames); not widespread.")
        n_rejected = 0
        pts3d_clean = pts3d.copy()
    else:
        reproj = np.load(reproj_path)
        pts3d_clean = pts3d.copy()
        n_rejected = 0
        for ci, cn in enumerate(CAM_NAMES):
            ceiling = 3.0 * REPROJ_P95[cn]
            bad = reproj[:, :, ci] > ceiling
            pts3d_clean[bad] = np.nan
            n_rejected += int(bad.sum())
            if bad.any():
                print(f"  [{cn}] ceiling={ceiling:.1f}px  rejected {int(bad.sum())} joint-frames")
        print(f"  Total rejected: {n_rejected}")

    print("\nBuilding BVH positions...")
    bvh_pos = build_bvh_positions(pts3d_clean)

    print("Computing bone lengths from BVH joint pairs (measured frames only)...")
    bl = compute_bvh_bone_lengths(bvh_pos, recon_mask)
    for ji in range(1, J):
        print(f"  [{ji:2d}] {BVH_NAMES[ji]:<14}  {bl[ji]:.2f} cm")

    print("\nAnkle inference (ray+sphere all cameras, FK fallback)...")
    calib = load_panoptic_calib(CALIB_JSON)

    if not S3_NPZ.exists():
        print("  [SKIP] Stage-3 NPZ not found; all frames use FK/IK fallback.")
        pts3d_w_ankles = pts3d_clean.copy()
        infer_stats = {15: {"n_ray": 0, "n_fk": 0, "n_noparent": 0},
                       16: {"n_ray": 0, "n_fk": 0, "n_noparent": 0}}
    else:
        npz_data  = np.load(S3_NPZ)
        kpts2d    = npz_data["keypoints"]
        scores2d  = npz_data["scores"]

        # joint_defs: {coco_child: (coco_parent, coco_grandparent)}
        ankle_joint_defs = {
            15: (13, 11),   # l_ankle  parent=l_knee  gp=l_hip
            16: (14, 12),   # r_ankle  parent=r_knee  gp=r_hip
        }
        # Bone lengths for ankles (COCO index → cm)
        ankle_bl = {15: bl[3], 16: bl[6]}   # bl[3]=l_ankle, bl[6]=r_ankle from BVH
        # Confidence gates
        ankle_gates = {15: 0.35, 16: 0.35}

        # 1. Preprocess observations based on active configuration
        scores2d, kpts2d = preprocess_observations_for_config(
            scores2d, kpts2d, DEFAULT_PIPELINE_CONFIG
        )

        # 2. Infer 3D ankles using the configured solver strategy
        pts3d_w_ankles, infer_stats = solve_ankles_for_config(
            pts3d_clean, ankle_bl, calib, kpts2d, scores2d, ankle_gates, ankle_joint_defs,
            DEFAULT_PIPELINE_CONFIG
        )

    for coco_ki in [15, 16]:
        s = infer_stats[coco_ki]
        print(f"  COCO {coco_ki}: ray={s['n_ray']}  fk={s['n_fk']}  no_parent={s['n_noparent']}")
    n_ray   = sum(infer_stats[k]["n_ray"] for k in [15, 16])
    n_prior = sum(infer_stats[k]["n_fk"]  for k in [15, 16])

    ankle_errs = []
    for f in range(N_FRAMES):
        if not gt_valid[f]: continue
        for ji in [15, 16]:
            p = pts3d_w_ankles[f, ji]; g = gt_kpts[f, ji]
            if np.isfinite(p).all() and np.isfinite(g).all():
                ankle_errs.append(float(np.linalg.norm(p-g)*10.0))
    ankle_mpjpe = float(np.mean(ankle_errs)) if ankle_errs else float("nan")
    pct_ankle = 100.0 * len(ankle_errs) / (2*N_FRAMES)
    print(f"  Ankle MPJPE vs GT: {ankle_mpjpe:.1f}mm over {pct_ankle:.0f}% of frames (report only)")

    # ── Per-joint 0-camera gate coverage check ─────────────────────────────────
    print("\nPer-joint gate coverage check...")
    if S3_NPZ.exists():
        gate_coverage, gate_flagged = compute_joint_gate_coverage(
            kpts2d, scores2d, ankle_gates, ankle_joint_defs, N_FRAMES
        )
        for coco_ki, cov in gate_coverage.items():
            name = {15: "l_ankle", 16: "r_ankle"}.get(coco_ki, f"coco_{coco_ki}")
            flag = "  [FLAG: zero-camera >90% of clip]" if cov["pct_zero"] > ZERO_CAM_FLAG_THRESH * 100 else ""
            print(f"  {name} (gate={cov['gate']:.2f}): "
                  f"0-cam={cov['n0']}f({cov['pct_zero']:.0f}%)  "
                  f"1-cam={cov['n1']}f  2+cam={cov['n2plus']}f{flag}")
    else:
        gate_coverage = {}; gate_flagged = []
        print("  [SKIP] No Stage-3 NPZ.")

    bvh_pos = build_bvh_positions(pts3d_w_ankles)


    print("\nSolving per-frame rotations...")
    rotations, fk_pos = fit_skeleton_sequence(bvh_pos, bl)
    print(f"  Done. fk_pos shape: {fk_pos.shape}")

    print("\nFK MPJPE vs GT (measured frames, ankles excl.)...")
    fk_coco = np.full((N_FRAMES, 17, 3), np.nan)
    bvh_to_coco = {1:11, 2:13, 3:15, 4:12, 5:14, 6:16,
                   8:5, 9:7, 10:9, 11:6, 12:8, 13:10, 14:0}
    for bj, cj in bvh_to_coco.items():
        fk_coco[:, cj] = fk_pos[:, bj]

    mpjpe_clean = mpjpe_excl(pts3d_clean, gt_kpts, valid_measured, [15,16])
    mpjpe_fk    = mpjpe_excl(fk_coco, gt_kpts, valid_measured, [15,16])
    fk_delta    = 100.0*(mpjpe_fk - mpjpe_clean) / (mpjpe_clean + 1e-9)
    print(f"  Cleaned 3D (measured frames, excl. ankles): {mpjpe_clean:.1f}mm")
    print(f"  FK (measured frames, excl. ankles):         {mpjpe_fk:.1f}mm  "
          f"delta={fk_delta:+.1f}%  {'PASS' if fk_delta<=15 else 'FAIL'}")

    print("\nBone CV after FK fitting...")
    cv_nonzero = []
    for ji in range(1, J):
        p = BVH_PARENTS[ji]
        lens = np.linalg.norm(fk_pos[:, ji] - fk_pos[:, p], axis=1)
        finite = lens[np.isfinite(lens)]
        if len(finite) > 1:
            cv = float(np.std(finite) / (np.mean(finite) + 1e-9))
            if cv > 0.001:
                cv_nonzero.append(f"{BVH_NAMES[ji]}:CV={cv:.4f}")
    if cv_nonzero:
        print(f"  [WARN] {cv_nonzero}")
    else:
        print("  All bone CVs < 0.001. PASS.")

    print("\nRotation smoothness (>30 deg/frame outside recon spans)...")
    spike_joints = {}
    for ji in range(J):
        for fi in range(1, N_FRAMES):
            if recon_mask[fi] or recon_mask[fi-1]: continue
            diff = rotations[fi-1][ji].inv() * rotations[fi][ji]
            angle_deg = np.degrees(diff.magnitude())
            if angle_deg > 30.0:
                spike_joints[BVH_NAMES[ji]] = spike_joints.get(BVH_NAMES[ji], 0) + 1
    if spike_joints:
        print(f"  [FLAG] {spike_joints}")
    else:
        print("  No true angular velocity spikes > 30 deg/frame outside reconstructed spans.")

    print("\nExporting BVH...")
    bvh_path = OUT_DIR / "stage6a_mocap.bvh"
    write_bvh_coco(bvh_path, rotations, fk_pos, bl)

    print("Round-trip FK verification (re-derive positions from stored rotations)...")
    fk_verify   = bvh_roundtrip_fk(rotations, bl)
    max_rt_err  = roundtrip_error(fk_pos, fk_verify)
    rt_pass     = max_rt_err < 1.0
    print(f"  Max root-relative position error: {max_rt_err:.4f}mm  "
          f"{'PASS' if rt_pass else 'FAIL (>1mm)'}")

    sidecar = {
        "bvh_file": "stage6a_mocap.bvh",
        "fps": FPS, "frames": N_FRAMES, "joints": J,
        "coordinate_system": "Y-up, cm, Panoptic world",
        "joint_names": BVH_NAMES, "parents": BVH_PARENTS,
        "bone_lengths_cm": {BVH_NAMES[ji]: round(float(bl[ji]), 3) for ji in range(1, J)},
        "reconstructed_frames": [int(f) for f in np.where(recon_mask)[0]],
        "outlier_rejected_joint_frames": n_rejected,
        "ankle_inference": {
            "method": "ray_sphere_all_cameras_fk_fallback",
            "ray_frames": n_ray, "fk_frames": n_prior,
            "ankle_mpjpe_vs_gt_mm": round(ankle_mpjpe, 2),
            "per_joint_gate_coverage": {
                str(k): v for k, v in gate_coverage.items()
            },
        },
        "blender_import_checklist": [
            "File > Import > Motion Capture (.bvh)",
            "VERIFY: character is upright (Y-up), NOT tilted or upside-down",
            "VERIFY: left hand is on the character's LEFT (no mirroring)",
            "VERIFY: raising your right hand -> right arm bone moves",
            "VERIFY: feet point downward (-Y), not floating or penetrating floor",
            "VERIFY: head is at the top",
            "If skeleton is rotated 90deg: BVH Y-up conflicts with scene Z-up; "
            "adjust the import Forward/Up axis in the import dialog",
        ],
    }
    sidecar_path = OUT_DIR / "stage6a_mocap_sidecar.json"
    with open(sidecar_path, "w") as fp:
        json.dump(sidecar, fp, indent=2)
    print(f"  Sidecar JSON: {sidecar_path}")

    print("\nGenerating visuals...")
    make_3panel_gif(pts3d_clean, fk_pos, gt_kpts, gt_valid, recon_mask,
                    OUT_DIR / "skeleton_3panel.gif")
    plot_rotation_timeseries(rotations, OUT_DIR / "rotation_timeseries.png")

    print("\n" + "=" * 95)
    print(f"{'Metric':<52} {'Value':>10}  {'Threshold':<22} {'Status'}")
    print("-" * 95)

    print_row("Step 0: Stage 4.2 MPJPE excl. knees (mm)",
              s0_mpjpe, f"<={s0_thr:.1f}mm", s0_pass)
    print_row("Outlier rejection: joint-frames dropped",
              n_rejected, "report only", True)
    print_row("FK vs GT MPJPE excl. ankles, measured (mm)",
              mpjpe_fk, "<15% delta from cleaned 3D", fk_delta <= 15.0,
              f"cleaned={mpjpe_clean:.1f}mm  delta={fk_delta:+.1f}%")
    print_row("Bone CV after fitting",
              "0.000" if not cv_nonzero else cv_nonzero[0],
              "exactly 0 (<0.001)", not cv_nonzero)
    print_row("Ankle inference coverage (total)",
              f"{pct_ankle:.0f}%", "report only", True,
              f"ray={n_ray} fk={n_prior}")
    print_row("Ankle MPJPE vs GT (mm)",
              ankle_mpjpe, "report only", True)

    # Per-joint 0-camera coverage: always explicit rows ─────────────────────
    if gate_coverage:
        cov_l = gate_coverage.get(15, {})
        cov_r = gate_coverage.get(16, {})
        if cov_l:
            pz = cov_l["pct_zero"]
            flag = pz > ZERO_CAM_FLAG_THRESH * 100
            print_row("l_ankle: frames with 0 cameras above gate",
                      f"{pz:.0f}%", f"<{ZERO_CAM_FLAG_THRESH*100:.0f}% = pass", not flag,
                      f"n0={cov_l['n0']} n1={cov_l['n1']} n2+={cov_l['n2plus']}")
        if cov_r:
            pz = cov_r["pct_zero"]
            flag = pz > ZERO_CAM_FLAG_THRESH * 100
            print_row("r_ankle: frames with 0 cameras above gate",
                      f"{pz:.0f}%", f"<{ZERO_CAM_FLAG_THRESH*100:.0f}% = pass", not flag,
                      f"n0={cov_r['n0']} n1={cov_r['n1']} n2+={cov_r['n2plus']}")
        for coco_ki, jname, pct_zero in gate_flagged:
            if coco_ki not in (15, 16):  # already printed above
                print_row(f"{jname}: 0-camera rate [AUTO-FLAG]",
                          f"{pct_zero:.0f}%", f"<{ZERO_CAM_FLAG_THRESH*100:.0f}%", False,
                          "signal: insufficient camera coverage for this joint")
    print_row("Rotation spikes >30 deg/frame (outside recon)",
              len(spike_joints), "<=2 acceptable", len(spike_joints)<=2,
              str(list(spike_joints.keys())[:4]) if spike_joints else "")
    print_row("BVH round-trip max position error (mm)",
              max_rt_err, "<1.0mm", rt_pass)

    ready = s0_pass and fk_delta <= 15.0 and not cv_nonzero and rt_pass and len(spike_joints) <= 2
    print("\n" + "=" * 80)
    if ready:
        print("READY FOR 6b (RETARGETING): YES")
        print(f"  FK MPJPE={mpjpe_fk:.1f}mm ({fk_delta:+.1f}% vs cleaned), "
              f"bone CV=0, round-trip={max_rt_err:.4f}mm")
        print(f"  {recon_mask.sum()} reconstructed frames flagged in sidecar JSON.")
        print(f"  Ankles inferred ({n_ray} ray, {n_prior} prior); all frames complete.")
        print(f"\n  USER MANUAL CHECK (Blender):")
        print(f"  Import: {bvh_path}")
        for c in sidecar["blender_import_checklist"][:5]:
            print(f"    - {c}")
    else:
        reasons = []
        if not s0_pass:     reasons.append(f"Step0 FAIL: {s0_mpjpe:.1f}mm")
        if fk_delta > 10.0: reasons.append(f"FK degradation {fk_delta:+.1f}%")
        if cv_nonzero:      reasons.append(f"non-zero CV: {cv_nonzero[:2]}")
        if not rt_pass:     reasons.append(f"round-trip {max_rt_err:.2f}mm")
        print("READY FOR 6b (RETARGETING): NO")
        print("  Reasons: " + " | ".join(reasons))
    print("=" * 80)
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
