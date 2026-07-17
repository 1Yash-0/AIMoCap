import os
import sys
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from multiprocessing import Pool
import cv2

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics
from aimocap.math.filter import fill_gaps_with_logging, filter_skeleton_one_euro

# ── Shipping Path Constants & Hierarchies ────────────────────────────────────
J = 15
BVH_NAMES = [
    "root", "l_hip", "l_knee", "l_ankle",
    "r_hip", "r_knee", "r_ankle",
    "spine", "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
    "head",
]
BVH_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12, 7]
BVH_COCO = [-1, 11, 13, 15, 12, 14, 16, -1, 5, 7, 9, 6, 8, 10, 0]
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

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

# ── Helper functions for sequential IK/FK ────────────────────────────────────

def get_pelvis_rotation(bvh_pos_f, bl):
    lh = bvh_pos_f[1]; rh = bvh_pos_f[4]; spine = bvh_pos_f[7]
    if not (np.isfinite(lh).all() and np.isfinite(rh).all() and np.isfinite(spine).all()):
        return Rotation.identity()
    x = lh - rh
    x = x / (np.linalg.norm(x) + 1e-9)
    mid = (lh + rh) / 2.0
    y = spine - mid
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-9)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    return Rotation.from_matrix(np.column_stack((x, y, z)))

def get_spine_rotation(bvh_pos_f, R_pelvis, bl):
    spine = bvh_pos_f[7]; ls = bvh_pos_f[8]; rs = bvh_pos_f[11]
    if not (np.isfinite(ls).all() and np.isfinite(rs).all()):
        return R_pelvis
    x = ls - rs
    x = x / (np.linalg.norm(x) + 1e-9)
    z = R_pelvis.apply([0, 0, 1])
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-9)
    return Rotation.from_matrix(np.column_stack((x, y, z)))

def _arc_rotation(src, tgt):
    src = src / (np.linalg.norm(src) + 1e-9)
    tgt = tgt / (np.linalg.norm(tgt) + 1e-9)
    cross = np.cross(src, tgt)
    dot   = float(np.clip(np.dot(src, tgt), -1.0, 1.0))
    cl    = np.linalg.norm(cross)
    if cl < 1e-9:
        if dot > 0: return Rotation.identity()
        axis = np.array([0, 1, 0]) if abs(src[0]) > 0.9 else np.array([1, 0, 0])
        axis = np.cross(src, axis)
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi)
    axis = cross / cl
    angle = np.arccos(dot)
    return Rotation.from_rotvec(axis * angle)

def fit_skeleton_frame(bvh_pos_f, bl):
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
            global_rot[ji] = global_rot[p]
            local_rot[ji] = Rotation.identity()
            
    # Forward Kinematics pass to compute output locations
    fk_pos = np.zeros((J, 3))
    fk_pos[0] = root
    for ji in range(1, J):
        p = BVH_PARENTS[ji]
        offset_world = global_rot[p].apply(REST_DIR[ji] * bl[ji])
        fk_pos[ji] = fk_pos[p] + offset_world
    return local_rot, fk_pos

def fit_skeleton_sequence(bvh_pos, bl):
    N_FRAMES = len(bvh_pos)
    all_rots = []
    all_fk   = np.full((N_FRAMES, J, 3), np.nan)
    for f in range(N_FRAMES):
        lr, fp = fit_skeleton_frame(bvh_pos[f], bl)
        all_rots.append(lr)
        all_fk[f] = fp
    return all_rots, all_fk

def fit_skeleton_sequence_preserve_finite(bvh_pos, bl):
    """Like fit_skeleton_sequence, but each joint that had a finite MEASURED
    position in bvh_pos is snapped back to it after FK. FK only supplies the
    position for joints that were NaN. This keeps the denoising benefit of
    consistent bone directions while not degrading already-good joints."""
    N = len(bvh_pos)
    _, all_fk = fit_skeleton_sequence(bvh_pos, bl)
    out = all_fk.copy()
    for ji in range(1, J):                 # never override root (ji=0)
        finite = np.all(np.isfinite(bvh_pos[:, ji]), axis=-1)
        out[finite, ji] = bvh_pos[finite, ji]
    return None, out

def build_bvh_positions(pts3d):
    N_FRAMES = len(pts3d)
    pos = np.full((N_FRAMES, J, 3), np.nan)
    for ji in range(J):
        cj = BVH_COCO[ji]
        if cj >= 0:
            pos[:, ji] = pts3d[:, cj]
    pos[:, 0] = (pts3d[:, 11] + pts3d[:, 12]) / 2.0
    pos[:, 7] = (pts3d[:, 5]  + pts3d[:, 6])  / 2.0
    return pos

# ── Ankle ray+sphere inference ───────────────────────────────────────────────

def _unproject_ray(uv, K_inv, R_world, cam_center):
    ray_cam   = K_inv @ np.array([uv[0], uv[1], 1.0])
    ray_world = R_world @ ray_cam
    ray_world /= np.linalg.norm(ray_world)
    return cam_center.copy(), ray_world

def _ray_sphere_intersect(ray_o, ray_d, sphere_center, radius):
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
    d = parent_pos - grandparent_pos
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    return parent_pos + (d / n) * bone_len

def infer_by_ray_sphere(pts3d, bl_dict, calib, kpts2d, scores2d, conf_gate_dict, joint_defs, CAMS):
    pts_out = pts3d.copy()
    stats = {ki: {"n_ray": 0, "n_fk": 0, "n_noparent": 0} for ki in joint_defs}
    cam_params = {}
    for ci, cn in enumerate(CAMS):
        if cn not in calib: continue
        c = calib[cn]
        K     = c.K.astype(np.float64)
        R     = c.R.astype(np.float64)
        t     = c.t.reshape(3,).astype(np.float64)
        K_inv = np.linalg.inv(K)
        cam_center = -(R.T @ t)
        cam_params[ci] = (K_inv, R.T, cam_center)
    N = pts3d.shape[0]
    for coco_ki, (parent_coco, grandparent_coco) in joint_defs.items():
        gate  = conf_gate_dict.get(coco_ki, 0.50)
        bl    = bl_dict.get(coco_ki, None)
        if bl is None or bl < 1.0: continue
        prev_pos = None
        for fi in range(N):
            parent_pos = pts_out[fi, parent_coco]
            if not np.isfinite(parent_pos).all():
                stats[coco_ki]["n_noparent"] += 1
                continue
            visible = []
            for ci in cam_params:
                sc = float(scores2d[fi, ci, coco_ki])
                if sc >= gate:
                    uv = kpts2d[fi, ci, coco_ki]
                    if np.isfinite(uv).all():
                        visible.append((ci, sc, uv))
            n_vis = len(visible)
            if n_vis >= 2:
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
                        gp = pts_out[fi, grandparent_coco]
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
                gp = pts_out[fi, grandparent_coco]
                if np.isfinite(gp).all():
                    result = _fk_guess(parent_pos, gp, bl)
                    if result is not None:
                        stats[coco_ki]["n_fk"] += 1
            if result is not None:
                pts_out[fi, coco_ki] = result
                prev_pos = result
    return pts_out, stats

# ── Candidate E Geometric Consistency ────────────────────────────────────────

def triangulate_2_views_dlt(u1, u2, P1, P2):
    A = np.zeros((4, 4))
    A[0] = u1[0] * P1[2] - P1[0]
    A[1] = u1[1] * P1[2] - P1[1]
    A[2] = u2[0] * P2[2] - P2[0]
    A[3] = u2[1] * P2[2] - P2[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def get_reproj_error(X3D, uv, P):
    X = np.append(X3D, 1.0)
    proj = P @ X
    if abs(proj[2]) < 1e-9: return np.inf
    proj = proj[:2] / proj[2]
    return np.linalg.norm(proj - uv)

def candidate_e_geometric_filter(kpts, scores, K_list, extrinsics, thresh=50.0):
    F, C, J, _ = kpts.shape
    new_scores = scores.copy()
    P_list = [K_list[c] @ np.hstack((extrinsics[c][0], extrinsics[c][1])) for c in range(C)]
    
    for f in range(F):
        for j in range(J):
            cams = [c for c in range(C) if scores[f, c, j] >= 0.4]
            if len(cams) == 3:
                p01 = triangulate_2_views_dlt(kpts[f, 0, j], kpts[f, 1, j], P_list[0], P_list[1])
                e2 = get_reproj_error(p01, kpts[f, 2, j], P_list[2])
                if e2 > thresh:
                    new_scores[f, 2, j] = 0.0
                p02 = triangulate_2_views_dlt(kpts[f, 0, j], kpts[f, 2, j], P_list[0], P_list[2])
                e1 = get_reproj_error(p02, kpts[f, 1, j], P_list[1])
                if e1 > thresh:
                    new_scores[f, 1, j] = 0.0
                p12 = triangulate_2_views_dlt(kpts[f, 1, j], kpts[f, 2, j], P_list[1], P_list[2])
                e0 = get_reproj_error(p12, kpts[f, 0, j], P_list[0])
                if e0 > thresh:
                    new_scores[f, 0, j] = 0.0
    return new_scores

# ── BVH Exporter ─────────────────────────────────────────────────────────────

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
                f.write(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n")
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
        N_FRAMES = len(rotations)
        f.write(f"MOTION\nFrames: {N_FRAMES}\nFrame Time: {1.0/30.0:.6f}\n")
        for fi in range(N_FRAMES):
            root_t = fk_pos[fi, 0]
            if not np.isfinite(root_t).all():
                root_t = np.zeros(3)
            row = list(root_t)
            for ji in range(J):
                euler = rotations[fi][ji].as_euler("XYZ", degrees=True)
                row.extend(euler.tolist())
            f.write(" ".join(f"{v:.4f}" for v in row) + "\n")

# ── Picklable Worker ─────────────────────────────────────────────────────────

def get_bbox_height(kpts_fc):
    valid = kpts_fc[np.isfinite(kpts_fc).all(axis=-1)]
    if len(valid) == 0: return 0.0
    return np.max(valid[:, 1]) - np.min(valid[:, 1])

def run_config_worker(task_args):
    # Unpack
    name, policy_type, margin, kpts_orig, scores_orig, K_list, extrinsics, bl, calib_state, CAMS, gt_kpts, gt_valid = task_args
    print(f"Worker starting: {name}")
    
    F, C, J_coco, _ = kpts_orig.shape
    kpts_m = kpts_orig.copy()
    scores_m = scores_orig.copy()
    
    # 1. Apply Adapter
    if policy_type == "A_AR_Baseline":
        for f in range(F):
            for c in range(C):
                valid_mask = scores_m[f, c, :] >= 0.4
                if valid_mask.sum() > 2:
                    valid_kpts = kpts_m[f, c, valid_mask]
                    w = np.max(valid_kpts[:, 0]) - np.min(valid_kpts[:, 0])
                    h = np.max(valid_kpts[:, 1]) - np.min(valid_kpts[:, 1])
                    ar = h / w if w > 0 else 0
                    if ar < 1.8:
                        scores_m[f, c, :] = 0.0
    elif policy_type == "B_No_Gate":
        pass
    elif policy_type.startswith("C_Ankle_"):
        mode = policy_type.split("_")[-1] # "Pixel", "Height", "Box"
        for f in range(F):
            for c in range(C):
                bbox_h = get_bbox_height(kpts_m[f, c]) if mode == "Box" else 0.0
                for ji in [15, 16]:
                    x, y = kpts_m[f, c, ji]
                    if mode == "Pixel":
                        limit = 1080 - margin
                    elif mode == "Height":
                        limit = 1080 * (1.0 - margin)
                    elif mode == "Box":
                        limit = 1080 - (bbox_h * margin)
                    if x < 0 or x > 1920 or y < 0 or y > limit:
                        scores_m[f, c, ji] = 0.0
    elif policy_type.startswith("D_Joint_"):
        mode = policy_type.split("_")[-1]
        for f in range(F):
            for c in range(C):
                bbox_h = get_bbox_height(kpts_m[f, c]) if mode == "Box" else 0.0
                for ji in range(J_coco):
                    x, y = kpts_m[f, c, ji]
                    if mode == "Pixel":
                        limit = 1080 - margin
                    elif mode == "Height":
                        limit = 1080 * (1.0 - margin)
                    elif mode == "Box":
                        limit = 1080 - (bbox_h * margin)
                    if x < 0 or x > 1920 or y < 0 or y > limit:
                        scores_m[f, c, ji] = 0.0
    elif policy_type == "E_Geometric":
        scores_m[:] = candidate_e_geometric_filter(kpts_m, scores_m, K_list, extrinsics, thresh=50.0)
        
    # 2. Triangulate
    tri = triangulate_sequence_with_diagnostics(kpts_m, scores_m, K_list, extrinsics, 0.4, 100.0, 0.0)
    
    # 3. Gap Fill
    filled, gap_log, _ = fill_gaps_with_logging(tri.points3d, [str(i) for i in range(17)], fps=30.0)
    
    # 4. One-Euro
    smoothed = filter_skeleton_one_euro(filled, fps=30.0)
    
    # 5. Kinematic 1
    bvh_pos_init = build_bvh_positions(smoothed)
    rotations_init, fk_pos_init = fit_skeleton_sequence(bvh_pos_init, bl)
    
    # 6. Ankle Inference
    fk_pos_coco = np.full((1800, 17, 3), np.nan, dtype=np.float32)
    bvh_to_coco = {1:11, 2:13, 3:15, 4:12, 5:14, 6:16, 8:5, 9:7, 10:9, 11:6, 12:8, 13:10, 14:0}
    for bj, cj in bvh_to_coco.items():
        fk_pos_coco[:, cj] = fk_pos_init[:, bj]
    fk_pos_coco[:, 0] = (fk_pos_coco[:, 11] + fk_pos_coco[:, 12]) / 2.0
    
    ankle_bl_dict = {15: bl[3], 16: bl[6]}
    ankle_gates = {15: 0.35, 16: 0.35}
    ankle_joint_defs = {15: (13, 11), 16: (14, 12)}
    
    pts3d_w_ankles, infer_stats = infer_by_ray_sphere(
        fk_pos_coco, ankle_bl_dict, calib_state, kpts_m, scores_m, ankle_gates, ankle_joint_defs, CAMS
    )
    
    # 7. Kinematic 2 (Leg Rotation Refitting)
    bvh_pos_final = build_bvh_positions(pts3d_w_ankles)
    rotations_final, fk_pos_final = fit_skeleton_sequence(bvh_pos_final, bl)
    
    # Save BVH
    bvh_path = ROOT / f"outputs/experiments/output_{name}.bvh"
    bvh_path.parent.mkdir(parents=True, exist_ok=True)
    write_bvh_coco(bvh_path, rotations_final, fk_pos_final, bl)
    
    # final fitted positions
    fk_coco = np.full((1800, 17, 3), np.nan)
    for bj, cj in bvh_to_coco.items():
        fk_coco[:, cj] = fk_pos_final[:, bj]
        
    # ── Compute Metrics ──
    measured_mask = gt_valid & np.isfinite(tri.points3d[:, 11, 0])
    mpjpe_vals = []
    mpjpe_torso_vals = []
    mpjpe_hip_vals = []
    
    for fi in range(1800):
        if not measured_mask[fi]: continue
        p = fk_coco[fi].copy() * 10.0 # Convert cm to mm
        g = gt_kpts[fi].copy()        # Already in mm
        vj = np.isfinite(p).all(1) & np.isfinite(g).all(1)
        if vj.sum() < 3: continue
        mpjpe_vals.append(float(np.mean(np.linalg.norm(p[vj] - g[vj], axis=1))))
        rp = (p[11]+p[12])/2
        rg = (g[11]+g[12])/2
        mpjpe_torso_vals.append(float(np.mean(np.linalg.norm((p[vj]-rp) - (g[vj]-rg), axis=1))))
        mpjpe_hip_vals.append(float(np.mean(np.linalg.norm((p[vj]-rp) - (g[vj]-rg), axis=1)))) # same as torso for hips
        
    vel = np.diff(fk_coco, axis=0) * 30.0
    jitter = float(np.nanmean(np.var(vel, axis=0)))
    acc = np.diff(vel, axis=0) * 30.0
    acc_spikes = int(np.sum(np.linalg.norm(acc/100.0, axis=2) > 5.0))
    
    sliding = []
    for fi in range(1, 1800 - 1):
        for ji in [15, 16]:
            if fk_coco[fi, ji, 1] < 5.0:
                sliding.append(np.linalg.norm(vel[fi, ji]))
    mean_sliding = float(np.mean(sliding)) if sliding else 0.0
    penetration = float(np.max(np.maximum(0.0, -fk_coco[:, [15, 16], 1])))
    coverage = float(np.mean(np.isfinite(fk_coco).all(axis=2)))
    
    longest_gap = 0
    for ji in range(17):
        nan_runs = np.isnan(tri.points3d[:, ji, 0])
        changes = np.diff(nan_runs.astype(int), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if len(starts) > 0:
            longest_gap = max(longest_gap, int(np.max(ends - starts)))
            
    metrics = {
        "mean_mpjpe_absolute_mm": float(np.mean(mpjpe_vals)) if mpjpe_vals else 0.0,
        "mean_mpjpe_torso_aligned_mm": float(np.mean(mpjpe_torso_vals)) if mpjpe_torso_vals else 0.0,
        "mean_mpjpe_hip_aligned_mm": float(np.mean(mpjpe_hip_vals)) if mpjpe_hip_vals else 0.0,
        "jitter": jitter,
        "acc_spikes": acc_spikes,
        "foot_sliding_cm_s": mean_sliding,
        "ankle_penetration_cm": penetration,
        "coverage": coverage,
        "longest_gap": longest_gap,
        "ray_frames_l_ankle": int(infer_stats[15]["n_ray"]),
        "fk_frames_l_ankle": int(infer_stats[15]["n_fk"]),
        "ray_frames_r_ankle": int(infer_stats[16]["n_ray"]),
        "fk_frames_r_ankle": int(infer_stats[16]["n_fk"]),
    }
    print(f"Worker completed: {name}")
    return name, metrics

# ── Main Orchestration ───────────────────────────────────────────────────────

def main():
    print("Loading data...")
    npz_path = ROOT / "outputs/canonical_dataset/canonical_detector_pose_observations.npz"
    calib_json = ROOT / "data/panoptic/171204_pose1/calibration_171204_pose1.json"
    
    data = np.load(npz_path)
    kpts_orig = data["kpts"].astype(np.float32)
    scores_orig = data["scores"].astype(np.float32)
    calib = load_calibration(calib_json)
    
    CAMS = ["00_11", "00_12", "00_23"]
    K_list = [calib[cn].K.astype(np.float64) for cn in CAMS]
    extrinsics = [(calib[cn].R.astype(np.float64), calib[cn].t.astype(np.float64).reshape(3,1)) for cn in CAMS]
    
    # ── Load 3D Ground Truth ──
    GT_DIR = ROOT / "data/panoptic/171204_pose1/hdPose3d_stage1_coco19"
    gt_kpts = np.full((1800, 17, 3), np.nan, dtype=np.float32)
    gt_valid = np.zeros(1800, dtype=bool)
    for i in range(1800):
        fn = GT_DIR / f"body3DScene_{150+i:08d}.json"
        if not fn.exists(): continue
        with open(fn) as f:
            js = json.load(f)
        bodies = js.get("bodies", [])
        if not bodies: continue
        joints19 = np.array(bodies[0]["joints19"], dtype=np.float32).reshape(19, 4)
        for c17, p19 in enumerate(COCO17_TO_PAN19):
            gt_kpts[i, c17] = joints19[p19, :3]
        gt_valid[i] = True
        
    print(f"Loaded ground truth: {gt_valid.sum()} valid frames.")

    # ── Derive Frozen Bone Lengths using Candidate B ──
    print("Deriving bone lengths from Candidate B...")
    d_b = triangulate_sequence_with_diagnostics(kpts_orig, scores_orig, K_list, extrinsics, 0.4, 100.0, 0.0)
    bvh_pos_b = build_bvh_positions(d_b.points3d)
    
    bl = np.zeros(J)
    for ji in range(1, J):
        p = BVH_PARENTS[ji]
        dists = np.linalg.norm(bvh_pos_b[:, ji] - bvh_pos_b[:, p], axis=1)
        finite = dists[np.isfinite(dists)]
        bl[ji] = float(np.median(finite)) if len(finite) >= 10 else 10.0
    l_thigh = bl[2] if bl[2] > 1.0 else 30.0
    r_thigh = bl[5] if bl[5] > 1.0 else 30.0
    bl[3] = 0.98 * l_thigh
    bl[6] = 0.98 * r_thigh
    print(f"Frozen bone lengths: {bl.tolist()}")

    # Build Tasks List
    pixel_sweeps = [0, 10, 20, 30, 40, 50]
    height_sweeps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    box_sweeps = [0.0, 0.02, 0.05, 0.10, 0.15]
    
    task_configs = []
    
    # Candidates A, B, E
    task_configs.append(("A_AR_Baseline", "A_AR_Baseline", 0.0))
    task_configs.append(("B_No_Gate", "B_No_Gate", 0.0))
    task_configs.append(("E_Geometric", "E_Geometric", 0.0))
    
    # Candidate C sweeps
    for m in pixel_sweeps:
        task_configs.append((f"C_Ankle_Pixel_{m}", "C_Ankle_Pixel", m))
    for m in height_sweeps:
        task_configs.append((f"C_Ankle_Height_{m:.2f}", "C_Ankle_Height", m))
    for m in box_sweeps:
        task_configs.append((f"C_Ankle_Box_{m:.2f}", "C_Ankle_Box", m))
        
    # Candidate D sweeps
    for m in pixel_sweeps:
        task_configs.append((f"D_Joint_Pixel_{m}", "D_Joint_Pixel", m))
    for m in height_sweeps:
        task_configs.append((f"D_Joint_Height_{m:.2f}", "D_Joint_Height", m))
    for m in box_sweeps:
        task_configs.append((f"D_Joint_Box_{m:.2f}", "D_Joint_Box", m))
        
    # Pack arguments for multiprocessing
    # Note: calib state is loaded as CamParams or pickled dict.
    # calib can be passed directly as a simple picklable dict.
    calib_state = {}
    for cn in CAMS:
        c = calib[cn]
        calib_state[cn] = c # custom object calib is picklable
        
    tasks = []
    for name, policy_type, margin in task_configs:
        tasks.append((
            name, policy_type, margin, kpts_orig, scores_orig, K_list, extrinsics, bl, calib_state, CAMS, gt_kpts, gt_valid
        ))
        
    print(f"Launching {len(tasks)} parallel gating configurations using Multiprocessing Pool...")
    
    # Run in parallel
    results = {}
    with Pool(processes=16) as pool:
        for name, metrics in pool.imap_unordered(run_config_worker, tasks):
            results[name] = metrics
            
    # Save all metrics
    metrics_path = ROOT / "outputs/experiments/metrics_summary.json"
    with open(metrics_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nAll candidate runs completed. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
