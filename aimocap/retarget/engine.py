import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.bvh import write_bvh
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.foot_lock import apply_foot_lock
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.mocap_skeleton import (
    MocapSkeleton,
    extract_mocap_points,
)
from aimocap.retarget.temporal import stabilize_lower_body_animation
from aimocap.retarget.temporal import stabilize_mocap_fit_targets


def _mocap_global_rotations(
    mocap_ik: MocapIKSolver,
    root_t: np.ndarray,
    local_quats: np.ndarray,
) -> np.ndarray:
    """Return (num_joints, 4) global quaternions of the solved proxy skeleton."""
    _, global_rot = mocap_ik.forward_kinematics(root_t, local_quats)
    return global_rot


def _rotation_between_vectors(rest_vec: np.ndarray, target_vec: np.ndarray) -> Rotation:
    rest_norm = np.linalg.norm(rest_vec)
    target_norm = np.linalg.norm(target_vec)
    if rest_norm < 1e-8 or target_norm < 1e-8:
        return Rotation.identity()

    try:
        rot, _ = Rotation.align_vectors(
            (target_vec / target_norm).reshape(1, 3),
            (rest_vec / rest_norm).reshape(1, 3),
        )
    except Exception:
        rot = Rotation.identity()
    return rot


def _validate_fbx_hierarchy(fbx_skel: Skeleton) -> tuple[int, int]:
    root_indices = [i for i, p in enumerate(fbx_skel.parents) if p == -1]
    if len(root_indices) != 1:
        raise ValueError(f"Expected exactly one FBX root, found {len(root_indices)}")

    for i, p in enumerate(fbx_skel.parents):
        if p >= i:
            raise ValueError(
                f"FBX hierarchy is not topologically sorted at {fbx_skel.node_names[i]}"
            )

    return root_indices[0], fbx_skel.name_to_idx["pelvis"]


def retarget_to_fbx(
    triangulated_npz: str,
    fbx_rig_path: str,
    output_bvh: str,
    max_frames: int = None,
    fps: float = 30.0,
    mocap_target_stabilization: bool = True,
    mocap_target_cutoff_hz: float = 1.5,
    leg_stabilization: bool = True,
    leg_stabilize_cutoff_hz: float = 1.5,
    leg_max_deg_per_frame: float = 30.0,
    foot_lock: bool = True,
    foot_lock_debug_dir: str | None = None,
    pose2d_npz: str | None = None,
):
    print(f"Loading triangulated data from {triangulated_npz}")
    triangulated_data = np.load(triangulated_npz)
    pts3d_raw = triangulated_data["skeleton3d"] * 100.0  # m -> cm

    # Y-up (triangulation) -> Z-up (Unreal/Manny): [x, y, z] -> [x, -z, y]
    pts3d = np.zeros_like(pts3d_raw)
    pts3d[..., 0] = pts3d_raw[..., 0]
    pts3d[..., 1] = -pts3d_raw[..., 2]
    pts3d[..., 2] = pts3d_raw[..., 1]

    if "confidence" in triangulated_data:
        weights = triangulated_data["confidence"]
    else:
        weights = np.any(pts3d != 0, axis=-1).astype(np.float32)

    num_frames = len(pts3d)
    if max_frames is not None and max_frames > 0:
        num_frames = min(num_frames, max_frames)

    fbx_skel = Skeleton(fbx_rig_path)
    fbx_root_idx, pelvis_idx = _validate_fbx_hierarchy(fbx_skel)

    print("Stage 1: Fitting MocapSkeleton...")
    mocap_skel = MocapSkeleton(pts3d, weights, fbx_skel=fbx_skel)
    mocap_ik = MocapIKSolver(mocap_skel)

    mocap_target_pts = extract_mocap_points(pts3d)
    if mocap_target_stabilization:
        print("Stage 1a: Stabilizing lower-body fit targets...")
        mocap_target_pts, target_diag = stabilize_mocap_fit_targets(
            mocap_target_pts,
            fps=fps,
            cutoff_hz=mocap_target_cutoff_hz,
            debug_dir=foot_lock_debug_dir,
        )
        if target_diag.get("enabled"):
            for name in ("ankle_l", "ankle_r"):
                before = target_diag["before"][name]
                after = target_diag["after"][name]
                print(
                    f"  {name}: p95 speed "
                    f"{before['speed_p95_cm_per_sec']:.1f} -> "
                    f"{after['speed_p95_cm_per_sec']:.1f} cm/s"
                )
        else:
            print(f"  skipped: {target_diag.get('reason', 'unknown reason')}")

    solved_root = []
    solved_local_quats = []
    prev_x = None
    prev_prev_x = None
    for f in range(num_frames):
        if f % 10 == 0:
            print(f"Solving Mocap Frame {f}/{num_frames}")
        measured = {k: v[f] for k, v in mocap_target_pts.items()}
        x_opt = mocap_ik.solve_frame(
            measured,
            prev_x=prev_x,
            temporal_weight=0.03,
            frame_idx=f,
            prev_prev_x=prev_prev_x,
            accel_weight=0.3,
        )
        prev_prev_x = prev_x.copy() if prev_x is not None else None
        prev_x = x_opt
        root_t, local_quats = mocap_ik._state_to_local_rotations(x_opt)
        solved_root.append(root_t)
        solved_local_quats.append(local_quats)

    # ── Stage 1b: Side-Agnostic Arm Orientation Rewrite ──────────────────────
    print("Stage 1b: Arm orientation rewrite...")
    from aimocap.retarget.arm_orient import (
        solve_upperarm_global,
        solve_forearm_global,
        solve_hand_global_v2,
        resolve_bone_axis_parent_local,
    )

    proxy_rest_global = [Rotation.identity()] * mocap_ik.num_joints

    arm_meta = {}
    for side in ("l", "r"):
        clav_name  = f"clavicle_{side}"
        upper_name = f"shoulder_{side}"    # proxy name for upper arm
        fore_name  = f"elbow_{side}"       # proxy name for forearm pivot
        hand_name  = f"wrist_{side}"       # proxy name for hand pivot
        
        names = [clav_name, upper_name, fore_name, hand_name]
        if not all(n in mocap_skel.name_to_idx for n in names):
            arm_meta[side] = None
            continue
        
        clav_i  = mocap_skel.name_to_idx[clav_name]
        upper_i = mocap_skel.name_to_idx[upper_name]
        fore_i  = mocap_skel.name_to_idx[fore_name]
        hand_i  = mocap_skel.name_to_idx[hand_name]
        
        rest_t = mocap_skel.rest_t  # shape (J, 3)
        
        e_upper_parent = resolve_bone_axis_parent_local(
            rest_t, proxy_rest_global[clav_i], upper_i, fore_i
        )
        e_fore_upper_local = resolve_bone_axis_parent_local(
            rest_t, proxy_rest_global[upper_i], fore_i, hand_i
        )
        
        arm_meta[side] = {
            "clav_i":  clav_i,
            "upper_i": upper_i,
            "fore_i":  fore_i,
            "hand_i":  hand_i,
            "e_upper_parent":     e_upper_parent,
            "e_fore_upper_local": e_fore_upper_local,
            "G_parent_rest": proxy_rest_global[clav_i],
            "G_rest_upper":  proxy_rest_global[upper_i],
            "G_rest_fore":   proxy_rest_global[fore_i],
            "G_rest_hand":   proxy_rest_global[hand_i],
        }

    def _globals_to_locals(mocap_ik, old_local_quats, replacements, solved_root):
        _, global_q = mocap_ik.forward_kinematics(solved_root, old_local_quats)
        global_rots = list(Rotation.from_quat(global_q))
        
        for idx, G_new in replacements.items():
            global_rots[idx] = G_new
        
        new_local_quats = np.empty_like(old_local_quats)
        for idx in range(mocap_ik.num_joints):
            p = int(mocap_ik.parents[idx])
            if p < 0:
                L = global_rots[idx]
            else:
                L = global_rots[p].inv() * global_rots[idx]
            new_local_quats[idx] = L.as_quat()
        
        return new_local_quats

    phi_prev = {"l": 0.0, "r": 0.0}
    arm_source_histogram = {}
    arm_phi_upper = {"l": [], "r": []}
    arm_phi_fore  = {"l": [], "r": []}

    for f in range(num_frames):
        old_local_quats = solved_local_quats[f]
        
        _, global_q = mocap_ik.forward_kinematics(solved_root[f], old_local_quats)
        global_rots = list(Rotation.from_quat(global_q))
        
        replacements = {}
        
        for side in ("l", "r"):
            meta = arm_meta.get(side)
            if meta is None:
                continue
            
            clav_i  = meta["clav_i"]
            upper_i = meta["upper_i"]
            fore_i  = meta["fore_i"]
            hand_i  = meta["hand_i"]
            
            sh_w = mocap_target_pts[f"shoulder_{side}"][f]
            el_w = mocap_target_pts[f"elbow_{side}"][f]
            wr_w = mocap_target_pts[f"wrist_{side}"][f]
            
            if sh_w is None or el_w is None or wr_w is None:
                continue
            if not (np.all(np.isfinite(sh_w)) and np.all(np.isfinite(el_w))
                    and np.all(np.isfinite(wr_w))):
                continue
            
            D_parent = global_rots[clav_i] * meta["G_parent_rest"].inv()
            
            G_upper, obs_upper = solve_upperarm_global(
                D_parent=D_parent,
                G_parent_rest=meta["G_parent_rest"],
                e_upper_parent=meta["e_upper_parent"],
                e_fore_upper_local=meta["e_fore_upper_local"],
                G_rest_upper=meta["G_rest_upper"],
                sh_w=sh_w, el_w=el_w, wr_w=wr_w,
                phi_prev=phi_prev[side],
                G_upper_ik=global_rots[upper_i],
            )
            
            G_fore, obs_fore = solve_forearm_global(
                G_upper_w=G_upper,
                G_rest_fore=meta["G_rest_fore"],
                e_fore_loc=meta["e_fore_upper_local"],
                el_w=el_w, wr_w=wr_w,
                phi_prev=0.0,
            )
            
            G_hand = solve_hand_global_v2(
                G_fore_w=G_fore,
                G_rest_fore=meta["G_rest_fore"],
                G_rest_hand=meta["G_rest_hand"],
            )
            
            replacements[upper_i] = G_upper
            replacements[fore_i]  = G_fore
            replacements[hand_i]  = G_hand
            
            phi_prev[side] = obs_upper.value
            src = obs_upper.source
            arm_source_histogram[src] = arm_source_histogram.get(src, 0) + 1
            arm_phi_upper[side].append(obs_upper.value)
            arm_phi_fore[side].append(obs_fore.value)
        
        if not replacements:
            continue
        
        new_local_quats = _globals_to_locals(
            mocap_ik, old_local_quats, replacements, solved_root[f]
        )
        
        tracked_names = [
            f"shoulder_{s}" for s in ("l", "r")
        ] + [
            f"elbow_{s}" for s in ("l", "r")
        ]
        tracked_idx = np.array([
            mocap_skel.name_to_idx[n]
            for n in tracked_names
            if n in mocap_skel.name_to_idx
        ])
        
        before_pos, _ = mocap_ik.forward_kinematics(solved_root[f], old_local_quats)
        after_pos, _  = mocap_ik.forward_kinematics(solved_root[f], new_local_quats)
        
        max_delta = np.max(np.linalg.norm(
            before_pos[tracked_idx] - after_pos[tracked_idx], axis=1
        ))
        assert max_delta < 1e-9, (
            f"Frame {f}: arm rewrite moved shoulder/elbow by {max_delta:.3e} cm — "
            "axial twist must be position-preserving"
        )
        
        solved_local_quats[f] = new_local_quats

    print(f"  Arm source histogram: {arm_source_histogram}")
    for side in ("l", "r"):
        phis = np.array(arm_phi_upper[side])
        if len(phis) > 0:
            print(
                f"  Upper {side}: p05={np.percentile(np.degrees(phis), 5):.1f}° "
                f"p50={np.percentile(np.degrees(phis), 50):.1f}° "
                f"p95={np.percentile(np.degrees(phis), 95):.1f}°"
            )

    print("Stage 2: Rotation Transfer to FBX...")
    fbx_rest_global_pos, fbx_rest_global_rot = fbx_skel.get_forward_kinematics()
    fbx_rest_global = Rotation.from_quat(fbx_rest_global_rot)
    fbx_rest_local = Rotation.from_quat(np.array(fbx_skel.rest_rotations))

    mocap_to_fbx_idx = {
        mocap_i: fbx_skel.name_to_idx[name]
        for mocap_i, name in mocap_skel.fbx_mapping.items()
    }
    fbx_to_mocap_idx = {
        fbx_i: mocap_i for mocap_i, fbx_i in mocap_to_fbx_idx.items()
    }
    # Hip/knee roll is underdetermined from sparse 3D joints, but handled in IK now.
    
    fbx_animation = np.zeros((num_frames, 3 + fbx_skel.num_joints * 3))

    for f in range(num_frames):
        mocap_pos, mocap_global_quats = mocap_ik.forward_kinematics(
            solved_root[f],
            solved_local_quats[f],
        )
        mocap_global = Rotation.from_quat(mocap_global_quats)

        frame_global = [None] * fbx_skel.num_joints
        frame_local = [None] * fbx_skel.num_joints

        for fbx_i in range(fbx_skel.num_joints):
            p = fbx_skel.parents[fbx_i]

            if fbx_i in fbx_to_mocap_idx:
                mocap_i = fbx_to_mocap_idx[fbx_i]
                # Proxy global rotation is a world-space delta from rest.
                R_global_target = mocap_global[mocap_i] * fbx_rest_global[fbx_i]
            elif p == -1:
                R_global_target = fbx_rest_global[fbx_i]
            else:
                # Preserve unmapped bones in rest local pose under the animated
                # parent, so twist/finger/intermediate bones inherit correctly.
                R_global_target = frame_global[p] * fbx_rest_local[fbx_i]

            if p == -1:
                R_local = R_global_target
            else:
                R_local = frame_global[p].inv() * R_global_target

            frame_global[fbx_i] = R_global_target
            frame_local[fbx_i] = R_local

        # Align Manny's pelvis position to the solved proxy pelvis. BVH writes
        # root translation relative to the FBX root rest translation.
        root_world = (
            solved_root[f]
            - frame_global[fbx_root_idx].apply(fbx_skel.rest_translations[pelvis_idx])
        )
        fbx_animation[f, 0:3] = root_world - fbx_skel.rest_translations[fbx_root_idx]

        frame_local_quats = np.array([r.as_quat() for r in frame_local])
        eulers = Rotation.from_quat(frame_local_quats).as_euler("xyz", degrees=False)
        fbx_animation[f, 3:] = eulers.flatten()

    if leg_stabilization:
        print("Stage 3: Target-space lower-body stabilization...")
        fbx_animation, leg_diag = stabilize_lower_body_animation(
            fbx_skel,
            fbx_animation,
            fps=fps,
            cutoff_hz=leg_stabilize_cutoff_hz,
            max_deg_per_frame=leg_max_deg_per_frame,
            debug_dir=foot_lock_debug_dir,
        )
        if leg_diag.get("enabled"):
            for name in ("calf_l", "foot_l", "calf_r", "foot_r"):
                if name not in leg_diag["before"]:
                    continue
                before = leg_diag["before"][name]
                after = leg_diag["after"][name]
                print(
                    f"  {name}: p95 rot delta "
                    f"{before['p95_deg_per_frame']:.2f} -> "
                    f"{after['p95_deg_per_frame']:.2f} deg/frame"
                )
        else:
            print(f"  skipped: {leg_diag.get('reason', 'unknown reason')}")

    if foot_lock:
        print("Stage 4: Target-space foot locking...")
        fbx_animation, foot_diag = apply_foot_lock(
            fbx_skel,
            fbx_animation,
            pts3d[:num_frames],
            fps=fps,
            debug_dir=foot_lock_debug_dir,
            pose2d_npz=pose2d_npz,
        )
        if foot_diag.get("enabled"):
            for side in ("left", "right"):
                before = foot_diag["before"][side]
                after = foot_diag["after"][side]
                print(
                    f"  {side}: foot drift {before['foot_drift_cm']:.2f} -> "
                    f"{after['foot_drift_cm']:.2f} cm, ball drift "
                    f"{before['ball_drift_cm']:.2f} -> {after['ball_drift_cm']:.2f} cm"
                )
        else:
            print(f"  skipped: {foot_diag.get('reason', 'unknown reason')}")

    npy_out = str(output_bvh).replace(".bvh", "_solved.npy")
    print(f"Saving backup to {npy_out}")
    np.save(npy_out, fbx_animation)

    print(f"Saving BVH to {output_bvh}")
    write_bvh(output_bvh, fbx_skel, fbx_animation, fps=fps)
    
    # FBX export: direct fcurve injection (no COPY_TRANSFORMS, no rest-folding)
    from aimocap.retarget.fbx_export import write_fbx
    fbx_out = str(output_bvh).replace(".bvh", ".fbx")
    print(f"Exporting FBX to {fbx_out}...")
    write_fbx(
        npy_path=npy_out,
        fbx_in_path=str(fbx_rig_path),
        fbx_out_path=fbx_out,
        fps=fps,
        bone_names=fbx_skel.node_names,
    )

    print("Done.")


retarget_sequence = retarget_to_fbx
