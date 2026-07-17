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
    for f in range(num_frames):
        if f % 10 == 0:
            print(f"Solving Mocap Frame {f}/{num_frames}")
        measured = {k: v[f] for k, v in mocap_target_pts.items()}
        x_opt = mocap_ik.solve_frame(
            measured,
            prev_x=prev_x,
            temporal_weight=0.03,
        )
        prev_x = x_opt
        root_t, local_quats = mocap_ik._state_to_local_rotations(x_opt)
        solved_root.append(root_t)
        solved_local_quats.append(local_quats)

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
