from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from aimocap.retarget.fbx_rig import Skeleton


LOWER_BODY_JOINTS = (
    "thigh_l",
    "calf_l",
    "foot_l",
    "ball_l",
    "thigh_r",
    "calf_r",
    "foot_r",
    "ball_r",
)

MOCAP_LOWER_BODY_INDICES = (0, 9, 10, 11, 12, 13, 14)


def _continuous_quaternions(quats: np.ndarray) -> np.ndarray:
    out = quats.copy()
    for frame in range(1, len(out)):
        if np.dot(out[frame - 1], out[frame]) < 0.0:
            out[frame] *= -1.0
    return out


def _filter_track(values: np.ndarray, fps: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    if len(values) < 9:
        return values.copy()
    cutoff = min(cutoff_hz / (0.5 * fps), 0.99)
    sos = butter(order, cutoff, output="sos")
    out = values.copy()
    padlen = min(15, len(values) - 1)
    for dim in range(values.shape[1]):
        try:
            out[:, dim] = sosfiltfilt(sos, values[:, dim], padlen=padlen)
        except ValueError:
            out[:, dim] = values[:, dim]
    return out


def _limit_quaternion_velocity(quats: np.ndarray, max_deg_per_frame: float) -> np.ndarray:
    if max_deg_per_frame <= 0.0 or len(quats) < 2:
        return quats.copy()

    max_rad = np.deg2rad(max_deg_per_frame)
    out = quats.copy()
    for frame in range(1, len(out)):
        prev = Rotation.from_quat(out[frame - 1])
        desired = Rotation.from_quat(out[frame])
        delta = prev.inv() * desired
        angle = delta.magnitude()
        if angle <= max_rad or angle < 1e-8:
            continue
        limited_delta = Rotation.from_rotvec(delta.as_rotvec() * (max_rad / angle))
        out[frame] = (prev * limited_delta).as_quat()
    return _continuous_quaternions(out)


def _rotation_delta_metrics(
    skel: Skeleton,
    root_seq: np.ndarray,
    local_quats: np.ndarray,
    joint_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    global_quats = np.zeros_like(local_quats)
    for frame in range(len(local_quats)):
        _, global_quats[frame] = skel.get_forward_kinematics(
            local_quats[frame],
            root_translation=root_seq[frame],
        )

    metrics: dict[str, dict[str, float]] = {}
    for name in joint_names:
        if name not in skel.name_to_idx:
            continue
        idx = skel.name_to_idx[name]
        rots = Rotation.from_quat(global_quats[:, idx])
        deltas = (rots[:-1].inv() * rots[1:]).magnitude()
        deltas_deg = np.degrees(deltas)
        metrics[name] = {
            "p50_deg_per_frame": float(np.percentile(deltas_deg, 50)),
            "p95_deg_per_frame": float(np.percentile(deltas_deg, 95)),
            "max_deg_per_frame": float(np.max(deltas_deg)),
        }
    return metrics


def _position_metrics_dict(
    positions: dict[str, np.ndarray],
    joint_names: tuple[str, ...],
    fps: float,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for name in joint_names:
        if name not in positions: continue
        vel = np.linalg.norm(np.diff(positions[name], axis=0), axis=1) * fps
        acc = np.linalg.norm(np.diff(positions[name], n=2, axis=0), axis=1)
        metrics[name] = {
            "speed_p95_cm_per_sec": float(np.percentile(vel, 95)) if len(vel) else 0.0,
            "speed_max_cm_per_sec": float(np.max(vel)) if len(vel) else 0.0,
            "accel_p95_cm_per_frame2": float(np.percentile(acc, 95)) if len(acc) else 0.0,
            "accel_max_cm_per_frame2": float(np.max(acc)) if len(acc) else 0.0,
        }
    return metrics


def stabilize_mocap_fit_targets(
    target_pts: dict[str, np.ndarray],
    fps: float = 30.0,
    cutoff_hz: float = 1.5,
    debug_dir: str | Path | None = None,
    adaptive: bool = True,
    fast_cutoff_hz: float = 6.0,
    speed_threshold_cm_per_s: float = 50.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Smooth noisy lower-body 3D fit targets before proxy IK solves rotations.

    When adaptive=True, the cutoff is raised during high-motion segments
    (per-frame speed > speed_threshold_cm_per_s) so fast bends are not lagged.
    """
    if "pelvis" not in target_pts or len(target_pts["pelvis"]) < 9:
        return target_pts, {"enabled": False, "reason": "too few frames"}

    names = ("pelvis", "hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r")
    before_metrics = _position_metrics_dict(target_pts, names, fps)

    out = {k: v.copy() for k, v in target_pts.items()}

    if adaptive:
        # Compute per-frame speed across all lower-body targets
        all_speeds = []
        for name in names:
            if name in target_pts:
                v = np.linalg.norm(np.diff(target_pts[name], axis=0), axis=1) * fps
                all_speeds.append(v)
        if all_speeds:
            max_speed = np.max(np.stack(all_speeds), axis=0)
            kernel = np.ones(5) / 5
            smooth_speed = np.convolve(
                np.pad(max_speed, (2, 2), mode="edge"), kernel, mode="valid"
            )
            excess = np.clip(smooth_speed / speed_threshold_cm_per_s - 1.0, 0, 1)
            per_frame_cutoff = cutoff_hz + excess * (fast_cutoff_hz - cutoff_hz)
        else:
            per_frame_cutoff = np.full(len(target_pts.get("pelvis", [])), cutoff_hz)
        # Use mean cutoff (higher during fast motion)
        effective_cutoff = float(np.mean(per_frame_cutoff))
        max_eff_cutoff = float(np.max(per_frame_cutoff))
    else:
        effective_cutoff = cutoff_hz
        max_eff_cutoff = cutoff_hz

    for name in names:
        if name in target_pts:
            out[name] = _filter_track(target_pts[name], fps=fps, cutoff_hz=effective_cutoff)

    after_metrics = _position_metrics_dict(out, names, fps)
    diagnostics = {
        "enabled": True,
        "fps": float(fps),
        "cutoff_hz": float(cutoff_hz),
        "adaptive": adaptive,
        "fast_cutoff_hz": float(fast_cutoff_hz),
        "max_effective_cutoff_hz": max_eff_cutoff,
        "before": before_metrics,
        "after": after_metrics,
    }

    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        with (debug_path / "mocap_target_stabilization_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

    return out, diagnostics


def stabilize_lower_body_animation(
    skel: Skeleton,
    animation_data: np.ndarray,
    fps: float = 30.0,
    cutoff_hz: float = 1.5,
    max_deg_per_frame: float = 30.0,
    debug_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Zero-phase smooth target lower-limb rotations without changing bone lengths."""
    if animation_data.size == 0:
        return animation_data, {}

    joint_names = tuple(name for name in LOWER_BODY_JOINTS if name in skel.name_to_idx)
    if not joint_names:
        return animation_data, {"enabled": False, "reason": "missing lower body bones"}

    root_seq = animation_data[:, :3].copy()
    eulers = animation_data[:, 3:].reshape(len(animation_data), skel.num_joints, 3)
    local_quats = Rotation.from_euler("xyz", eulers.reshape(-1, 3), degrees=False).as_quat()
    local_quats = local_quats.reshape(len(animation_data), skel.num_joints, 4)

    before_metrics = _rotation_delta_metrics(skel, root_seq, local_quats, joint_names)
    smoothed = local_quats.copy()
    for name in joint_names:
        idx = skel.name_to_idx[name]
        track = _continuous_quaternions(local_quats[:, idx])
        filtered = _filter_track(track, fps=fps, cutoff_hz=cutoff_hz)
        norms = np.linalg.norm(filtered, axis=1, keepdims=True)
        bad = norms[:, 0] < 1e-8
        filtered = filtered / np.where(norms < 1e-8, 1.0, norms)
        filtered[bad] = track[bad]
        filtered = _limit_quaternion_velocity(filtered, max_deg_per_frame)
        smoothed[:, idx] = filtered

    after_metrics = _rotation_delta_metrics(skel, root_seq, smoothed, joint_names)
    smoothed_eulers = Rotation.from_quat(smoothed.reshape(-1, 4)).as_euler(
        "xyz",
        degrees=False,
    )
    out = animation_data.copy()
    out[:, 3:] = smoothed_eulers.reshape(len(animation_data), -1)

    diagnostics = {
        "enabled": True,
        "fps": float(fps),
        "cutoff_hz": float(cutoff_hz),
        "max_deg_per_frame": float(max_deg_per_frame),
        "before": before_metrics,
        "after": after_metrics,
    }

    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        with (debug_path / "leg_stabilization_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

    return out, diagnostics
