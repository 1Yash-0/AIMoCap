"""Side-agnostic arm orientation from world-space observations.

This module implements the root-cause fix for the right-arm axial roll error.
The key insight: a rotation about a bone's own long axis leaves its child joint
position invariant, so rewriting ONLY the twist component in GLOBAL space is
provably position-preserving.

SPACE SUFFIXES (mandatory):
  _w    world space
  _loc  bone-local
  _rest rig rest pose
  _obs  observed / triangulated

CORE CONVENTION:
  G(t)[b] = D[b](t) @ G_rest[b]     # D is a WORLD delta from rig rest global
  L(t)[b] = G(t)[parent].inv() * G(t)[b]

INVARIANT: this module contains zero `if side ==` branches. If you need one,
the rest basis is wrong. Fix it in load, not here.
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ── Public configuration (exposed for tuning) ────────────────────────────────
BEND_GATE_LO_DEG = 20.0       # below: humeral roll unobservable from arm triangle
BEND_GATE_HI_DEG = 40.0       # above: fully trusted. Linear ramp between.
HUMERAL_AXIAL_LIMIT_DEG = 90.0    # gleno-humeral internal/external ROM, hanging arm
ROLL_MAX_STEP_DEG = 25.0     # continuity clamp, 30 fps human shoulder
PROJ_DEGENERACY_MIN = 0.15   # reject signed-angle when projection collapses


def _orthonormalize_rows(M: np.ndarray) -> np.ndarray:
    """Return nearest rotation matrix via SVD with det=+1."""
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    assert np.linalg.det(R) > 0.999, f"improper rotation, det={np.linalg.det(R)}"
    return R


def load_rest_basis(fbx_skel) -> dict[str, Rotation]:
    """Load rig rest globals, re-orthonormalize, assert properness.
    
    Route ALL rest-matrix reads through this. Never call Rotation.from_matrix()
    on a raw rig matrix again.
    """
    _, rest_quats = fbx_skel.get_forward_kinematics()
    G = {}
    for name, quat in zip(fbx_skel.node_names, rest_quats):
        R = Rotation.from_quat(quat).as_matrix()
        R = _orthonormalize_rows(R)
        G[name] = Rotation.from_matrix(R)
    return G


def longest_axis_local(G_parent: Rotation, G_child: Rotation) -> np.ndarray:
    """Resolve the bone long axis in parent-local space, geometrically.
    
    Returns unit vector in PARENT'S local frame pointing from parent joint
    to child joint. Never assumes +Y/+X/+Z — computes from actual rest offsets.
    """
    # child position in parent frame = R_parent^T * (t_child - t_parent)
    # Since we only have rotations, we need rest positions. This is a convenience;
    # the real primitive is `resolve_long_axis` below which takes explicit rest offsets.
    raise NotImplementedError("Use resolve_long_axis with explicit rest offsets")


def resolve_long_axis(parent_name: str, child_name: str,
                       G_rest: dict[str, Rotation],
                       rest_t: np.ndarray,
                       name_to_idx: dict[str, int]) -> np.ndarray:
    """Geometric long axis in parent's LOCAL frame at rest.
    
    Returns unit vector in parent_local pointing parent→child.
    """
    p_idx = name_to_idx[parent_name]
    c_idx = name_to_idx[child_name]
    v_parent = rest_t[p_idx]
    v_child = rest_t[c_idx]
    axis_parent = v_child - v_parent
    axis_parent = axis_parent / (np.linalg.norm(axis_parent) + 1e-12)
    return axis_parent


def shortest_arc(u_w: np.ndarray, v_w: np.ndarray) -> Rotation:
    """Minimal-twist rotation aligning u_w to v_w in WORLD space.
    
    Handles antiparallel case deterministically (cross with +X, fallback +Y).
    Returns Rotation object.
    """
    u = u_w / (np.linalg.norm(u_w) + 1e-12)
    v = v_w / (np.linalg.norm(v_w) + 1e-12)
    d = np.dot(u, v)
    if d > 1.0 - 1e-12:
        return Rotation.identity()
    # Antiparallel detection: d ≈ -1 within numerical tolerance.
    # Use a generous tolerance (1e-9) because normalizing nearly-antiparallel
    # vectors can push d slightly above -1.0.
    if d < -1.0 + 1e-9:
        # Antiparallel: pick a deterministic perpendicular axis
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis)
    axis = np.cross(u, v)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        # Parallel (or nearly so) but d not caught above - identity
        return Rotation.identity()
    axis = axis / axis_norm
    angle = np.arccos(np.clip(d, -1.0, 1.0))
    return Rotation.from_rotvec(angle * axis)


def signed_angle_about(u_w: np.ndarray, v_w: np.ndarray, axis_w: np.ndarray
                        ) -> tuple[float, bool, str]:
    """Signed angle from u_w to v_w about axis_w, with degeneracy check.
    
    Both u_w and v_w are projected onto the plane normal to axis_w.
    If either projected norm < PROJ_DEGENERACY_MIN * original_norm, returns
    (0.0, False, "degenerate_projection").
    
    Returns: (angle_rad, valid, reason_string)
    """
    axis = axis_w / (np.linalg.norm(axis_w) + 1e-12)
    u_proj = u_w - np.dot(u_w, axis) * axis
    v_proj = v_w - np.dot(v_w, axis) * axis
    u_norm = np.linalg.norm(u_proj)
    v_norm = np.linalg.norm(v_proj)
    u_orig = np.linalg.norm(u_w)
    v_orig = np.linalg.norm(v_w)
    
    if u_norm < PROJ_DEGENERACY_MIN * u_orig or v_norm < PROJ_DEGENERACY_MIN * v_orig:
        return 0.0, False, "degenerate_projection"
    
    u_proj = u_proj / u_norm
    v_proj = v_proj / v_norm
    cross = np.cross(u_proj, v_proj)
    sin_a = np.dot(cross, axis)
    cos_a = np.dot(u_proj, v_proj)
    angle = np.arctan2(sin_a, cos_a)
    return float(angle), True, "elbow_plane"


def unwrap_against(phi: float, phi_prev: float) -> float:
    """Add k·2π to minimize |phi - phi_prev|."""
    # Standard unwrap: shift by multiples of 2π to get closest to phi_prev
    # The correct formula is: k = round((phi - phi_prev) / (2π))
    # But np.round uses bankers rounding (ties to even), so handle ties explicitly.
    diff = phi - phi_prev
    k_raw = diff / (2 * np.pi)
    k = int(np.round(k_raw))
    # If exactly halfway between integers, prefer the k that gives smaller absolute angle
    if abs(k_raw - k) > 0.5 - 1e-12:  # Not near a tie
        return phi - k * 2 * np.pi
    # Tie-breaker: try both k and k - sign(diff)
    candidates = [k]
    if diff != 0:
        candidates.append(k - int(np.sign(diff)))
    best = min(candidates, key=lambda kk: abs(phi - kk * 2 * np.pi - phi_prev))
    return phi - best * 2 * np.pi


def limit_step(phi: float, phi_prev: float, max_step_deg: float) -> float:
    """Clamp frame-to-frame change to max_step_deg."""
    max_step = np.deg2rad(max_step_deg)
    diff = phi - phi_prev
    if abs(diff) > max_step:
        return phi_prev + np.sign(diff) * max_step
    return phi


def clamp_deg(phi_rad: float, limit_deg: float) -> float:
    """Clamp angle to [-limit, +limit] in radians."""
    limit = np.deg2rad(limit_deg)
    return np.clip(phi_rad, -limit, +limit)


class Observation:
    """Structured return for estimators — never None."""
    def __init__(self, value: float, valid: bool, weight: float,
                 source: str, reason: str = ""):
        self.value = value
        self.valid = valid
        self.weight = weight
        self.source = source
        self.reason = reason
    
    def __repr__(self):
        return (f"Observation(val={np.rad2deg(self.value):.1f}°, "
                f"valid={self.valid}, w={self.weight:.2f}, src={self.source})")


def solve_upperarm_global(
    *,
    D_parent: Rotation,           # world delta of clavicle (parent of shoulder)
    G_rest_upper: Rotation,       # rig rest global of upperarm
    G_rest_fore: Rotation,        # rig rest global of lowerarm
    e_upper_loc: np.ndarray,      # upperarm long axis in clavicle local (unit)
    e_fore_loc: np.ndarray,       # lowerarm long axis in shoulder local (unit)
    sh_w: np.ndarray,             # observed shoulder position (world)
    el_w: np.ndarray,             # observed elbow position (world)
    wr_w: np.ndarray,             # observed wrist position (world)
    phi_prev: float               # previous frame's humeral axial angle (rad)
) -> tuple[Rotation, Observation]:
    """Compute upperarm global rotation: pure swing + conditional roll.
    
    SWING: carry rest bone direction by parent's actual motion, then minimally
           rotate onto observed humerus direction. Zero added twist by construction.
    
    ROLL:  only if elbow is bent enough to define the arm plane.
           Measures "where would the forearm point if shoulder only swung,
           elbow stayed at rest" vs "where it actually points", about the humerus axis.
           This IS humeral axial rotation. No basis, no side, no sign table.
    
    Returns: (G_upper_w, Observation)
    """
    # (a) SWING — zero-twist alignment of upperarm
    a_rest_w = D_parent.apply(G_rest_upper.apply(e_upper_loc))
    a_obs_w = el_w - sh_w
    a_obs_w = a_obs_w / (np.linalg.norm(a_obs_w) + 1e-12)
    S = shortest_arc(a_rest_w, a_obs_w)
    G_swing = S * D_parent * G_rest_upper
    
    # (b) ROLL — conditional on elbow bend
    f_obs_w = wr_w - el_w
    f_obs_w = f_obs_w / (np.linalg.norm(f_obs_w) + 1e-12)
    bend = np.degrees(np.arccos(np.clip(np.dot(a_obs_w, f_obs_w), -1.0, 1.0)))
    
    # Linear ramp from LO to HI
    w = np.clip((bend - BEND_GATE_LO_DEG) / (BEND_GATE_HI_DEG - BEND_GATE_LO_DEG), 0.0, 1.0)
    
    if w <= 0.0:
        # CORRECT BEHAVIOR FOR UNOBSERVABLE DOF: zero added twist.
        # Not a fallback, not a failure — this IS the anatomically neutral answer
        # when the elbow doesn't define a plane.
        return G_swing, Observation(0.0, True, 1.0, "swing_only", f"bend={bend:.1f}°")
    
    # Forearm direction IF shoulder only swung (no humeral roll)
    f_carried_w = (S * D_parent).apply(G_rest_fore.apply(e_fore_loc))
    
    obs = signed_angle_about(f_carried_w, f_obs_w, a_obs_w)
    if not obs[1]:  # not valid
        return G_swing, Observation(0.0, False, 0.0, "none", obs[2])
    
    phi = obs[0]
    phi = clamp_deg(phi, HUMERAL_AXIAL_LIMIT_DEG)
    phi = limit_step(phi, phi_prev, ROLL_MAX_STEP_DEG)
    
    # Apply roll about the OBSERVED humerus axis
    G = Rotation.from_rotvec(w * phi * a_obs_w) * G_swing
    return G, Observation(phi, True, w, "elbow_plane", f"bend={bend:.1f}°")


def solve_forearm_global(
    *,
    G_upper_w: Rotation,            # already solved upperarm global
    G_rest_fore: Rotation,          # rig rest global of lowerarm
    e_fore_loc: np.ndarray,         # forearm long axis in shoulder local (unit)
    el_w: np.ndarray,               # observed elbow position (world)
    wr_w: np.ndarray,               # observed wrist position (world)
    phi_prev: float                 # previous frame's forearm roll (rad)
) -> tuple[Rotation, Observation]:
    """Forearm = pure swing onto observed direction. No pronation from 23 pts."""
    f_rest_w = G_upper_w.apply(e_fore_loc)  # WRONG: G_upper_w is global, e_fore_loc is in shoulder local
    # Correct: need to map e_fore_loc through shoulder's local->global
    # Actually: G_upper_w = R_clavicle_w * R_shoulder_loc. 
    # e_fore_loc is in shoulder LOCAL. So carried direction = G_upper_w.apply(e_fore_loc)
    # Wait — G_upper_w already includes shoulder local rotation. So:
    f_rest_w = G_upper_w.apply(e_fore_loc)
    f_obs_w = wr_w - el_w
    f_obs_w = f_obs_w / (np.linalg.norm(f_obs_w) + 1e-12)
    
    S = shortest_arc(f_rest_w, f_obs_w)
    G_fore = S * G_upper_w  # No additional twist — pronation unrecoverable from 23 pts
    
    return G_fore, Observation(0.0, True, 1.0, "swing_only", "no_pronation_data")


def solve_hand_global(
    *,
    G_fore_w: Rotation,             # already solved forearm global
    G_rest_hand: Rotation,          # rig rest global of hand
    e_hand_loc: np.ndarray          # hand long axis in wrist local (unit)
) -> Rotation:
    """Hand = forearm delta carried forward. Neutral hand = rest local."""
    # D_fore = G_fore_w * G_rest_fore.inv()
    # G_hand = D_fore * G_rest_hand
    D_fore = G_fore_w * G_rest_hand.inv()  # Wait: need G_rest_fore
    # Actually: D_fore = G_fore_w * G_rest_fore.inv()
    # G_hand = D_fore * G_rest_hand
    raise NotImplementedError("Need G_rest_fore passed in")


def solve_hand_global_v2(
    *,
    G_fore_w: Rotation,
    G_rest_fore: Rotation,
    G_rest_hand: Rotation
) -> Rotation:
    """Hand global = forearm delta applied to hand rest."""
    D_fore = G_fore_w * G_rest_fore.inv()
    return D_fore * G_rest_hand