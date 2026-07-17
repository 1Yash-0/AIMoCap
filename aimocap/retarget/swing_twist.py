"""Constrained swing-twist rotation solver.

Replaces free-roll ``Rotation.align_vectors`` in the IK chain walk. Given a
bone's rest direction and its desired direction, plus an optional roll-pinning
child direction, returns the rotation that maps the bone direction AND fixes
the roll from the child.

Pure numpy/scipy. No Blender, no ufbx.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v.copy()


def constrained_rotation(
    rest_dir: np.ndarray,
    desired_dir: np.ndarray,
    roll_child_rest: np.ndarray | None = None,
    roll_child_desired: np.ndarray | None = None,
) -> Rotation:
    """Rotation mapping rest_dir -> desired_dir, with roll fixed by child vectors.

    Algorithm (swing-twist decomposition in direction space):
      1. Swing = shortest rotation rest_dir -> desired_dir.
      2. If roll children given, apply swing to roll_child_rest, then compute the
         residual rotation ABOUT desired_dir that aligns the swung child onto
         roll_child_desired (project both into the plane perpendicular to
         desired_dir, take the in-plane angle). That residual is the twist.
      3. Result = twist * swing.
    If no roll children, return swing only (minimal roll = free, same as before).
    """
    rest = _normalize(np.asarray(rest_dir, dtype=np.float64))
    desired = _normalize(np.asarray(desired_dir, dtype=np.float64))

    # Swing: shortest rotation rest -> desired. align_vectors on a single vector
    # is well-defined and picks the minimal rotation (zero twist component).
    swing, _ = Rotation.align_vectors(desired.reshape(1, 3), rest.reshape(1, 3))

    if roll_child_rest is None or roll_child_desired is None:
        return swing

    cr = _normalize(np.asarray(roll_child_rest, dtype=np.float64))
    cd = _normalize(np.asarray(roll_child_desired, dtype=np.float64))

    # If the roll child is nearly parallel to the bone direction, it carries
    # no useful roll information (the projection into the perpendicular plane
    # is noise-dominated).  This happens when a limb is nearly straight: the
    # hand/foot is almost collinear with the shin/forearm, so projecting it
    # perpendicular to the bone gives a near-zero vector with an arbitrary
    # direction.  The resulting twist is garbage -- often 40-100 deg of
    # arbitrary roll that makes the limb "cave in" or the hand spin.  Skip
    # the twist and return swing-only when the roll child is within ~35 deg
    # of the bone (covers near-straight arms and legs; a bent limb at >35deg
    # has enough perpendicular signal for a reliable twist).
    if abs(np.dot(cd, desired)) > np.cos(np.deg2rad(35.0)):
        return swing

    swung_child = swing.apply(cr)
    # project both into plane perp to desired
    sc = swung_child - np.dot(swung_child, desired) * desired
    cd_p = cd - np.dot(cd, desired) * desired
    sc_n, cd_n = np.linalg.norm(sc), np.linalg.norm(cd_p)
    if sc_n < 0.05 or cd_n < 1e-9:
        return swing
    sc_u, cd_u = sc / sc_n, cd_p / cd_n
    # in-plane angle about desired axis
    cos_a = np.clip(np.dot(sc_u, cd_u), -1.0, 1.0)
    axis_angle = np.arctan2(np.dot(np.cross(sc_u, cd_u), desired), cos_a)
    twist = Rotation.from_rotvec(axis_angle * desired)
    return twist * swing
