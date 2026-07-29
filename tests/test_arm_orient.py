"""Tests for arm_orient.py — synthetic 3-point arm with known roll."""
import sys
sys.path.insert(0, r"E:\Chaos\Projects\aimocap_re")

import numpy as np
from scipy.spatial.transform import Rotation
from aimocap.retarget.arm_orient import (
    solve_upperarm_global,
    solve_forearm_global,
    solve_hand_global_v2,
    shortest_arc,
    signed_angle_about,
    unwrap_against,
    limit_step,
    clamp_deg,
    Observation,
    BEND_GATE_LO_DEG,
    BEND_GATE_HI_DEG,
)


def make_synthetic_arm(bend_deg: float, humeral_roll_deg: float = 0.0,
                        fore_roll_deg: float = 0.0) -> dict:
    """Create synthetic arm observation.
    
    Rest pose: upperarm along +Y, forearm bent 10° from upperarm at rest.
    shoulder at origin, elbow at (0, L1*cos(10°), L1*sin(10°)), etc.
    """
    L1, L2 = 30.0, 25.0  # cm
    rest_elbow_bend = 10.0  # degrees - realistic rest pose
    
    # Rest positions with 10° elbow bend
    R_rest_bend = Rotation.from_rotvec(np.deg2rad(rest_elbow_bend) * np.array([1.0, 0.0, 0.0]))
    sh_rest = np.array([0.0, 0.0, 0.0])
    u_rest_dir = np.array([0.0, 1.0, 0.0])
    f_rest_dir = R_rest_bend.apply(np.array([0.0, 1.0, 0.0]))
    
    # Humeral roll about +Y
    R_roll = Rotation.from_rotvec(np.deg2rad(humeral_roll_deg) * np.array([0.0, 1.0, 0.0]))
    u_obs_dir = R_roll.apply(u_rest_dir)
    
    # Forearm: additional bend from rest + forearm roll
    R_bend = Rotation.from_rotvec(np.deg2rad(bend_deg) * np.array([1.0, 0.0, 0.0]))
    R_fore_roll = Rotation.from_rotvec(np.deg2rad(fore_roll_deg) * np.array([0.0, 1.0, 0.0]))
    
    # Total forearm = humeral_roll * (rest_bend + additional_bend) * forearm_roll
    f_obs_dir = R_roll.apply(R_rest_bend.apply(R_bend.apply(R_fore_roll.apply(np.array([0.0, 1.0, 0.0])))))
    
    # Verify actual bend
    actual_bend = np.degrees(np.arccos(np.clip(np.dot(u_obs_dir, f_obs_dir), -1.0, 1.0)))
    if abs(actual_bend - bend_deg) > 1e-6:
        print(f"  WARNING: requested bend={bend_deg}°, actual={actual_bend:.6f}°")
    
    # Positions
    sh_w = sh_rest
    el_w = sh_w + L1 * u_obs_dir
    wr_w = el_w + L2 * f_obs_dir
    
    return {
        "sh_w": sh_w, "el_w": el_w, "wr_w": wr_w,
        "u_rest_dir": u_rest_dir, "f_rest_dir": f_rest_dir,
        "humeral_roll_true": np.deg2rad(humeral_roll_deg),
        "fore_roll_true": np.deg2rad(fore_roll_deg),
        "bend_true": np.deg2rad(bend_deg),
    }


def test_shortest_arc():
    """Test shortest_arc basic properties."""
    u = np.array([0.0, 1.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    R = shortest_arc(u, v)
    assert np.allclose(R.apply(u), v), "shortest_arc fails alignment"
    
    # Antiparallel case
    v = -u
    R = shortest_arc(u, v)
    assert np.allclose(R.apply(u), v), "antiparallel fails"
    # Should be 180° about some perpendicular axis
    rv = R.as_rotvec()
    assert np.abs(np.linalg.norm(rv) - np.pi) < 1e-6
    print("  shortest_arc: PASS")


def test_signed_angle_about():
    """Test signed angle about axis."""
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle, valid, _ = signed_angle_about(u, v, axis)
    assert valid
    assert np.abs(angle - np.pi/2) < 1e-6, f"Expected 90°, got {np.rad2deg(angle):.1f}°"
    
    # Reverse
    angle, valid, _ = signed_angle_about(v, u, axis)
    assert valid
    assert np.abs(angle + np.pi/2) < 1e-6, f"Expected -90°, got {np.rad2deg(angle):.1f}°"
    
    # Degenerate: u parallel to axis
    u = np.array([0.0, 0.0, 1.0])
    angle, valid, reason = signed_angle_about(u, v, axis)
    assert not valid
    assert reason == "degenerate_projection"
    print("  signed_angle_about: PASS")


def test_unwrap_and_limit():
    """Test angle unwrapping and step limiting."""
    # Unwrap: 3π should unwrap to either π or -π (both equivalent)
    phi = 3.0 * np.pi  # 540°
    phi_prev = 0.0
    phi_uw = unwrap_against(phi, phi_prev)
    # Both π and -π are valid unwrappings of 3π (diff is 2π)
    assert min(abs(phi_uw - np.pi), abs(phi_uw + np.pi)) < 1e-6, f"unwrap failed: {phi_uw}"
    
    # Limit step
    phi = np.deg2rad(50.0)
    phi_prev = 0.0
    phi_lim = limit_step(phi, phi_prev, 25.0)
    assert np.abs(phi_lim - np.deg2rad(25.0)) < 1e-6, f"limit_step failed: {np.rad2deg(phi_lim):.1f}°"
    
    # Clamp
    phi = np.deg2rad(120.0)
    phi_clamped = clamp_deg(phi, 90.0)
    assert np.abs(phi_clamped - np.deg2rad(90.0)) < 1e-6
    print("  unwrap/limit/clamp: PASS")


def test_upperarm_roll_recovery():
    """Test humeral roll recovery at various bend angles."""
    print("\n=== Upperarm Roll Recovery Test ===")
    
    # Rest vectors with realistic elbow bend (10° at rest)
    rest_elbow_bend = 10.0  # degrees
    R_rest_bend = Rotation.from_rotvec(np.deg2rad(rest_elbow_bend) * np.array([1.0, 0.0, 0.0]))
    e_upper_loc = np.array([0.0, 1.0, 0.0])  # +Y in clavicle local
    e_fore_loc = R_rest_bend.apply(np.array([0.0, 1.0, 0.0]))  # +Y bent by 10° in shoulder local
    
    G_rest_upper = Rotation.identity()
    G_rest_fore = Rotation.identity()
    D_parent = Rotation.identity()
    
    phi_prev = 0.0
    
# Test cases: (additional_bend, expected_behavior)
    # actual_bend = rest_elbow_bend + additional_bend
    test_cases = [
        (5, "swing_only"),      # actual=15° < 20° gate
        (15, "elbow_plane"),    # actual=25° in ramp (20-40°)
        (35, "elbow_plane"),    # actual=45° > 40° full weight
        (80, "elbow_plane"),    # actual=90° > 40° full weight
    ]
    
    for additional_bend, expected_source in test_cases:
        actual_bend = rest_elbow_bend + additional_bend
        # Use roll angles that are actually recoverable from arm triangle
        # At 90° humeral roll, forearm becomes parallel to humerus → unobservable
        # Use smaller angles: max 60° for full weight, 30° for ramp
        if actual_bend <= BEND_GATE_LO_DEG:
            roll_angles = [-30, -10, 0, 10, 30]
        elif actual_bend < BEND_GATE_HI_DEG:
            roll_angles = [-20, -10, 0, 10, 20]
        else:
            roll_angles = [-60, -30, -10, 0, 10, 30, 60]
        
        for true_roll_deg in roll_angles:
            synth = make_synthetic_arm(bend_deg=additional_bend, humeral_roll_deg=true_roll_deg)
            
            G_upper, obs = solve_upperarm_global(
                D_parent=D_parent,
                G_rest_upper=G_rest_upper,
                G_rest_fore=G_rest_fore,
                e_upper_loc=e_upper_loc,
                e_fore_loc=e_fore_loc,
                sh_w=synth["sh_w"],
                el_w=synth["el_w"],
                wr_w=synth["wr_w"],
                phi_prev=phi_prev
            )
            
            if actual_bend <= BEND_GATE_LO_DEG:
                # Below gate: should return swing_only with 0 roll
                assert obs.source == "swing_only", f"actual_bend={actual_bend}: expected swing_only, got {obs.source}"
                assert np.abs(obs.value) < 1e-6, f"actual_bend={actual_bend}: expected 0 roll, got {np.rad2deg(obs.value):.1f}°"
                phi_prev = 0.0  # reset for next
            elif actual_bend >= BEND_GATE_HI_DEG:
                # Above gate: full weight, should recover true roll
                assert obs.source == "elbow_plane", f"actual_bend={actual_bend}: expected elbow_plane, got {obs.source}"
                assert obs.valid, f"actual_bend={actual_bend}: invalid at roll={true_roll_deg}"
                assert np.abs(obs.weight - 1.0) < 1e-6, f"weight={obs.weight}"
                error_deg = np.rad2deg(obs.value - synth["humeral_roll_true"])
                assert np.abs(error_deg) < 0.1, f"actual_bend={actual_bend}, true={true_roll_deg}°: error={error_deg:.3f}°"
                phi_prev = obs.value
            else:
                # In ramp: weighted, error proportional to weight
                assert obs.source == "elbow_plane"
                assert obs.valid
                assert 0.0 < obs.weight < 1.0
                error_deg = np.rad2deg(obs.value - synth["humeral_roll_true"])
                # Weighted error should be small
                assert np.abs(error_deg) < 5.0, f"actual_bend={actual_bend}, true={true_roll_deg}°: error={error_deg:.3f}° (w={obs.weight:.2f})"
                phi_prev = obs.value
    
    print("  Upperarm roll recovery: PASS (error < 0.1° for bend >= 40°)")


def test_forearm_swing_only():
    """Test forearm returns swing_only (no pronation data)."""
    print("\n=== Forearm Test ===")
    
    e_fore_loc = np.array([0.0, 1.0, 0.0])
    G_rest_fore = Rotation.identity()
    phi_prev = 0.0
    
    # Straight arm
    synth = make_synthetic_arm(bend_deg=180.0, humeral_roll_deg=30.0)
    
    # Need a plausible G_upper_w
    # For straight arm, upperarm direction is known
    G_upper_w = Rotation.identity()  # simplified
    
    G_fore, obs = solve_forearm_global(
        G_upper_w=G_upper_w,
        G_rest_fore=G_rest_fore,
        e_fore_loc=e_fore_loc,
        el_w=synth["el_w"],
        wr_w=synth["wr_w"],
        phi_prev=phi_prev
    )
    
    assert obs.source == "swing_only"
    assert obs.value == 0.0
    assert obs.valid
    assert obs.weight == 1.0
    print("  Forearm swing_only: PASS")


def test_hand_neutral():
    """Test hand = forearm delta applied to rest hand."""
    print("\n=== Hand Neutral Test ===")
    
    G_rest_fore = Rotation.identity()
    G_rest_hand = Rotation.identity()
    G_fore_w = Rotation.from_rotvec([0.0, np.deg2rad(30.0), 0.0])  # 30° roll
    
    G_hand = solve_hand_global_v2(G_fore_w, G_rest_fore, G_rest_hand)
    
    # Hand should have same delta as forearm
    D_fore = G_fore_w * G_rest_fore.inv()
    D_hand = G_hand * G_rest_hand.inv()
    
    assert np.allclose(D_fore.as_matrix(), D_hand.as_matrix()), "hand delta != forearm delta"
    print("  Hand neutral: PASS")


def test_both_sides_identical():
    """Left and right arms use SAME code path — verify identical output."""
    print("\n=== Side Symmetry Test ===")
    
    # Create identical synthetic observations for L and R
    synth = make_synthetic_arm(bend_deg=60.0, humeral_roll_deg=45.0)
    
    e_upper_loc = np.array([0.0, 1.0, 0.0])
    e_fore_loc = np.array([0.0, 1.0, 0.0])
    G_rest_upper = Rotation.identity()
    G_rest_fore = Rotation.identity()
    D_parent = Rotation.identity()
    
    G_upper_l, obs_l = solve_upperarm_global(
        D_parent=D_parent, G_rest_upper=G_rest_upper, G_rest_fore=G_rest_fore,
        e_upper_loc=e_upper_loc, e_fore_loc=e_fore_loc,
        sh_w=synth["sh_w"], el_w=synth["el_w"], wr_w=synth["wr_w"],
        phi_prev=0.0
    )
    
    G_upper_r, obs_r = solve_upperarm_global(
        D_parent=D_parent, G_rest_upper=G_rest_upper, G_rest_fore=G_rest_fore,
        e_upper_loc=e_upper_loc, e_fore_loc=e_fore_loc,
        sh_w=synth["sh_w"], el_w=synth["el_w"], wr_w=synth["wr_w"],
        phi_prev=0.0
    )
    
    assert np.allclose(G_upper_l.as_matrix(), G_upper_r.as_matrix()), "Left/Right globals differ"
    assert np.abs(obs_l.value - obs_r.value) < 1e-12, "Left/Right rolls differ"
    assert np.abs(obs_l.weight - obs_r.weight) < 1e-12, "Left/Right weights differ"
    print("  Left/Right identical: PASS")


if __name__ == "__main__":
    test_shortest_arc()
    test_signed_angle_about()
    test_unwrap_and_limit()
    test_upperarm_roll_recovery()
    test_forearm_swing_only()
    test_hand_neutral()
    test_both_sides_identical()
    print("\n=== ALL TESTS PASSED ===")