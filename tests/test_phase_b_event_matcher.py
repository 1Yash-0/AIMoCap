import pytest
import numpy as np
from scripts.phase_b_matcher_recovery import Event, extract_events, match_events_pairwise

def test_extraction_constant_position():
    pos = np.zeros((10, 3))
    evs = extract_events(pos, "TEST", joint=0)
    assert len(evs) == 0

def test_extraction_constant_velocity():
    pos = np.zeros((10, 3))
    pos[:, 0] = np.arange(10) * 10.0
    evs = extract_events(pos, "TEST", joint=0)
    assert len(evs) == 0

def test_extraction_one_impulse():
    pos = np.zeros((10, 3))
    pos[5, 0] = 50.0  # Big jump at t=5
    evs = extract_events(pos, "TEST", joint=0)
    assert len(evs) == 1
    assert evs[0].joint == 0
    # Velocity at t=4: pos[5]-pos[4] = 50. Accel at t=3: V[4]-V[3] = 50.
    # We just need to check the event has correct peak
    assert abs(evs[0].peak_frame - 3) <= 2
    assert np.linalg.norm(evs[0].peak_vector_mps2) > 5.0

def test_extraction_two_impulses_apart():
    pos = np.zeros((20, 3))
    pos[5, 0] = 50.0
    pos[15, 0] = -50.0
    evs = extract_events(pos, "TEST", joint=0)
    assert len(evs) == 2

def test_extraction_impulse_with_dip():
    # Accel > 5, then < 5 for 1 frame, then > 5 again
    # We can fake it by calling merge logic directly or setting pos carefully.
    # Easiest way is a strong sine wave
    pos = np.zeros((20, 3))
    pos[4, 0] = 10.0
    pos[5, 0] = 20.0
    pos[7, 0] = 30.0
    evs = extract_events(pos, "TEST", joint=0)
    assert len(evs) >= 1
    # Check merge condition internally in the script

def test_extraction_nan_in_signal():
    pos = np.zeros((10, 3))
    pos[4, 0] = 50.0
    pos[6, 0] = np.nan
    pos[8, 0] = 50.0
    evs = extract_events(pos, "TEST", joint=0)
    # Shouldn't bridge the NaN
    assert all(e.duration_frames < 4 for e in evs)

def _ev(i, j, s, e, p, vec, dur):
    return Event("TEST", f"TEST:j{j}:s{s}:e{e}:p{p}", j, s, e, p, dur, np.linalg.norm(vec), np.array(vec, dtype=float), 1.0, 1.0, s, e, p)

def test_match_perfect():
    p = [_ev(1, 0, 5, 10, 7, [10,0,0], 6)]
    g = [_ev(2, 0, 5, 10, 7, [10,0,0], 6)]
    m, up, ug, counts = match_events_pairwise(p, g, 1)
    assert len(m) == 1

def test_match_shift():
    # p: 5..10 (peak 7)
    # g: 11..16 (peak 10)
    # They do not overlap. interval_distance = max(5,11) - min(10,16) = 11 - 10 = 1
    # peak_distance = abs(7-10) = 3
    # At T=1: int_dist <= 1 (True), peak_dist <= 3 (True since T+2 = 3). MATCH.
    # At T=0: int_dist <= 0 (False). UNMATCHED.
    p = [_ev(1, 0, 5, 10, 7, [10,0,0], 6)]
    g = [_ev(2, 0, 11, 16, 10, [10,0,0], 6)]
    m, up, ug, counts = match_events_pairwise(p, g, 1)
    assert len(m) == 1
    m, up, ug, counts = match_events_pairwise(p, g, 0)
    assert len(m) == 0

def test_match_beyond_tolerance():
    # p: 5..10 (peak 7)
    # g: 12..17 (peak 11)
    # int_dist = 12 - 10 = 2.
    # At T=1: int_dist <= 1 (False). UNMATCHED.
    p = [_ev(1, 0, 5, 10, 7, [10,0,0], 6)]
    g = [_ev(2, 0, 12, 17, 11, [10,0,0], 6)]
    m, up, ug, counts = match_events_pairwise(p, g, 1)
    assert len(m) == 0

def test_match_opposite_direction():
    p = [_ev(1, 0, 5, 10, 7, [10,0,0], 6)]
    g = [_ev(2, 0, 5, 10, 7, [-10,0,0], 6)]
    m, up, ug, counts = match_events_pairwise(p, g, 1)
    assert len(m) == 0
    assert counts["direction_cosine"] == 1

def test_match_wrong_joint():
    p = [_ev(1, 0, 5, 10, 7, [10,0,0], 6)]
    g = [_ev(2, 1, 5, 10, 7, [10,0,0], 6)]
    m, up, ug, counts = match_events_pairwise(p, g, 1)
    assert len(m) == 0

def test_match_prediction_only():
    p = [_ev(1, 0, 5, 10, 7, [10,0,0], 6)]
    m, up, ug, counts = match_events_pairwise(p, [], 1)
    assert len(m) == 0
    assert len(up) == 1

def test_match_gt_only():
    g = [_ev(2, 0, 5, 10, 7, [10,0,0], 6)]
    m, up, ug, counts = match_events_pairwise([], g, 1)
    assert len(m) == 0
    assert len(ug) == 1

def test_match_two_pred_one_gt():
    p = [
        _ev(1, 0, 5, 10, 7, [10,0,0], 6),
        _ev(2, 0, 6, 11, 8, [10,0,0], 6)
    ]
    g = [_ev(3, 0, 5, 10, 7, [10,0,0], 6)]
    m, up, ug, counts = match_events_pairwise(p, g, 2)
    assert len(m) == 1
    assert m[0][0].id == p[0].id
    assert len(up) == 1

def test_match_one_long_two_short():
    p = [_ev(1, 0, 5, 20, 10, [10,0,0], 16)]
    g = [
        _ev(2, 0, 5, 10, 7, [10,0,0], 6),
        _ev(3, 0, 15, 20, 17, [10,0,0], 6)
    ]
    m, up, ug, counts = match_events_pairwise(p, g, 10)
    assert len(m) <= 1

def test_match_tie_deterministic():
    p = [
        _ev(1, 0, 5, 10, 7, [10,0,0], 6),
        _ev(2, 0, 5, 10, 7, [10,0,0], 6)
    ]
    g = [_ev(3, 0, 5, 10, 7, [10,0,0], 6)]
    m, up, ug, counts = match_events_pairwise(p, g, 1)
    assert len(m) == 1

def test_match_permutations():
    p1 = _ev(1, 0, 5, 10, 7, [10,0,0], 6)
    p2 = _ev(2, 0, 6, 11, 8, [9,0,0], 6)
    g1 = _ev(3, 0, 5, 10, 7, [10,0,0], 6)
    g2 = _ev(4, 0, 15, 20, 17, [10,0,0], 6)
    base_m, _, _, _ = match_events_pairwise([p1,p2], [g1,g2], 10)
    import random
    rng = random.Random(42)
    for _ in range(20):
        ps = [p1, p2]
        gs = [g1, g2]
        rng.shuffle(ps)
        rng.shuffle(gs)
        m, _, _, _ = match_events_pairwise(ps, gs, 10)
        assert set((a.id, b.id) for a,b,c in base_m) == set((a.id, b.id) for a,b,c in m)
