import numpy as np
from aimocap.retarget.spine_chain import distribute_spine_targets


def test_two_segments_split_midpoint():
    pelvis = np.array([0.0, 0.0, 0.0])
    neck = np.array([0.0, 0.0, 10.0])
    # proportions: 2 segment lengths [4, 6]
    targets = distribute_spine_targets(pelvis, neck, np.array([4.0, 6.0]))
    assert targets.shape == (1, 3)  # one intermediate joint
    assert np.allclose(targets[0], [0.0, 0.0, 4.0], atol=1e-9)


def test_three_segments_proportional():
    pelvis = np.array([0.0, 0.0, 0.0])
    neck = np.array([0.0, 0.0, 12.0])
    targets = distribute_spine_targets(pelvis, neck, np.array([3.0, 6.0, 3.0]))
    assert targets.shape == (2, 3)
    assert np.allclose(targets[0], [0.0, 0.0, 3.0], atol=1e-9)
    assert np.allclose(targets[1], [0.0, 0.0, 9.0], atol=1e-9)


def test_bent_spine_interpolates_along_line():
    pelvis = np.array([0.0, 0.0, 0.0])
    neck = np.array([6.0, 0.0, 8.0])  # diagonal
    targets = distribute_spine_targets(pelvis, neck, np.array([5.0, 5.0]))
    # midpoint of the line
    assert np.allclose(targets[0], [3.0, 0.0, 4.0], atol=1e-9)


def test_single_segment_returns_empty():
    pelvis = np.array([0.0, 0.0, 0.0])
    neck = np.array([0.0, 0.0, 10.0])
    targets = distribute_spine_targets(pelvis, neck, np.array([10.0]))
    assert targets.shape == (0, 3)


def test_degenerate_zero_length_fallback():
    pelvis = np.array([0.0, 0.0, 0.0])
    neck = np.array([0.0, 0.0, 9.0])
    targets = distribute_spine_targets(pelvis, neck, np.array([0.0, 0.0, 0.0]))
    # should fall back to even spacing -> 2 intermediates at 1/3 and 2/3
    assert targets.shape == (2, 3)
    assert np.allclose(targets[0], [0.0, 0.0, 3.0], atol=1e-9)
    assert np.allclose(targets[1], [0.0, 0.0, 6.0], atol=1e-9)
