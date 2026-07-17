"""Test that the IK solver fits the synthetic ground-truth data at < 0.5 cm.

The synthetic data is generated from the proxy skeleton's own FK, so a correct
solver should fit it to sub-cm. This tests the full pipeline: target
construction (with joint offsets), bone-length estimation, and the optimizer.

The pelvis/neck_01 targets are offset from mid-hip/mid-shoulder to the actual
joint positions (the joints are offset from the COCO midpoints by ~2.4 cm and
~7 cm in Manny). Without the offset, no FK state can place the pelvis joint at
mid-hip AND the hip children at their measured positions, making the targets
unsatisfiable. See mocap_ik.analytic_init for the offset implementation.
"""
from aimocap.retarget.ik_probe import run_probe


def test_solver_fits_synthetic_data():
    """The IK solver must fit the synthetic data at < 0.5 cm mean."""
    res = run_probe(verbose=False)
    assert res["mean_cm"] < 0.5, (
        f"IK mean residual {res['mean_cm']:.3f}cm exceeds 0.5cm bar. "
        f"Max {res['max_cm']:.3f}cm. Worst joints: "
        f"{sorted(res['per_joint_mean'].items(), key=lambda kv: -kv[1])[:5]}"
    )
