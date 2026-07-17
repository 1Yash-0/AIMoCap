from aimocap.retarget.ik_probe import run_probe


def test_ik_residual_under_bar():
    res = run_probe(verbose=False)
    assert res["mean_cm"] < 0.5, (
        f"IK mean residual {res['mean_cm']:.3f}cm exceeds 0.5cm bar. "
        f"Worst joints: {sorted(res['per_joint_mean'].items(), key=lambda kv: -kv[1])[:3]}"
    )
