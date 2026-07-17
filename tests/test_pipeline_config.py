import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from scripts.stage6a_kinematic_bvh import (
    DEFAULT_PIPELINE_CONFIG,
    experimental_candidate_c_config,
    resolve_pipeline_config,
    preprocess_observations_for_config,
    solve_ankles_for_config
)
import subprocess
import sys

def test_default_config_resolves_to_candidate_b():
    config = resolve_pipeline_config(DEFAULT_PIPELINE_CONFIG)
    assert config["ankle_strategy"] == "ray_sphere_with_fk_fallback"
    assert config["boundary_rejection_gate"] is False

@pytest.mark.parametrize("missing_key", [
    "ankle_strategy", "boundary_rejection_gate", "boundary_margin_px", "image_width_px", "image_height_px"
])
def test_missing_each_required_key_fails_closed(missing_key):
    config = dict(DEFAULT_PIPELINE_CONFIG)
    del config[missing_key]
    with pytest.raises(KeyError, match=missing_key):
        resolve_pipeline_config(config)

def test_gate_requires_boolean():
    config = dict(DEFAULT_PIPELINE_CONFIG)
    config["boundary_rejection_gate"] = "False"  # string, not bool
    with pytest.raises(TypeError, match="must be bool"):
        resolve_pipeline_config(config)

def test_invalid_dimensions_and_margin_fail():
    config = dict(DEFAULT_PIPELINE_CONFIG)
    
    config["image_width_px"] = -10
    with pytest.raises(ValueError): resolve_pipeline_config(config)
    
    config["image_width_px"] = 1920
    config["image_height_px"] = 0
    with pytest.raises(ValueError): resolve_pipeline_config(config)

    config["image_height_px"] = 1080
    config["boundary_margin_px"] = -5
    with pytest.raises(ValueError): resolve_pipeline_config(config)

    config["boundary_margin_px"] = 1000  # 2 * 1000 >= 1080
    with pytest.raises(ValueError, match="margin too large"): resolve_pipeline_config(config)

def test_default_runtime_does_not_call_boundary_gate():
    scores2d = np.ones((10, 3, 17))
    kpts2d = np.ones((10, 3, 17, 2)) * 500
    mock_gate_fn = MagicMock()
    
    preprocess_observations_for_config(scores2d, kpts2d, DEFAULT_PIPELINE_CONFIG, boundary_gate_fn=mock_gate_fn)
    mock_gate_fn.assert_not_called()

def test_explicit_experimental_runtime_calls_boundary_gate_once():
    scores2d = np.ones((10, 3, 17))
    kpts2d = np.ones((10, 3, 17, 2)) * 500
    mock_gate_fn = MagicMock()
    
    exp_config = experimental_candidate_c_config(boundary_margin_px=45)
    preprocess_observations_for_config(scores2d, kpts2d, exp_config, boundary_gate_fn=mock_gate_fn)
    
    mock_gate_fn.assert_called_once_with(
        scores2d, kpts2d, image_height_px=1080, image_width_px=1920, margin_px=45
    )

@patch("scripts.stage6a_kinematic_bvh.infer_by_ray_sphere")
def test_default_runtime_dispatches_to_ray_sphere_fk_solver(mock_solver):
    mock_solver.return_value = ("mock_pts3d", "mock_stats")
    
    pts3d_clean = np.ones((10, 17, 3))
    ankle_bl = {15: 10, 16: 10}
    calib = {}
    kpts2d = np.ones((10, 3, 17, 2))
    scores2d = np.ones((10, 3, 17))
    ankle_gates = {}
    ankle_joint_defs = {}
    
    result = solve_ankles_for_config(
        pts3d_clean, ankle_bl, calib, kpts2d, scores2d, ankle_gates, ankle_joint_defs, DEFAULT_PIPELINE_CONFIG
    )
    
    mock_solver.assert_called_once_with(pts3d_clean, ankle_bl, calib, kpts2d, scores2d, ankle_gates, ankle_joint_defs)
    assert result == ("mock_pts3d", "mock_stats")

@patch("scripts.stage6a_kinematic_bvh.infer_by_ray_sphere")
def test_unsupported_ankle_strategy_fails_before_solver_call(mock_solver):
    config = dict(DEFAULT_PIPELINE_CONFIG)
    config["ankle_strategy"] = "invalid_strategy"
    
    with pytest.raises(ValueError, match="Unsupported ankle strategy"):
        solve_ankles_for_config(
            None, None, None, None, None, None, None, config
        )
    mock_solver.assert_not_called()

def test_default_config_is_not_mutated():
    # Due to MappingProxyType this will raise TypeError if one tries to assign to it
    with pytest.raises(TypeError):
        DEFAULT_PIPELINE_CONFIG["boundary_rejection_gate"] = True

def test_import_has_no_pipeline_execution_side_effect():
    cmd = [sys.executable, "-c", "import scripts.stage6a_kinematic_bvh; print('IMPORT_OK')"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "IMPORT_OK" in result.stdout
    assert "Stage 6a" not in result.stdout  # Ensure main() print output is not present
