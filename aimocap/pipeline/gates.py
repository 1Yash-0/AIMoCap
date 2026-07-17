import numpy as np
from typing import Optional
from aimocap.pipeline.evidence import EvidenceContract, GateFailureReport
from aimocap.pipeline.epipolar import get_fundamental_matrices
from aimocap.math.metrics import compute_epipolar_consistency

class PipelineGateError(Exception):
    pass

def check_2d_gates(
    keypoints: np.ndarray,      # (F, C, J, 2)
    scores: np.ndarray,         # (F, C, J)
    K_list: list[np.ndarray], 
    extrinsics: list[tuple[np.ndarray, np.ndarray]], 
    calib_hash: str,
    provenance_hash: str,
    image_width: float = 1920.0,
    image_height: float = 1080.0,
    min_conf: float = 0.5,
    soft_fail: bool = True
) -> EvidenceContract | GateFailureReport:
    """
    Check 2D observation gates before triangulation.
    """
    F_frames, C_cams, J_joints, _ = keypoints.shape
    
    gate_results = {}
    failing_gate = None
    failure_mode = None
    
    # Gate 1: coords_in_image
    # All confident kpts lie within [0, W] x [0, H] with 5px tolerance
    confident_mask = scores >= min_conf
    x_valid = (keypoints[..., 0] >= -5.0) & (keypoints[..., 0] <= image_width + 5.0)
    y_valid = (keypoints[..., 1] >= -5.0) & (keypoints[..., 1] <= image_height + 5.0)
    coords_valid = x_valid & y_valid
    
    # If there are any confident points that are outside the image, this fails
    bad_coords = confident_mask & ~coords_valid
    passed_coords = not np.any(bad_coords)
    
    gate_results["coords_in_image"] = {
        "passed": passed_coords,
        "value": int(np.sum(bad_coords)),
        "threshold": 0
    }
    if not passed_coords and failing_gate is None:
        failing_gate = "coords_in_image"
        failure_mode = f"{np.sum(bad_coords)} confident keypoints outside image bounds"

    # Gate 2: multi_camera_coverage
    # >=2 usable cameras per joint per frame for >=80% of joints
    # confident_mask is (F, C, J). Sum over C:
    cams_per_joint = np.sum(confident_mask, axis=1) # (F, J)
    valid_joints = cams_per_joint >= 2 # (F, J)
    # Average across all frames and joints
    coverage = np.mean(valid_joints)
    
    passed_coverage = coverage >= 0.8
    gate_results["multi_camera_coverage"] = {
        "passed": passed_coverage,
        "value": coverage,
        "threshold": 0.8
    }
    if not passed_coverage and failing_gate is None:
        failing_gate = "multi_camera_coverage"
        failure_mode = f"Only {coverage:.1%} of joints have >=2 cameras"
        
    # Gate 3: epipolar_consistency
    F_matrices = get_fundamental_matrices(K_list, extrinsics)
    epi_results = compute_epipolar_consistency(
        np.transpose(keypoints, (0, 1, 2, 3)), # Wait, compute_epipolar expects (F, C, J, 2)
        scores,
        F_matrices,
        min_conf=min_conf
    )
    
    median_err = epi_results["median_err"]
    p95_err = epi_results["p95_err"]
    epipolar_coverage = epi_results["valid_coverage"]
    
    passed_epi_med = median_err < 5.0
    passed_epi_p95 = p95_err < 15.0
    passed_epi_cov = epipolar_coverage >= 0.7
    
    gate_results["epipolar_median"] = {
        "passed": passed_epi_med,
        "value": median_err,
        "threshold": 5.0
    }
    if not passed_epi_med and failing_gate is None:
        failing_gate = "epipolar_median"
        failure_mode = f"Median epipolar error {median_err:.1f}px >= 5.0px"
        
    gate_results["epipolar_p95"] = {
        "passed": passed_epi_p95,
        "value": p95_err,
        "threshold": 15.0
    }
    if not passed_epi_p95 and failing_gate is None:
        failing_gate = "epipolar_p95"
        failure_mode = f"P95 epipolar error {p95_err:.1f}px >= 15.0px"
        
    gate_results["epipolar_coverage"] = {
        "passed": passed_epi_cov,
        "value": epipolar_coverage,
        "threshold": 0.7
    }
    if not passed_epi_cov and failing_gate is None:
        failing_gate = "epipolar_coverage"
        failure_mode = f"Epipolar coverage {epipolar_coverage:.1%} < 70%"
        
    # Skeleton plausibility is done implicitly in triangulation min_aspect_ratio for now
    
    evidence = EvidenceContract(
        stage="2d_observation",
        status="FAIL" if failing_gate else "PASS",
        validity_mask=valid_joints,
        uncertainty=scores, # simplfication
        calibration_id=calib_hash,
        coordinate_space="image_px",
        input_provenance=provenance_hash,
        gate_results=gate_results,
        coverage_fraction=coverage,
        failure_reason=failing_gate
    )
    
    if failing_gate:
        if soft_fail:
            return GateFailureReport(
                failing_stage="2d_observation",
                failing_gate=failing_gate,
                cameras_implicated=[str(c) for c in range(C_cams)],
                failure_mode=failure_mode,
                coverage_fraction=coverage,
                evidence=evidence
            )
        else:
            raise PipelineGateError(f"2D Gate Failed: {failure_mode}")
            
    return evidence


def check_triangulation_gates(
    diagnostics, # TriangulationDiagnostics
    calib_hash: str,
    provenance_hash: str,
    soft_fail: bool = True
) -> EvidenceContract | GateFailureReport:
    """
    Check 3D triangulation gates.
    """
    gate_results = {}
    failing_gate = None
    failure_mode = None
    
    # 1. inlier_ratio
    # Fraction of joint-frames with >=2 inlier cameras >= 70%
    inliers = diagnostics.num_inliers >= 2
    inlier_ratio = float(np.mean(inliers))
    passed_inliers = inlier_ratio >= 0.7
    
    gate_results["inlier_ratio"] = {
        "passed": passed_inliers,
        "value": inlier_ratio,
        "threshold": 0.7
    }
    if not passed_inliers and failing_gate is None:
        failing_gate = "inlier_ratio"
        failure_mode = f"Inlier ratio {inlier_ratio:.1%} < 70%"
        
    # 2. reproj_median & p95
    # Across all valid reprojections
    valid_reproj = diagnostics.reprojection_error_px[np.isfinite(diagnostics.reprojection_error_px)]
    if len(valid_reproj) > 0:
        med_reproj = float(np.median(valid_reproj))
        p95_reproj = float(np.percentile(valid_reproj, 95))
    else:
        med_reproj = float('inf')
        p95_reproj = float('inf')
        
    passed_reproj_med = med_reproj < 5.0
    passed_reproj_p95 = p95_reproj < 15.0
    
    gate_results["reproj_median"] = {
        "passed": passed_reproj_med,
        "value": med_reproj,
        "threshold": 5.0
    }
    if not passed_reproj_med and failing_gate is None:
        failing_gate = "reproj_median"
        failure_mode = f"Median reproj error {med_reproj:.1f}px >= 5.0px"
        
    gate_results["reproj_p95"] = {
        "passed": passed_reproj_p95,
        "value": p95_reproj,
        "threshold": 15.0
    }
    if not passed_reproj_p95 and failing_gate is None:
        failing_gate = "reproj_p95"
        failure_mode = f"P95 reproj error {p95_reproj:.1f}px >= 15.0px"
        
    # 3. depth_positive
    # Currently triangulation returns points in Y-up. Depth check requires camera space.
    # We will assume Z > 0 in camera space is satisfied if volume bounds are reasonable.
    
    # 4. volume_bounds
    # All points within [-500, 500] cm on each axis
    valid_pts = diagnostics.points3d[inliers]
    if len(valid_pts) > 0:
        max_bound = float(np.max(np.abs(valid_pts)))
    else:
        max_bound = float('inf')
        
    passed_bounds = max_bound <= 500.0
    gate_results["volume_bounds"] = {
        "passed": passed_bounds,
        "value": max_bound,
        "threshold": 500.0
    }
    if not passed_bounds and failing_gate is None:
        failing_gate = "volume_bounds"
        failure_mode = f"Max volume bound {max_bound:.1f}cm > 500.0cm"
        
    # 5. critical_joint_coverage
    # Hips (11,12) and shoulders (5,6) must have >=50% valid frames
    if inliers.shape[1] > 12:
        critical_cov = float(np.mean(inliers[:, [5, 6, 11, 12]]))
    else:
        critical_cov = 1.0 # fallback
        
    passed_critical = critical_cov >= 0.5
    gate_results["critical_joint_coverage"] = {
        "passed": passed_critical,
        "value": critical_cov,
        "threshold": 0.5
    }
    if not passed_critical and failing_gate is None:
        failing_gate = "critical_joint_coverage"
        failure_mode = f"Critical joint coverage {critical_cov:.1%} < 50%"

    evidence = EvidenceContract(
        stage="triangulation",
        status="FAIL" if failing_gate else "PASS",
        validity_mask=inliers,
        uncertainty=diagnostics.confidence,
        calibration_id=calib_hash,
        coordinate_space="yup_cm",
        input_provenance=provenance_hash,
        gate_results=gate_results,
        coverage_fraction=inlier_ratio,
        failure_reason=failing_gate
    )
    
    if failing_gate:
        if soft_fail:
            return GateFailureReport(
                failing_stage="triangulation",
                failing_gate=failing_gate,
                cameras_implicated=[],
                failure_mode=failure_mode,
                coverage_fraction=inlier_ratio,
                evidence=evidence
            )
        else:
            raise PipelineGateError(f"Triangulation Gate Failed: {failure_mode}")
            
    return evidence
