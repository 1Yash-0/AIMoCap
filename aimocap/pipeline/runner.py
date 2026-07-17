import numpy as np
import json
from pathlib import Path
from aimocap.pipeline.gates import check_2d_gates, check_triangulation_gates, GateFailureReport
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics, TriangulationDiagnostics
from aimocap.diagnostics.residuals import compute_residual_vectors, analyze_residuals, save_residual_diagnostics
from aimocap.pipeline.evidence import EvidenceContract

def check_stage6b_gates(
    initial_diagnostics: TriangulationDiagnostics,
    stage6b_residuals_px: np.ndarray,
    calib_hash: str,
    provenance_hash: str,
    soft_fail: bool = True
) -> EvidenceContract | GateFailureReport:
    """
    Reject any Stage 6b result that materially worsens the initial reprojection error.
    """
    # Calculate median residuals before and after optimization
    # ignoring NaNs
    med_initial = np.nanmedian(initial_diagnostics.reprojection_error_px)
    med_stage6b = np.nanmedian(stage6b_residuals_px)
    
    p95_initial = np.nanpercentile(initial_diagnostics.reprojection_error_px, 95)
    p95_stage6b = np.nanpercentile(stage6b_residuals_px, 95)
    
    if med_stage6b > med_initial * 1.5 or p95_stage6b > p95_initial * 1.5:
        # It forced a rigid fit that destroys reprojection evidence
        if not soft_fail:
            raise ValueError(f"Stage 6b Gate Failed: med_err worsened {med_initial:.2f}->{med_stage6b:.2f}")
        return GateFailureReport(
            failing_stage="Stage 6b Reprojection",
            failing_gate="Rigidity Overfitting",
            cameras_implicated=[],
            failure_mode=f"residuals worsened significantly",
            coverage_fraction=0.0,
            evidence=EvidenceContract(calib_hash, provenance_hash, initial_diagnostics.points3d.shape[0])
        )
        
    return EvidenceContract(calib_hash, provenance_hash, initial_diagnostics.points3d.shape[0])

def run_evidence_gated_triangulation(
    keypoints: np.ndarray,      # (F, C, J, 2)
    scores: np.ndarray,         # (F, C, J)
    K_list: list[np.ndarray], 
    extrinsics: list[tuple[np.ndarray, np.ndarray]], 
    calib_hash: str,
    provenance_hash: str,
    camera_names: list[str],
    output_dir: Path | str,
    output_name: str = "triangulated.npz",
    image_width: float = 1920.0,
    image_height: float = 1080.0,
    min_conf: float = 0.5,
    reproj_threshold_px: float = 5.0, # Note: using 5.0 here instead of 25.0
    f_scale: float = 10.0,
    soft_fail: bool = True
):
    """
    Run triangulation with strictly gated evidence checks.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Check 2D Observation Gates
    report_2d = check_2d_gates(
        keypoints, scores, K_list, extrinsics, calib_hash, provenance_hash,
        image_width, image_height, min_conf, soft_fail
    )
    
    if isinstance(report_2d, GateFailureReport):
        print(report_2d.summary())
        # Write failure report
        with open(out_dir / f"{Path(output_name).stem}_gate_failure.json", "w") as f:
            json.dump({
                "failing_stage": report_2d.failing_stage,
                "failing_gate": report_2d.failing_gate,
                "cameras_implicated": report_2d.cameras_implicated,
                "failure_mode": report_2d.failure_mode,
                "coverage_fraction": report_2d.coverage_fraction
            }, f, indent=2)
        report_2d.evidence.save_sidecar(out_dir / output_name)
        return report_2d
        
    # 2. Triangulate
    # We pass the transposed keypoints (F, C, J, 2) -> Triangulation expects (F, C, J, 2) wait.
    # engine.py: num_frames, num_cameras, num_kpts, _ = keypoints.shape
    # Yes, shape is (F, C, J, 2).
    
    diagnostics = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        K_list,
        extrinsics,
        min_conf=min_conf,
        reproj_threshold_px=reproj_threshold_px,
        f_scale=f_scale
    )
    
    # 3. Check Triangulation Gates
    report_3d = check_triangulation_gates(
        diagnostics, calib_hash, provenance_hash, soft_fail
    )
    
    if isinstance(report_3d, GateFailureReport):
        print(report_3d.summary())
        # Write failure report
        with open(out_dir / f"{Path(output_name).stem}_gate_failure.json", "w") as f:
            json.dump({
                "failing_stage": report_3d.failing_stage,
                "failing_gate": report_3d.failing_gate,
                "cameras_implicated": report_3d.cameras_implicated,
                "failure_mode": report_3d.failure_mode,
                "coverage_fraction": report_3d.coverage_fraction
            }, f, indent=2)
        report_3d.evidence.save_sidecar(out_dir / output_name)
        return report_3d
        
    # 4. Compute residual diagnostics (informational)
    P_list = [K_list[c] @ np.hstack(extrinsics[c]) for c in range(len(K_list))]
    
    # engine returns pts3d in Y-up. But compute_residual_vectors expects the same coordinate
    # space as P_list. P_list maps world (Y-down) to image. 
    # Therefore, we need to convert Y-up back to OpenCV (Y-down) for residual computation.
    # Wait, OpenCV to Y-up in engine is:
    # `pts3d_internal = opencv_to_internal(pts3d)`
    # where opencv_to_internal rotates.
    
    from aimocap.math.coords import internal_to_opencv
    pts3d_opencv = internal_to_opencv(diagnostics.points3d)
    
    residuals = compute_residual_vectors(
        np.transpose(keypoints, (0, 2, 1, 3)), # (F, J, C, 2)
        pts3d_opencv,
        P_list,
        diagnostics.inlier_mask # (F, J, C)
    )
    
    res_stats = analyze_residuals(residuals, camera_names)
    save_residual_diagnostics(res_stats, out_dir / f"{Path(output_name).stem}_residuals.json")
    
    # 5. Success! Write valid artifact
    np.savez(
        out_dir / output_name,
        points3d=diagnostics.points3d,
        confidence=diagnostics.confidence,
        inlier_mask=diagnostics.inlier_mask,
    )
    report_3d.save_sidecar(out_dir / output_name)
    
    return diagnostics
