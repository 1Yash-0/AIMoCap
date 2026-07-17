"""Triangulation orchestration engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aimocap.math.coords import opencv_to_internal
from aimocap.math.triangulate import triangulate_robust


@dataclass(frozen=True)
class TriangulationDiagnostics:
    """Per-frame/per-joint observability emitted by triangulation.

    All arrays are in the original OpenCV camera projection space, except
    ``points3d`` which is returned in the project's Y-up internal space to
    preserve the historical downstream contract.
    """

    points3d: np.ndarray
    confidence: np.ndarray
    inlier_mask: np.ndarray
    reprojection_error_px: np.ndarray
    num_inliers: np.ndarray
    low_quality_mask: np.ndarray
    ray_angle_deg: np.ndarray


def _project_point(point3d_cv: np.ndarray, P: np.ndarray) -> np.ndarray | None:
    """Project one OpenCV-space 3D point. Return None for invalid depth."""
    X = np.append(point3d_cv, 1.0)
    p = P @ X
    if not np.isfinite(p).all() or abs(float(p[2])) < 1e-9:
        return None
    return p[:2] / p[2]


def _reprojection_errors(point3d_cv: np.ndarray, pts2d: np.ndarray, P_list: list[np.ndarray]) -> np.ndarray:
    err = np.full((len(P_list),), np.inf, dtype=np.float64)
    for i, P in enumerate(P_list):
        proj = _project_point(point3d_cv, P)
        if proj is not None:
            err[i] = float(np.linalg.norm(proj - pts2d[i]))
    return err


def _triangulate_with_inlier_selection(
    pts2d: np.ndarray,
    P_list: list[np.ndarray],
    confidences: np.ndarray,
    camera_indices: np.ndarray,
    reproj_threshold_px: float,
    f_scale: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Triangulate one joint with leave-one-out style inlier selection.

    The old path used all valid views at once. With 3+ cameras, one bad 2D
    ankle/wrist can drag the solution. This routine tries all valid camera
    subsets of size >=2, scores them by reprojection residual and confidence,
    then refines using the best inlier set.
    """
    n = len(P_list)
    if n < 2:
        raise ValueError("At least 2 views are required.")

    best_score = np.inf
    best_point: np.ndarray | None = None
    best_err_local: np.ndarray | None = None
    best_subset: tuple[int, ...] | None = None

    # Keep the combinatorics bounded: common captures are 2-5 cameras.  For
    # larger rigs, first score all pairs plus the full set; this avoids an
    # accidental exponential blow-up.
    subsets: list[tuple[int, ...]] = []
    if n <= 5:
        import itertools

        for size in range(2, n + 1):
            subsets.extend(tuple(x) for x in itertools.combinations(range(n), size))
    else:
        import itertools

        subsets.extend(tuple(x) for x in itertools.combinations(range(n), 2))
        subsets.append(tuple(range(n)))

    for subset in subsets:
        sub_idx = np.array(subset, dtype=np.int32)
        try:
            x = triangulate_robust(pts2d[sub_idx], [P_list[i] for i in sub_idx], confidences[sub_idx], f_scale=f_scale)
        except Exception:
            continue
        if not np.isfinite(x).all():
            continue
        err_all = _reprojection_errors(x, pts2d, P_list)
        finite = np.isfinite(err_all)
        if not finite.any():
            continue
        # Prefer low median/max residual, but also prefer retaining more
        # confident cameras when geometric error is similar.
        subset_err = err_all[sub_idx]
        robust_err = float(np.median(subset_err) + 0.25 * np.max(subset_err))
        support_bonus = 0.5 * float(np.sum(confidences[sub_idx]))
        score = robust_err - support_bonus
        if score < best_score:
            best_score = score
            best_point = x
            best_err_local = err_all
            best_subset = subset

    if best_point is None or best_err_local is None or best_subset is None:
        x = triangulate_robust(pts2d, P_list, confidences)
        err_all = _reprojection_errors(x, pts2d, P_list)
    else:
        x = best_point
        err_all = best_err_local

    inliers_local = (err_all <= reproj_threshold_px) & np.isfinite(err_all)
    # Never shrink below the hypothesis that produced the point.
    if int(inliers_local.sum()) < 2 and best_subset is not None:
        inliers_local[:] = False
        inliers_local[np.array(best_subset, dtype=np.int32)] = True
    if int(inliers_local.sum()) >= 2 and int(inliers_local.sum()) < n:
        try:
            x = triangulate_robust(
                pts2d[inliers_local],
                [P_list[i] for i in np.flatnonzero(inliers_local)],
                confidences[inliers_local],
            )
            err_all = _reprojection_errors(x, pts2d, P_list)
            inliers_local = (err_all <= reproj_threshold_px) & np.isfinite(err_all)
        except Exception:
            pass

    inlier_mask_global = np.zeros((int(camera_indices.max()) + 1,), dtype=bool)
    inlier_mask_global[camera_indices[inliers_local]] = True
    med_err = float(np.nanmedian(np.where(np.isfinite(err_all), err_all, np.nan)))
    conf = float(np.mean(confidences[inliers_local])) if int(inliers_local.sum()) else 0.0
    # Penalize high reprojection error into a compact 0..1 confidence.
    conf *= float(1.0 / (1.0 + max(0.0, med_err) / max(1.0, reproj_threshold_px)))
    return x, inlier_mask_global, err_all, conf


def triangulate_sequence(
    keypoints: np.ndarray,
    scores: np.ndarray,
    K_list: list[np.ndarray],
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    min_conf: float = 0.5,
    min_aspect_ratio: float = 1.8,
) -> np.ndarray:
    """Back-compatible wrapper returning only Y-up 3D points."""
    return triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        K_list,
        extrinsics,
        min_conf=min_conf,
        min_aspect_ratio=min_aspect_ratio,
    ).points3d


def triangulate_sequence_with_diagnostics(
    keypoints: np.ndarray,
    scores: np.ndarray,
    K_list: list[np.ndarray],
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    min_conf: float = 0.5,
    reproj_threshold_px: float = 5.0, # Updated to 5.0 from 25.0
    min_aspect_ratio: float = 1.8,
    f_scale: float = 10.0,
) -> TriangulationDiagnostics:
    """
    Triangulate a 3D skeleton sequence and retain geometric diagnostics.

    Returns points in the Y-up internal coordinate space. Diagnostic reprojection
    errors and inlier masks are indexed as (frame, joint, camera).
    """
    num_frames, num_cameras, num_kpts, _ = keypoints.shape
    
    # Compute projection matrices P = K[R|t] for all cameras
    P_list = [K_list[c] @ np.hstack(extrinsics[c]) for c in range(num_cameras)]
    
    pts3d = np.full((num_frames, num_kpts, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((num_frames, num_kpts), dtype=np.float32)
    inlier_mask = np.zeros((num_frames, num_kpts, num_cameras), dtype=bool)
    reprojection_error_px = np.full((num_frames, num_kpts, num_cameras), np.nan, dtype=np.float32)
    num_inliers = np.zeros((num_frames, num_kpts), dtype=np.uint8)
    ray_angle_deg = np.zeros((num_frames, num_kpts), dtype=np.float32)
    
    C_list = [-extrinsics[c][0].T @ extrinsics[c][1] for c in range(num_cameras)]
    low_quality_mask = np.zeros(num_frames, dtype=bool)
    
    for f in range(num_frames):
        valid_cam_for_frame = []
        for c in range(num_cameras):
            if min_aspect_ratio > 0:
                valid_mask = scores[f, c, :] >= min_conf
                if valid_mask.sum() > 2:
                    valid_kpts = keypoints[f, c, valid_mask]
                    w = np.max(valid_kpts[:, 0]) - np.min(valid_kpts[:, 0])
                    h = np.max(valid_kpts[:, 1]) - np.min(valid_kpts[:, 1])
                    ar = h / w if w > 0 else 0
                    if ar >= min_aspect_ratio:
                        valid_cam_for_frame.append(c)
            else:
                valid_cam_for_frame.append(c)
                
        # Compute baseline angle of surviving cameras to set diagnostic low_quality_mask
        import itertools
        max_angle = 0.0
        if len(valid_cam_for_frame) >= 2:
            # We approximate the view center by averaging valid hip/shoulder joints, or just use origin
            # For robustness, we use a fixed target at origin, or we can just compute angle directly between camera centers
            # since the subject is near origin
            target = np.zeros((3, 1))
            for c1, c2 in itertools.combinations(valid_cam_for_frame, 2):
                v1 = target - C_list[c1]
                v2 = target - C_list[c2]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    cos_t = np.clip(np.dot(v1.flatten(), v2.flatten()) / (n1 * n2), -1.0, 1.0)
                    ang = np.arccos(cos_t) * 180 / np.pi
                    if ang > max_angle:
                        max_angle = ang
        
        if max_angle < 35.0:
            low_quality_mask[f] = True

        for k in range(num_kpts):
            valid_pts2d = []
            valid_P = []
            valid_conf = []
            valid_cam = []
            
            for c in valid_cam_for_frame:
                if scores[f, c, k] >= min_conf and np.isfinite(keypoints[f, c, k]).all():
                    valid_pts2d.append(keypoints[f, c, k])
                    valid_P.append(P_list[c])
                    valid_conf.append(scores[f, c, k])
                    valid_cam.append(c)
                    
            if len(valid_pts2d) >= 2:
                valid_pts2d_arr = np.array(valid_pts2d, dtype=np.float64)
                valid_conf_arr = np.array(valid_conf, dtype=np.float64)
                valid_cam_arr = np.array(valid_cam, dtype=np.int32)
                if len(valid_pts2d_arr) >= 3:
                    pt, mask_global, err_local, conf3d = _triangulate_with_inlier_selection(
                        valid_pts2d_arr,
                        valid_P,
                        valid_conf_arr,
                        valid_cam_arr,
                        reproj_threshold_px,
                        f_scale=f_scale,
                    )
                    inlier_mask[f, k, :len(mask_global)] = mask_global
                    reprojection_error_px[f, k, valid_cam_arr] = err_local.astype(np.float32)
                    confidence[f, k] = np.float32(conf3d)
                else:
                    pt = triangulate_robust(valid_pts2d_arr, valid_P, valid_conf_arr, f_scale=f_scale)
                    err_local = _reprojection_errors(pt, valid_pts2d_arr, valid_P)
                    reprojection_error_px[f, k, valid_cam_arr] = err_local.astype(np.float32)
                    inlier_mask[f, k, valid_cam_arr] = np.isfinite(err_local)
                    med_err = float(np.nanmedian(np.where(np.isfinite(err_local), err_local, np.nan)))
                    confidence[f, k] = np.float32(
                        np.mean(valid_conf_arr) / (1.0 + max(0.0, med_err) / max(1.0, reproj_threshold_px))
                    )
                num_inliers[f, k] = np.uint8(inlier_mask[f, k].sum())
                
                if num_inliers[f, k] < 2:
                    pts3d[f, k] = np.nan
                    confidence[f, k] = 0.0
                else:
                    pts3d[f, k] = pt
                
                # Compute ray angle
                inlier_cams = np.flatnonzero(inlier_mask[f, k])
                if len(inlier_cams) >= 2:
                    max_ang = 0.0
                    pt_col = pt.reshape((3, 1))
                    for c1, c2 in itertools.combinations(inlier_cams, 2):
                        v1 = pt_col - C_list[c1]
                        v2 = pt_col - C_list[c2]
                        n1 = np.linalg.norm(v1)
                        n2 = np.linalg.norm(v2)
                        if n1 > 0 and n2 > 0:
                            cos_t = np.clip(np.dot(v1.flatten(), v2.flatten()) / (n1 * n2), -1.0, 1.0)
                            ang = np.arccos(cos_t) * 180 / np.pi
                            if ang > max_ang:
                                max_ang = ang
                    ray_angle_deg[f, k] = np.float32(max_ang)
                
    # Convert from OpenCV (Y-down) to internal (Y-up) space
    pts3d_internal = opencv_to_internal(pts3d)
    
    return TriangulationDiagnostics(
        points3d=pts3d_internal,
        confidence=confidence,
        inlier_mask=inlier_mask,
        reprojection_error_px=reprojection_error_px,
        num_inliers=num_inliers,
        low_quality_mask=low_quality_mask,
        ray_angle_deg=ray_angle_deg,
    )
