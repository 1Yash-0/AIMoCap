import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import cv2

@dataclass
class MultiViewObservations:
    points3d: np.ndarray  # (F, K, 3)
    valid: np.ndarray     # (F, K) boolean
    covariance3d: np.ndarray  # (F, K, 3, 3)
    information3d: np.ndarray # (F, K, 3, 3)
    reprojection_rmse_px: np.ndarray # (F, K)
    max_reprojection_px: np.ndarray  # (F, K)
    inlier_mask: np.ndarray          # (F, C, K)
    observation_weight: np.ndarray   # (F, C, K)
    ray_angle_deg: np.ndarray        # (F, K)
    condition_number: np.ndarray     # (F, K)
    leave_one_out_spread: np.ndarray # (F, K)
    camera_count: np.ndarray         # (F, K)
    inlier_count: np.ndarray         # (F, K)
    provenance_flags: np.ndarray     # (F, K) string or int flags
    kpts2d: np.ndarray               # (F, C, K, 2) raw 2D keypoints

def build_multiview_observations(
    keypoints: np.ndarray, # (F, C, K, 2)
    scores: np.ndarray,    # (F, C, K)
    cameras: List["aimocap.motion.camera.CameraModel"],
    image_sizes: List[tuple[int, int]],
    fps: float,
    conf_floor: float = 0.3,
    border_margin: float = 50.0,
    inlier_thresh_px: float = 20.0
) -> MultiViewObservations:
    """Builds robust geometry-aware 3D observations from 2D points."""
    F, C, K, _ = keypoints.shape
    
    # Pre-allocate output arrays
    points3d = np.full((F, K, 3), np.nan)
    valid = np.zeros((F, K), dtype=bool)
    covariance3d = np.zeros((F, K, 3, 3))
    information3d = np.zeros((F, K, 3, 3))
    repro_rmse = np.zeros((F, K))
    max_repro = np.zeros((F, K))
    inlier_mask = np.zeros((F, C, K), dtype=bool)
    obs_weight = np.zeros((F, C, K))
    ray_angle_deg = np.zeros((F, K))
    cond_number = np.zeros((F, K))
    loo_spread = np.zeros((F, K))
    cam_count = np.zeros((F, K), dtype=int)
    inlier_count = np.zeros((F, K), dtype=int)
    prov = np.zeros((F, K), dtype=np.int32)
    
    # Undistort and compute rays for all cameras ahead of time
    from aimocap.math.triangulate import triangulate_n_views
    import scipy.linalg
    
    for f in range(F):
        for k in range(K):
            # 1. Filter by confidence
            valid_c = []
            pts2d = []
            P_mats = []
            weights = []
            c_indices = []
            
            for c in range(C):
                if scores[f, c, k] > conf_floor:
                    # Border downweighting
                    x, y = keypoints[f, c, k]
                    w, h = image_sizes[c]
                    if x > border_margin and x < w - border_margin and y > border_margin and y < h - border_margin:
                        w_border = 1.0
                    else:
                        w_border = 0.5
                        
                    valid_c.append(c)
                    c_indices.append(c)
                    pts2d.append([x, y])
                    # Extrinsics
                    R_c = cameras[c].R
                    t_c = cameras[c].t.reshape(3, 1)
                    P = cameras[c].K @ np.hstack((R_c, t_c))
                    P_mats.append(P)
                    
                    # Weight = conf * border_weight
                    weights.append(scores[f, c, k] * w_border)
                    obs_weight[f, c, k] = weights[-1]
            
            cam_count[f, k] = len(valid_c)
            
            if len(valid_c) < 2:
                continue
                
            # Basic DLT to get an initial point
            pts2d_np = np.array(pts2d)
            try:
                pt3d_initial = triangulate_n_views(pts2d_np, P_mats)
            except:
                continue
                
            # Robust scoring (Huber-like)
            res = []
            for i, c in enumerate(valid_c):
                proj = cameras[c].project(pt3d_initial.reshape(1, 3))
                if len(proj) > 0:
                    dist = np.linalg.norm(proj[0] - pts2d_np[i])
                    res.append(dist)
                else:
                    res.append(9999.0)
                    
            res = np.array(res)
            inliers = res < inlier_thresh_px
            inlier_count[f, k] = np.sum(inliers)
            
            if inlier_count[f, k] < 2:
                continue
                
            for i, c in enumerate(valid_c):
                if inliers[i]:
                    inlier_mask[f, c, k] = True
                    
            # Refine over inliers using least squares
            from scipy.optimize import least_squares
            
            def fun(x3d):
                errs = []
                for i, c in enumerate(valid_c):
                    if inliers[i]:
                        proj = cameras[c].project(x3d.reshape(1, 3))
                        if len(proj) > 0:
                            # Residual scaling: sqrt(weight) * pixel_residual
                            err = np.sqrt(weights[i]) * (proj[0] - pts2d_np[i])
                            errs.extend(err)
                        else:
                            errs.extend([999.0, 999.0])
                return np.array(errs)
                
            res_opt = least_squares(fun, pt3d_initial, loss='huber')
            pt3d_refined = res_opt.x
            
            points3d[f, k] = pt3d_refined
            valid[f, k] = True
            
            # Compute reprojection stats for inliers
            final_res = []
            for i, c in enumerate(valid_c):
                if inliers[i]:
                    proj = cameras[c].project(pt3d_refined.reshape(1, 3))
                    if len(proj) > 0:
                        dist = np.linalg.norm(proj[0] - pts2d_np[i])
                        final_res.append(dist)
                        
            if len(final_res) > 0:
                repro_rmse[f, k] = np.sqrt(np.mean(np.array(final_res)**2))
                max_repro[f, k] = np.max(final_res)
                
            # Compute Covariance / Information matrix
            # H = J.T W J
            J = res_opt.jac
            H = J.T @ J
            information3d[f, k] = H
            try:
                cov = np.linalg.pinv(H)
                # Clamp eigenvalues
                eigvals, eigvecs = scipy.linalg.eigh(cov)
                eigvals = np.clip(eigvals, 1e-4, 1e4) # Floor and ceiling
                cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                covariance3d[f, k] = cov
                cond_number[f, k] = np.max(eigvals) / np.min(eigvals)
            except:
                pass
                
            # Ray angle (max angle between any two inlier rays)
            rays = []
            for i, c in enumerate(valid_c):
                if inliers[i]:
                    cam_pos = -cameras[c].R.T @ cameras[c].t
                    ray = pt3d_refined - cam_pos.flatten()
                    ray = ray / (np.linalg.norm(ray) + 1e-9)
                    rays.append(ray)
                    
            max_angle = 0.0
            for i in range(len(rays)):
                for j in range(i+1, len(rays)):
                    dot = np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0)
                    ang = np.arccos(dot)
                    if ang > max_angle:
                        max_angle = ang
            ray_angle_deg[f, k] = np.rad2deg(max_angle)
            prov[f, k] = 1 # Valid 3D from Multi-view

    return MultiViewObservations(
        points3d=points3d,
        valid=valid,
        covariance3d=covariance3d,
        information3d=information3d,
        reprojection_rmse_px=repro_rmse,
        max_reprojection_px=max_repro,
        inlier_mask=inlier_mask,
        observation_weight=obs_weight,
        ray_angle_deg=ray_angle_deg,
        condition_number=cond_number,
        leave_one_out_spread=loo_spread,
        camera_count=cam_count,
        inlier_count=inlier_count,
        provenance_flags=prov,
        kpts2d=keypoints
    )
