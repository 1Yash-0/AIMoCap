"""Anthropometric scaling for metric recovery."""

from __future__ import annotations

import numpy as np


# Approximate adult human bone lengths in meters.
# Using COCO-WholeBody keypoint indices (0-16 for body).
# 5: L-shoulder, 6: R-shoulder
# 11: L-hip, 12: R-hip
# 13: L-knee, 14: R-knee
# 15: L-ankle, 16: R-ankle
_BONE_PRIORS = [
    ((5, 6), 0.36),   # Shoulder width
    ((11, 12), 0.28), # Hip width
    ((11, 13), 0.40), # Left femur
    ((12, 14), 0.40), # Right femur
    ((13, 15), 0.40), # Left tibia
    ((14, 16), 0.40), # Right tibia
    ((5, 7), 0.30),   # Left upper arm
    ((6, 8), 0.30),   # Right upper arm
    ((7, 9), 0.25),   # Left lower arm
    ((8, 10), 0.25),  # Right lower arm
]


def apply_metric_scale(
    skeleton3d: np.ndarray,
    extrinsics: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], float]:
    """
    Scale the triangulated 3D skeleton and camera translations to metric units (meters)
    using average human anthropometric bone lengths.
    
    Args:
        skeleton3d: (num_frames, 133, 3) float64 array of 3D keypoints.
        extrinsics: List of N (R, t) camera poses.
        
    Returns:
        scaled_skeleton3d: (num_frames, 133, 3) scaled to meters.
        scaled_extrinsics: List of N (R, t) where t is scaled to meters.
        scale_factor: The multiplier applied to the scene.
    """
    if len(skeleton3d) == 0:
        return skeleton3d, extrinsics, 1.0
        
    estimated_scales = []
    
    for bone_idx, true_length in _BONE_PRIORS:
        idx_a, idx_b = bone_idx
        
        # Calculate lengths across all frames
        pts_a = skeleton3d[:, idx_a, :]
        pts_b = skeleton3d[:, idx_b, :]
        
        # Ignore frames where either point is NaN
        valid = ~np.isnan(pts_a[:, 0]) & ~np.isnan(pts_b[:, 0])
        
        if not np.any(valid):
            continue
            
        diff = pts_a[valid] - pts_b[valid]
        lengths = np.linalg.norm(diff, axis=1)
        
        # Median length across all valid frames to ignore outliers
        median_length = np.median(lengths)
        
        if median_length > 1e-5:
            # scale_factor = true_length / measured_length
            estimated_scales.append(true_length / median_length)
            
    if not estimated_scales:
        # Fallback if no bones were visible
        return skeleton3d, extrinsics, 1.0
        
    # Final robust scale factor
    scale_factor = np.median(estimated_scales)
    
    # Apply to skeleton
    scaled_skeleton3d = skeleton3d * scale_factor
    
    # Apply to camera translations (R remains unchanged)
    scaled_extrinsics = [
        (R, t * scale_factor) for R, t in extrinsics
    ]
    
    return scaled_skeleton3d, scaled_extrinsics, float(scale_factor)
