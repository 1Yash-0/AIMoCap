import numpy as np
from typing import Dict, Any, Tuple
from aimocap.motion.skeleton import CanonicalSkeleton
from aimocap.motion.observations import MultiViewObservations

def estimate_bone_lengths_robust(
    observations: MultiViewObservations,
    min_ray_angle_deg: float = 15.0,
    max_cond_number: float = 1000.0,
    pool_threshold: float = 0.05,
    warn_threshold: float = 0.10
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Robust bone length estimation with strict provenance and bilateral validation.
    
    Args:
        observations: The strict multi-view observations.
        
    Returns:
        bl: (J,) array of bone lengths.
        report: Dict containing estimation diagnostics and warnings.
    """
    F, K, _ = observations.points3d.shape
    J = CanonicalSkeleton.num_joints()
    
    # 1. Identify strong frames per joint
    # - Valid triangulated point (valid=True)
    # - inlier_count >= 2
    # - ray_angle >= min_ray_angle_deg
    # - condition_number <= max_cond_number
    # - provenance == 1 (pure multiview, no priors)
    
    strong_mask = (
        observations.valid & 
        (observations.inlier_count >= 2) &
        (observations.ray_angle_deg >= min_ray_angle_deg) &
        (observations.condition_number <= max_cond_number) &
        (observations.provenance_flags == 1)
    )
    
    bl_estimates = [[] for _ in range(J)]
    
    for j in range(1, J):
        p = CanonicalSkeleton.PARENTS[j]
        # A frame is strong for the bone if BOTH child and parent are strong
        bone_strong = strong_mask[:, j] & strong_mask[:, p]
        
        if np.any(bone_strong):
            p3d = observations.points3d[bone_strong]
            lengths = np.linalg.norm(p3d[:, j] - p3d[:, p], axis=1)
            # Weights could be based on inverse covariance, but robust median is required
            # Here we just use the unweighted median of the strong subset
            bl_estimates[j] = lengths
            
    bl_final = np.zeros(J)
    report = {}
    
    # Bilateral pairs (left, right)
    pairs = [
        ("shoulder", 5, 8),
        ("elbow", 6, 9),
        ("wrist", 7, 10),
        ("hip", 11, 14),
        ("knee", 12, 15),
        ("ankle", 13, 16)
    ]
    
    paired_indices = set([idx for pair in pairs for idx in pair[1:]])
    
    for j in range(1, J):
        if j in paired_indices: continue
        
        lengths = bl_estimates[j]
        name = CanonicalSkeleton.NAMES[j]
        if len(lengths) < 10:
            report[name] = {"warning": f"Insufficient strong frames ({len(lengths)})"}
            bl_final[j] = np.median(lengths) if len(lengths) > 0 else 10.0
        else:
            med = float(np.median(lengths))
            bl_final[j] = med
            report[name] = {
                "length": med,
                "support_frames": len(lengths),
                "support_pct": len(lengths) / F * 100,
                "uncertainty": float(np.std(lengths)),
                "warning": None
            }
            
    # Process pairs
    for name, l_idx, r_idx in pairs:
        l_len = bl_estimates[l_idx]
        r_len = bl_estimates[r_idx]
        
        l_med = float(np.median(l_len)) if len(l_len) >= 10 else 0.0
        r_med = float(np.median(r_len)) if len(r_len) >= 10 else 0.0
        
        if l_med == 0.0 or r_med == 0.0:
            # Fallback
            bl_final[l_idx] = l_med if l_med > 0 else (r_med if r_med > 0 else 10.0)
            bl_final[r_idx] = r_med if r_med > 0 else (l_med if l_med > 0 else 10.0)
            report[f"{name}_pair"] = {"warning": "Insufficient bilateral support"}
            continue
            
        diff = abs(l_med - r_med)
        asym = diff / ((l_med + r_med) / 2.0)
        
        report_data = {
            "l_length": l_med,
            "r_length": r_med,
            "l_support": len(l_len),
            "r_support": len(r_len),
            "asymmetry": asym,
            "pooled": False,
            "warning": None
        }
        
        if asym <= pool_threshold:
            # Pool robustly
            pooled_lengths = np.concatenate([l_len, r_len])
            pooled_med = float(np.median(pooled_lengths))
            bl_final[l_idx] = pooled_med
            bl_final[r_idx] = pooled_med
            report_data["pooled"] = True
            report_data["pooled_length"] = pooled_med
        elif asym <= warn_threshold:
            bl_final[l_idx] = l_med
            bl_final[r_idx] = r_med
            report_data["warning"] = "Moderate asymmetry (5-10%), retaining independent values."
        else:
            bl_final[l_idx] = l_med
            bl_final[r_idx] = r_med
            report_data["warning"] = "SEVERE ASYMMETRY (>10%). Classification: Unreliable estimation."
            
        report[f"{name}_pair"] = report_data
        
    return bl_final, report
