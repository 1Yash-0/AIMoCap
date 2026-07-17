import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

def stitch_windows(windows: list[np.ndarray], frame_indices: list[np.ndarray], total_frames: int, num_vars: int) -> np.ndarray:
    """Stitches overlapping optimization windows together.
    
    Args:
        windows: List of optimized state vectors (N_win, num_vars)
        frame_indices: List of global frame indices for each window
        total_frames: Total sequence length
        num_vars: Variables per frame (e.g. 54)
    """
    final_x = np.zeros((total_frames, num_vars))
    weights = np.zeros(total_frames)
    
    J = (num_vars - 3) // 3  # Usually 17
    
    # We will accumulate positions and quaternions
    accum_pos = np.zeros((total_frames, 3))
    # For quaternions, we need to slerp, which is strictly pairwise. 
    # Because windows are sequential, we can just blend the new window into the accumulated result.
    
    # Initialize with the first window
    if len(windows) > 0:
        idx0 = frame_indices[0]
        final_x[idx0] = windows[0]
        weights[idx0] = 1.0
        
    for i in range(1, len(windows)):
        win = windows[i]
        idx = frame_indices[i]
        
        for local_f, f_global in enumerate(idx):
            if weights[f_global] == 0:
                final_x[f_global] = win[local_f]
                weights[f_global] = 1.0
            else:
                # We are in an overlap region. 
                # Find relative position in the overlap to compute smoothstep weight.
                # Assuming constant overlap size, we can estimate it based on the overlap frames seen so far for this window.
                # A simpler robust way: find overlap start and end for THIS specific overlap.
                overlap_start = idx[0]
                # End of overlap is where the PREVIOUS window ended
                overlap_end = frame_indices[i-1][-1]
                
                if f_global <= overlap_end:
                    alpha = (f_global - overlap_start) / (overlap_end - overlap_start + 1e-9)
                    alpha = smoothstep(np.array([alpha]))[0]
                    
                    prev_x = final_x[f_global]
                    curr_x = win[local_f]
                    
                    # Blend root position
                    blend_pos = (1.0 - alpha) * prev_x[0:3] + alpha * curr_x[0:3]
                    
                    # Blend rotations
                    blend_x = np.zeros(num_vars)
                    blend_x[0:3] = blend_pos
                    
                    # We have J rotations (1 root + (J-1) local)
                    # Convert rotvecs to quats
                    prev_rots = Rotation.from_rotvec(prev_x[3:].reshape(J, 3))
                    curr_rots = Rotation.from_rotvec(curr_x[3:].reshape(J, 3))
                    
                    # Sign consistency and SLERP
                    prev_q = prev_rots.as_quat()
                    curr_q = curr_rots.as_quat()
                    
                    dot_products = np.sum(prev_q * curr_q, axis=1)
                    curr_q[dot_products < 0] *= -1
                    
                    curr_rots = Rotation.from_quat(curr_q)
                    
                    # Slerp
                    # Rotation.slerp requires times. 
                    # We can use scipy.spatial.transform.Slerp, or manual SLERP for single alpha.
                    # Manual SLERP for a single alpha is fast using quaternions
                    
                    dot_products = np.clip(np.sum(prev_q * curr_q, axis=1), -1.0, 1.0)
                    theta = np.arccos(dot_products)
                    sin_theta = np.sin(theta)
                    
                    blend_q = np.zeros_like(prev_q)
                    
                    # Handle small theta
                    small = sin_theta < 1e-6
                    blend_q[small] = (1.0 - alpha) * prev_q[small] + alpha * curr_q[small]
                    blend_q[small] /= np.linalg.norm(blend_q[small], axis=1, keepdims=True)
                    
                    large = ~small
                    w1 = np.sin((1.0 - alpha) * theta[large]) / sin_theta[large]
                    w2 = np.sin(alpha * theta[large]) / sin_theta[large]
                    blend_q[large] = w1[:, None] * prev_q[large] + w2[:, None] * curr_q[large]
                    
                    blend_rots = Rotation.from_quat(blend_q)
                    blend_x[3:] = blend_rots.as_rotvec().flatten()
                    
                    final_x[f_global] = blend_x
                else:
                    # Past the overlap
                    final_x[f_global] = win[local_f]
                    weights[f_global] = 1.0
                    
    return final_x
