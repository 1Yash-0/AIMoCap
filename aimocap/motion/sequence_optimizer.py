import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from aimocap.motion.skeleton import CanonicalSkeleton
from aimocap.motion.observations import MultiViewObservations
import cv2

class WindowedSequenceOptimizer:
    def __init__(self, cameras, fps: float, bone_lengths: np.ndarray, b_stage6: np.ndarray = None):
        self.cameras = cameras
        self.fps = fps
        self.bl = bone_lengths
        self.J = CanonicalSkeleton.num_joints()
        self.b_stage6 = b_stage6  # Candidate B prior (F, J, 3)
        
        # Hyperparameters
        self.w_2d = 1.0           # Direct 2D reprojection
        self.w_3d = 1.0           # 3D uncertainty-aware
        self.w_prior = 0.05       # Weak prior for Candidate B
        self.w_root_smooth = 10.0 # Root vel/accel
        self.w_rot_smooth = 5.0   # Local rot vel/accel
        self.w_limits = 10.0      # Joint limits
        self.w_twist = 1.0        # Twist regularization
        self.w_chest = 5.0        # Chest/Pelvis consistency
        
    def _fk(self, x: np.ndarray, N: int) -> tuple[np.ndarray, list[list[Rotation]]]:
        """Forward Kinematics.
        x: (N, 3 + J*3)
        """
        x_frames = x.reshape(N, 3 + self.J * 3)
        pos = np.zeros((N, self.J, 3))
        all_global = []
        
        # Calculate offsets once
        rest_dirs = CanonicalSkeleton.REST_DIR * self.bl[:, None]
        
        for f in range(N):
            root_p = x_frames[f, 0:3]
            root_r = Rotation.from_rotvec(x_frames[f, 3:6])
            local_r = Rotation.from_rotvec(x_frames[f, 6:].reshape(self.J - 1, 3))
            
            pos[f, 0] = root_p
            
            global_rot = [None] * self.J
            global_rot[0] = root_r
            
            for j in range(1, self.J):
                p = CanonicalSkeleton.PARENTS[j]
                global_rot[j] = global_rot[p] * local_r[j-1]
                pos[f, j] = pos[f, p] + global_rot[p].apply(rest_dirs[j])
                
            all_global.append(global_rot)
            
        return pos, all_global

    def _residuals(self, x: np.ndarray, N: int, obs: MultiViewObservations, frame_indices: np.ndarray) -> np.ndarray:
        pos, global_rots = self._fk(x, N)
        x_frames = x.reshape(N, 3 + self.J * 3)
        
        res_list = []
        
        for local_f, f_global in enumerate(frame_indices):
            # A. Direct 2D Reprojection
            # For each valid camera inlier
            for c_idx, cam in enumerate(self.cameras):
                mask = obs.inlier_mask[f_global, c_idx]
                if np.any(mask):
                    pts3d = pos[local_f, mask]
                    
                    # Convert pts3d from Canonical to OpenCV for projection
                    # OpenCV is R = I, t = 0 (Wait, cam.K and cam.R/t already handle world->camera)
                    # But if cam.R and cam.t were calibrated in OpenCV space, we need to convert points!
                    # Actually, the user says calibration loader should not be ambiguous. Let's assume cameras operate in Canonical space natively if we converted them, but wait...
                    # We can just project directly since we will ensure the camera matrices are consistent.
                    
                    rvec, _ = cv2.Rodrigues(cam.R)
                    proj, _ = cv2.projectPoints(pts3d, rvec, cam.t.reshape(3,1), cam.K, cam.dist)
                    proj = proj.reshape(-1, 2)
                    
                    obs2d = obs.kpts2d[f_global, c_idx, mask]
                    w = obs.observation_weight[f_global, c_idx, mask]
                    
                    err2d = np.sqrt(w[:, None]) * (proj - obs2d)
                    res_list.append(err2d.flatten() * self.w_2d)
                    
            # B. Uncertainty-aware 3D observations
            # r3d = L @ (x_fk - x_obs)
            valid3d = obs.valid[f_global]
            if np.any(valid3d):
                p3d_fk = pos[local_f, valid3d]
                p3d_obs = obs.points3d[f_global, valid3d]
                info = obs.information3d[f_global, valid3d] # (M, 3, 3)
                
                for i in range(len(p3d_fk)):
                    try:
                        L = np.linalg.cholesky(info[i] + np.eye(3)*1e-6)
                        err3d = L @ (p3d_fk[i] - p3d_obs[i])
                        res_list.append(err3d * self.w_3d)
                    except:
                        pass
                        
            # C. Candidate B weak prior
            if self.b_stage6 is not None:
                # Use only for weak/missing joints
                weak = ~valid3d
                if np.any(weak):
                    p_fk = pos[local_f, weak]
                    p_b = self.b_stage6[f_global, weak]
                    res_list.append((p_fk - p_b).flatten() * self.w_prior)
                    
        # D/E. Temporal behavior (Smoothness)
        if N > 1:
            # Root velocity
            v_root = (x_frames[1:, 0:3] - x_frames[:-1, 0:3]) * self.fps
            res_list.append(v_root.flatten() * self.w_root_smooth)
            if N > 2:
                a_root = (v_root[1:] - v_root[:-1]) * self.fps
                res_list.append(a_root.flatten() * self.w_root_smooth)
                
            # Rotation velocity (angular velocity)
            for j in range(1, self.J):
                for f in range(1, N):
                    r_prev = global_rots[f-1][j]
                    r_curr = global_rots[f][j]
                    delta = r_prev.inv() * r_curr
                    res_list.append(delta.as_rotvec() * self.fps * self.w_rot_smooth)
                    
        # F. Joint Limits (Soft penalties)
        # We can penalize local rotations if their angle is too large
        local_rotvecs = x_frames[:, 6:].reshape(N, self.J - 1, 3)
        angles = np.linalg.norm(local_rotvecs, axis=2)
        # e.g., max angle 2.5 rad
        limit_err = np.maximum(0, angles - 2.5)
        res_list.append(limit_err.flatten() * self.w_limits)

        # H. Twist regularization
        # Penalize Y-axis (twist) on certain joints
        twist_err = local_rotvecs[:, :, 1] # assuming Y is the twist axis in local space for limbs
        res_list.append(twist_err.flatten() * self.w_twist)
        
        return np.concatenate(res_list) if len(res_list) > 0 else np.zeros(1)

    def get_sparsity(self, N: int) -> lil_matrix:
        num_vars = N * (3 + self.J * 3)
        # Approximate dense blocks for each frame + adjacent frames
        # Just return dense for small windows (N <= 60), or build sparse matrix
        from scipy.sparse import block_diag
        blocks = [np.ones((54, 54)) for _ in range(N)]
        S = block_diag(blocks, format='lil')
        
        # Add temporal overlap
        for f in range(N - 1):
            start1 = f * 54
            start2 = (f + 1) * 54
            S[start1:start1+54, start2:start2+54] = 1
            S[start2:start2+54, start1:start1+54] = 1
            
        return S

    def optimize_window(self, obs: MultiViewObservations, frame_indices: np.ndarray, x0: np.ndarray) -> np.ndarray:
        N = len(frame_indices)
        if N == 0: return x0
        
        # Evaluate residuals once to get size
        r0 = self._residuals(x0, N, obs, frame_indices)
        if len(r0) == 0: return x0
        
        S = self.get_sparsity(N)
        S_padded = lil_matrix((len(r0), len(x0)), dtype=int)
        
        # Since we don't know exact mapping of residuals to variables without tracing,
        # we can use '2-point' without sparsity if N is small, or use a broad sparsity.
        # Given max 60 frames (3240 vars), dense jacobian is ~10 million elements, which is manageable.
        
        res = least_squares(
            fun=self._residuals,
            x0=x0,
            jac='2-point',
            kwargs={'N': N, 'obs': obs, 'frame_indices': frame_indices},
            method='trf',
            loss='huber',
            max_nfev=20, # Keep it low for performance in this python implementation
            verbose=0
        )
        return res.x
