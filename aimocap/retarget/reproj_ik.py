import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from aimocap.retarget.mocap_skeleton import MocapSkeleton


class MocapReprojIKSolver:
    """
    Stage 6b optimizer minimizing robust 2D reprojection error.
    Unlike 3D target fitting, this directly optimizes the skeleton 
    against pixels using the camera matrices (P = K[R|t]).
    """
    def __init__(self, skel: MocapSkeleton):
        self.skel = skel
        self.num_joints = skel.num_joints
        
        # 3 root translation + (num_joints * 3) rotation vectors
        self.num_vars = 3 + self.num_joints * 3
        
        self.parents = np.array(skel.parents)
        self.rest_offsets = skel.rest_offsets.copy()
        
        depths = {0: 0}
        for i in range(1, self.num_joints):
            depths[i] = depths[self.parents[i]] + 1
            
        max_depth = max(depths.values())
        self.levels = []
        for d in range(1, max_depth + 1):
            nodes_at_d = [i for i in range(1, self.num_joints) if depths[i] == d]
            self.levels.append(np.array(nodes_at_d))

    def _state_to_local_rotations(self, x: np.ndarray):
        root_t = x[0:3]
        rotvecs = x[3:].reshape(-1, 3)
        Rs = Rotation.from_rotvec(rotvecs).as_quat()
        return root_t, Rs

    def _state_delta(self, x: np.ndarray, ref_x: np.ndarray) -> np.ndarray:
        root_t, local_q = self._state_to_local_rotations(x)
        ref_root_t, ref_local_q = self._state_to_local_rotations(ref_x)

        d_root = root_t - ref_root_t
        d_rot = (
            Rotation.from_quat(ref_local_q).inv() * Rotation.from_quat(local_q)
        ).as_rotvec().reshape(-1)
        return np.concatenate([d_root, d_rot])

    def forward_kinematics(self, x: np.ndarray):
        root_t, local_rotations = self._state_to_local_rotations(x)
        
        global_pos = np.zeros((self.num_joints, 3))
        global_rot = np.zeros((self.num_joints, 4))

        global_pos[0] = self.skel.rest_t[0] + root_t
        global_rot[0] = local_rotations[0]

        for level_nodes in self.levels:
            p_nodes = self.parents[level_nodes]

            R_parent = Rotation.from_quat(global_rot[p_nodes])
            R_local = Rotation.from_quat(local_rotations[level_nodes])

            global_pos[level_nodes] = (
                global_pos[p_nodes] + R_parent.apply(self.rest_offsets[level_nodes])
            )
            global_rot[level_nodes] = (R_parent * R_local).as_quat()

        return global_pos, global_rot

    def _reproj_residuals(self, x: np.ndarray, 
                          pts2d: np.ndarray, 
                          confs: np.ndarray, 
                          P_matrices: np.ndarray, 
                          cam_indices: np.ndarray, 
                          prev_x: np.ndarray, 
                          temporal_weight: float) -> np.ndarray:
        """
        pts2d: (N_obs, 2)
        confs: (N_obs,)
        P_matrices: (N_obs, 3, 4)
        cam_indices: (N_obs,) indicates which camera this is for (optional, mostly implicit in P)
        """
        # Forward kinematics
        global_pos, _ = self.forward_kinematics(x)
        
        # Collect 3D points for each observation
        # To do this efficiently, we need a mapping from observation to joint index
        # Let's assume pts2d is provided as a list of valid observations per joint
        pass  # We need to restructure the inputs to make this vectorized

    def _residuals_vectorized(self, x: np.ndarray,
                              obs_jidx: np.ndarray,
                              obs_pts2d: np.ndarray,
                              obs_P: np.ndarray,
                              obs_conf: np.ndarray,
                              prev_x: np.ndarray,
                              temporal_weight: float) -> np.ndarray:
        global_pos, _ = self.forward_kinematics(x)
        
        # Gather 3D points corresponding to each 2D observation
        pts3d_gather = global_pos[obs_jidx]  # (N_obs, 3)
        pts3d_homo = np.concatenate([pts3d_gather, np.ones((len(pts3d_gather), 1))], axis=-1)  # (N_obs, 4)
        
        # Project using P_matrices (N_obs, 3, 4)
        # We can do bmm: (N_obs, 3, 4) @ (N_obs, 4, 1) -> (N_obs, 3, 1)
        proj_3d = np.einsum('nij,nj->ni', obs_P, pts3d_homo)
        
        z = proj_3d[:, 2]
        # Avoid division by zero
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        
        proj_2d = proj_3d[:, :2] / z[:, None]  # (N_obs, 2)
        
        # Residuals in pixels (N_obs, 2)
        err_2d = (proj_2d - obs_pts2d) * obs_conf[:, None]
        parts = [err_2d.flatten()]
        
        if prev_x is not None and temporal_weight > 0:
            parts.append(self._state_delta(x, prev_x) * temporal_weight)
            
        return np.concatenate(parts)

    def solve_frame(self, x0: np.ndarray, 
                    obs_jidx: np.ndarray,
                    obs_pts2d: np.ndarray,
                    obs_P: np.ndarray,
                    obs_conf: np.ndarray,
                    prev_x: np.ndarray = None,
                    temporal_weight: float = 0.5) -> np.ndarray:
        """
        Solve IK for one frame directly against 2D observations.
        
        obs_jidx: (N_obs,) int array of joint indices (must match self.skel.coco_anchor mapping)
        obs_pts2d: (N_obs, 2) float array of 2D coordinates
        obs_P: (N_obs, 3, 4) float array of projection matrices K[R|t]
        obs_conf: (N_obs,) float array of confidence weights
        """
        if prev_x is None:
            # If no prev_x, initialize from x0
            pass
            
        res = least_squares(
            self._residuals_vectorized, x0,
            args=(obs_jidx, obs_pts2d, obs_P, obs_conf, prev_x, temporal_weight),
            loss='soft_l1', f_scale=10.0, # robust loss
            method='lm', max_nfev=200, ftol=1e-6, xtol=1e-6, gtol=1e-6,
        )
        return res.x
