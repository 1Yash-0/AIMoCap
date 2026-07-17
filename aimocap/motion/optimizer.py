from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from aimocap.motion.skeleton import CanonicalSkeleton

class SequentialCanonicalFitter:
    """Standalone sequential canonical fitter (initializer)."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def estimate_bone_lengths(self, bvh_pos: np.ndarray, recon_mask: np.ndarray) -> np.ndarray:
        """
        Median ||child - parent|| per Canonical joint, measured frames only.
        
        Args:
            bvh_pos: (F, J, 3) positions
            recon_mask: (F,) boolean array True if frame is reconstructed/filled, False if measured.
            
        Returns:
            (J,) array of bone lengths.
        """
        meas = ~recon_mask
        J = CanonicalSkeleton.num_joints()
        bl = np.zeros(J)
        
        for ji in range(1, J):
            p = CanonicalSkeleton.get_parent(ji)
            dists = np.linalg.norm(bvh_pos[meas, ji] - bvh_pos[meas, p], axis=1)
            finite = dists[np.isfinite(dists)]
            bl[ji] = float(np.median(finite)) if len(finite) >= 3 else 10.0

        return bl

    def _arc_rotation(self, src: np.ndarray, tgt: np.ndarray) -> Rotation:
        """Shortest-arc Rotation mapping unit vector src to unit vector tgt."""
        src = src / (np.linalg.norm(src) + 1e-9)
        tgt = tgt / (np.linalg.norm(tgt) + 1e-9)
        cross = np.cross(src, tgt)
        dot   = float(np.clip(np.dot(src, tgt), -1.0, 1.0))
        cl    = np.linalg.norm(cross)
        if cl < 1e-9:
            if dot > 0: return Rotation.identity()
            perp = np.array([1,0,0]) if abs(src[0]) < 0.9 else np.array([0,1,0])
            ax = np.cross(src, perp); ax /= np.linalg.norm(ax)
            return Rotation.from_rotvec(ax * np.pi)
        return Rotation.from_rotvec((cross / cl) * np.arctan2(cl, dot))

    def _get_pelvis_rotation(self, pos: np.ndarray) -> Rotation:
        # pelvis is 0, l_hip is 11, r_hip is 14
        x = pos[11] - pos[14]
        if not np.isfinite(x).all(): x = np.array([1.0, 0.0, 0.0])
        # spine is 1
        y = pos[1] - pos[0]
        if not np.isfinite(y).all(): y = np.array([0.0, 1.0, 0.0])
            
        x = x / (np.linalg.norm(x) + 1e-9)
        y = y / (np.linalg.norm(y) + 1e-9)
        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + 1e-9)
        x = np.cross(y, z)
        x = x / (np.linalg.norm(x) + 1e-9)
        
        return Rotation.from_matrix(np.column_stack((x, y, z)))

    def _get_chest_rotation(self, pos: np.ndarray, parent_rot: Rotation) -> Rotation:
        # chest is 2, l_shoulder is 5, r_shoulder is 8, neck is 3
        x = pos[5] - pos[8]
        if not np.isfinite(x).all(): x = parent_rot.apply([1, 0, 0])
            
        y = pos[3] - pos[2]
        if not np.isfinite(y).all(): y = parent_rot.apply([0, 1, 0])
            
        x = x / (np.linalg.norm(x) + 1e-9)
        y = y / (np.linalg.norm(y) + 1e-9)
        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + 1e-9)
        y = np.cross(z, x)
        y = y / (np.linalg.norm(y) + 1e-9)
        
        return Rotation.from_matrix(np.column_stack((x, y, z)))

    def fit_frame(self, pos_f: np.ndarray, bl: np.ndarray) -> tuple[list[Rotation], np.ndarray]:
        J = CanonicalSkeleton.num_joints()
        global_rot = [Rotation.identity()] * J
        local_rot  = [Rotation.identity()] * J
        fk_pos     = np.full((J, 3), np.nan)

        root = pos_f[0]
        if not np.isfinite(root).all(): return local_rot, fk_pos
            
        fk_pos[0] = root
        global_rot[0] = self._get_pelvis_rotation(pos_f)
        local_rot[0]  = global_rot[0]
        
        # Calculate chest separately if available
        if np.isfinite(pos_f[2]).all() and np.isfinite(pos_f[5]).all() and np.isfinite(pos_f[8]).all():
            global_rot[2] = self._get_chest_rotation(pos_f, global_rot[0])
        else:
            global_rot[2] = global_rot[0]

        for ji in range(1, J):
            if ji == 2:
                p = CanonicalSkeleton.get_parent(ji)
                local_rot[ji] = global_rot[p].inv() * global_rot[ji]
                continue
                
            p = CanonicalSkeleton.get_parent(ji)
            children = [c for c in range(J) if CanonicalSkeleton.get_parent(c) == ji]
            
            if len(children) == 1:
                child = children[0]
                target_vec = pos_f[child] - pos_f[ji]
                if np.isfinite(target_vec).all() and np.linalg.norm(target_vec) > 1e-6:
                    target_dir = target_vec / np.linalg.norm(target_vec)
                else:
                    target_dir = global_rot[p].apply(CanonicalSkeleton.REST_DIR[child])
                
                default_dir = global_rot[p].apply(CanonicalSkeleton.REST_DIR[child])
                R_twist = self._arc_rotation(default_dir, target_dir)
                global_rot[ji] = R_twist * global_rot[p]
                local_rot[ji] = global_rot[p].inv() * global_rot[ji]
            else:
                global_rot[ji] = global_rot[p]
                local_rot[ji] = Rotation.identity()

        for ji in range(1, J):
            p = CanonicalSkeleton.get_parent(ji)
            offset_world = global_rot[p].apply(CanonicalSkeleton.REST_DIR[ji] * bl[ji])
            fk_pos[ji] = fk_pos[p] + offset_world

        return local_rot, fk_pos

    def optimize_sequence(self, pos_seq: np.ndarray, bl: np.ndarray) -> np.ndarray:
        """Returns sequence-level x0 arrays (F, 3 + J*3)."""
        F = pos_seq.shape[0]
        J = CanonicalSkeleton.num_joints()
        x0 = np.zeros((F, 3 + J * 3))
        
        for f in range(F):
            lr, fp = self.fit_frame(pos_seq[f], bl)
            if np.isfinite(fp[0]).all():
                x0[f, 0:3] = fp[0]
                # Root rot
                x0[f, 3:6] = lr[0].as_rotvec()
                # Local rots
                for j in range(1, J):
                    x0[f, 3 + j*3 : 6 + j*3] = lr[j].as_rotvec()
            else:
                # Fallback to previous frame if NaN
                if f > 0: x0[f] = x0[f-1]
        return x0
