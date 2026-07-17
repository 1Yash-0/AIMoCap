import numpy as np

class CanonicalSkeleton:
    """Target-independent canonical mocap skeleton (17 joints).
    
    Layout (indices):
    0: pelvis          (derived: midpoint of left/right hip)
    1: spine           (inferred/virtual between pelvis and chest)
    2: chest           (derived: midpoint of left/right shoulder)
    3: neck            (inferred/virtual between chest and head, or COCO 5/6 midpoint offset)
    4: head            (COCO 0: nose)
    5: left_shoulder   (COCO 5)
    6: left_elbow      (COCO 7)
    7: left_wrist      (COCO 9)
    8: right_shoulder  (COCO 6)
    9: right_elbow     (COCO 8)
    10: right_wrist    (COCO 10)
    11: left_hip       (COCO 11)
    12: left_knee      (COCO 13)
    13: left_ankle     (COCO 15)
    14: right_hip      (COCO 12)
    15: right_knee     (COCO 14)
    16: right_ankle    (COCO 16)
    """

    NAMES = [
        "pelvis", "spine", "chest", "neck", "head",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle"
    ]
    
    # -1 means root (pelvis)
    PARENTS = [
        -1, 0, 1, 2, 3,
        2, 5, 6,
        2, 8, 9,
        0, 11, 12,
        0, 14, 15
    ]
    
    # Bone direction vectors in canonical rest pose (parent-local coordinates)
    REST_DIR = np.zeros((17, 3))
    REST_DIR[1]  = [ 0,  1,  0]  # spine points +Y
    REST_DIR[2]  = [ 0,  1,  0]  # chest points +Y
    REST_DIR[3]  = [ 0,  1,  0]  # neck points +Y
    REST_DIR[4]  = [ 0,  1,  0]  # head points +Y
    REST_DIR[5]  = [ 1,  0,  0]  # left_shoulder points +X
    REST_DIR[6]  = [ 1,  0,  0]  # left_elbow points +X
    REST_DIR[7]  = [ 1,  0,  0]  # left_wrist points +X
    REST_DIR[8]  = [-1,  0,  0]  # right_shoulder points -X
    REST_DIR[9]  = [-1,  0,  0]  # right_elbow points -X
    REST_DIR[10] = [-1,  0,  0]  # right_wrist points -X
    REST_DIR[11] = [ 1,  0,  0]  # left_hip points +X (from pelvis out)
    REST_DIR[12] = [ 0, -1,  0]  # left_knee points -Y
    REST_DIR[13] = [ 0, -1,  0]  # left_ankle points -Y
    REST_DIR[14] = [-1,  0,  0]  # right_hip points -X
    REST_DIR[15] = [ 0, -1,  0]  # right_knee points -Y
    REST_DIR[16] = [ 0, -1,  0]  # right_ankle points -Y

    @classmethod
    def num_joints(cls) -> int:
        return len(cls.NAMES)

    @classmethod
    def get_parent(cls, idx: int) -> int:
        return cls.PARENTS[idx]

    @classmethod
    def build_positions_from_coco(cls, pts3d: np.ndarray) -> np.ndarray:
        """(F, 17, 3) world positions from COCO-17 points."""
        F = pts3d.shape[0]
        pos = np.full((F, cls.num_joints(), 3), np.nan)
        
        # Direct maps
        pos[:, 4] = pts3d[:, 0]    # head
        pos[:, 5] = pts3d[:, 5]    # l_shoulder
        pos[:, 6] = pts3d[:, 7]    # l_elbow
        pos[:, 7] = pts3d[:, 9]    # l_wrist
        pos[:, 8] = pts3d[:, 6]    # r_shoulder
        pos[:, 9] = pts3d[:, 8]    # r_elbow
        pos[:, 10] = pts3d[:, 10]  # r_wrist
        pos[:, 11] = pts3d[:, 11]  # l_hip
        pos[:, 12] = pts3d[:, 13]  # l_knee
        pos[:, 13] = pts3d[:, 15]  # l_ankle
        pos[:, 14] = pts3d[:, 12]  # r_hip
        pos[:, 15] = pts3d[:, 14]  # r_knee
        pos[:, 16] = pts3d[:, 16]  # r_ankle
        
        # Derived
        pos[:, 0] = (pts3d[:, 11] + pts3d[:, 12]) / 2.0   # pelvis
        pos[:, 2] = (pts3d[:, 5]  + pts3d[:, 6])  / 2.0   # chest
        
        # Virtual intermediate joints (spine, neck)
        # Spine = 1/2 between pelvis and chest
        pos[:, 1] = (pos[:, 0] + pos[:, 2]) / 2.0
        # Neck = 1/2 between chest and head (approximate structural offset)
        pos[:, 3] = pos[:, 2] + (pos[:, 4] - pos[:, 2]) * 0.33
        
        return pos
