import numpy as np
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.ik import FbxIKSolver

print("Loading data...")
animation_data = np.load("outputs/test_manny_10frames_solved.npy")
skel = Skeleton("Manny.FBX")
solver = FbxIKSolver(skel)

num_frames = len(animation_data)
num_joints = skel.num_joints
global_pos = np.zeros((num_frames, num_joints, 3))

print("Computing FK...")
for f in range(num_frames):
    frame_data = animation_data[f]
    root_t = frame_data[0:3]
    angles_rad = frame_data[3:].reshape(num_joints, 3)
    
    import scipy.spatial.transform as st
    local_rot = st.Rotation.from_euler('xyz', angles_rad, degrees=False).as_quat()
    
    pos = solver._forward_kinematics_fast(root_t, local_rot)
    global_pos[f] = pos

connections = []
for i in range(1, num_joints):
    p = skel.parents[i]
    if p != -1 and p != i:
        connections.append((p, i))
connections = np.array(connections)

np.savez("outputs/fk_pos.npz", pos=global_pos, conn=connections)
print("Saved outputs/fk_pos.npz")
