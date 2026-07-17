import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse
from aimocap.retarget.fbx_rig import Skeleton
from scipy.spatial.transform import Rotation

PREVIEW_BONES = {
    "pelvis",
    "spine_01",
    "spine_02",
    "spine_03",
    "spine_04",
    "spine_05",
    "neck_01",
    "neck_02",
    "head",
    "clavicle_l",
    "upperarm_l",
    "lowerarm_l",
    "hand_l",
    "clavicle_r",
    "upperarm_r",
    "lowerarm_r",
    "hand_r",
    "thigh_l",
    "calf_l",
    "foot_l",
    "ball_l",
    "thigh_r",
    "calf_r",
    "foot_r",
    "ball_r",
}

def visualize_bvh(npy_path, fbx_path, output_gif, start_seconds=5.0, fps=30.0, max_frames=300):
    print(f"Loading data from {npy_path}...")
    animation_data = np.load(npy_path)
    
    print(f"Loading rig from {fbx_path}...")
    skel = Skeleton(fbx_path)
    
    print("Computing Forward Kinematics for all frames...")
    num_frames = len(animation_data)
    num_joints = skel.num_joints
    start_frame = min(int(round(start_seconds * fps)), max(0, num_frames - 1))
    end_frame = min(num_frames, start_frame + max_frames)
    
    global_pos = np.zeros((num_frames, num_joints, 3))
    
    for f in range(num_frames):
        root_t = animation_data[f, 0:3]
        local_eulers = animation_data[f, 3:].reshape(skel.num_joints, 3)
        local_rot = Rotation.from_euler('xyz', local_eulers, degrees=False).as_quat()
        pos, _ = skel.get_forward_kinematics(local_rot, root_translation=root_t)
        global_pos[f] = pos
        
    print("Rendering GIF...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    preview_indices = [
        i for i, name in enumerate(skel.node_names)
        if name in PREVIEW_BONES
    ]
    
    # Calculate bounds over the preview body chain, not over hidden
    # twist/finger/helper bones.
    preview_pos = global_pos[start_frame:end_frame, preview_indices]
    min_b = np.min(preview_pos, axis=(0, 1))
    max_b = np.max(preview_pos, axis=(0, 1))
    max_range = np.array([max_b[0]-min_b[0], max_b[1]-min_b[1], max_b[2]-min_b[2]]).max() / 2.0
    mid_x = (max_b[0]+min_b[0]) * 0.5
    mid_y = (max_b[1]+min_b[1]) * 0.5
    mid_z = (max_b[2]+min_b[2]) * 0.5
    
    lines = []
    
    # Pre-compute valid connections
    connections = []
    preview_set = set(preview_indices)
    for i in range(1, num_joints):
        p = skel.parents[i]
        if p != -1 and p != i and i in preview_set and p in preview_set:
            connections.append((p, i))
            
    for _ in range(len(connections)):
        line, = ax.plot([], [], [], 'bo-', linewidth=2, markersize=2)
        lines.append(line)
        
    try:
        def update(frame):
            if frame % 10 == 0:
                print(f"Rendering frame {frame}...")
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_title(f"Retargeted Skeleton - Frame {frame}")
            
            pts = global_pos[frame]
            for idx, (p, c) in enumerate(connections):
                p1 = pts[p]
                p2 = pts[c]
                lines[idx].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                lines[idx].set_3d_properties([p1[2], p2[2]])
                
            return lines

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(start_frame, end_frame),
            interval=int(round(1000.0 / fps)),
            blit=False,
        )
        ani.save(output_gif, writer='pillow')
        print(f"Saved GIF to {output_gif}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Crashed during rendering:", e)
    
if __name__ == "__main__":
    visualize_bvh("outputs/manny_retargeted_solved.npy", "Manny.FBX", "outputs/retargeted_skeleton.gif")
