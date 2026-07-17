import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.spatial.transform import Rotation

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.math.filter import filter_skeleton3d
from aimocap.retarget.engine import retarget_to_fbx
from aimocap.retarget.fbx_rig import Skeleton

OUT_VIS = ROOT / "outputs/phase_b_final_visuals"
OUT_VIS.mkdir(parents=True, exist_ok=True)
GATE1_DIR = ROOT / "outputs/phase_b_gate1"

CONNECTIONS_COCO = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 0), (6, 0), (0, 1), (1, 2), (2, 3), (1, 4)]
PREVIEW_BONES = {
    "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
    "neck_01", "neck_02", "head",
    "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
    "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
    "thigh_l", "calf_l", "foot_l", "ball_l",
    "thigh_r", "calf_r", "foot_r", "ball_r",
}

def plot_skeleton_2d(ax, pts2d, conf, title):
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    if not np.isfinite(pts2d).any(): return
    for p, c in CONNECTIONS_COCO:
        if p < len(pts2d) and c < len(pts2d):
            if conf[p] > 0.1 and conf[c] > 0.1:
                ax.plot([pts2d[p,0], pts2d[c,0]], [pts2d[p,1], pts2d[c,1]], color='blue', alpha=0.5, linewidth=1)
    
    for j in range(len(pts2d)):
        if conf[j] > 0.1:
            ax.scatter(pts2d[j,0], pts2d[j,1], color='red', s=10)

def main():
    print("Loading Gate 1 arrays...")
    arrays = np.load(GATE1_DIR / "gate1_arrays.npz")
    b_s6 = arrays["b_stage6"] / 10.0  # mm to cm
    gt = arrays["gt"] / 10.0
    
    print("Loading 2D observations...")
    npz_obs = np.load(ROOT / "outputs/canonical_dataset/canonical_detector_pose_observations.npz")
    kpts = npz_obs["kpts"]
    scores = npz_obs["scores"]

    # Skip retargeting since it already ran
    temp_bvh = ROOT / "outputs/temp_b_retarget.bvh"
    fbx_rig = ROOT / "Manny.FBX"
    npy_out = str(temp_bvh).replace(".bvh", "_solved.npy")
    print(f"Loading already retargeted animation from {npy_out}")
    animation_data = np.load(npy_out)
    
    print("Computing Forward Kinematics on FBX rig...")
    skel = Skeleton(str(fbx_rig))
    num_frames = len(animation_data)
    num_joints = skel.num_joints
    
    # Render frames 840 to 1140 (10 seconds)
    start_f = 840
    end_f = 1140

    manny_pos = np.zeros((num_frames, num_joints, 3))
    frames_to_compute = [0] + list(range(start_f, end_f))
    for f in frames_to_compute:
        root_t = animation_data[f, 0:3]
        local_eulers = animation_data[f, 3:].reshape(num_joints, 3)
        local_rot = Rotation.from_euler('xyz', local_eulers, degrees=False).as_quat()
        pos, _ = skel.get_forward_kinematics(local_rot, root_translation=root_t)
        # Convert Manny (Z-up) to GT (Y-up) space
        manny_pos[f, :, 0] = pos[:, 0]    # X
        manny_pos[f, :, 1] = pos[:, 2]    # Z becomes Y (Up)
        manny_pos[f, :, 2] = -pos[:, 1]   # -Y becomes Z (Depth)

    # We will align X, Y, Z per frame so that they are always overlaid and bounded properly
    for f in frames_to_compute:
        if f < len(gt):
            gt_root = gt[f, 11]
            manny_root = manny_pos[f, skel.node_names.index('pelvis')]
            offset = gt_root - manny_root
            manny_pos[f] += offset

    preview_indices = [i for i, name in enumerate(skel.node_names) if name in PREVIEW_BONES]
    connections_manny = []
    preview_set = set(preview_indices)
    for i, name in enumerate(skel.node_names):
        p = skel.parents[i]
        if p != -1 and i in preview_set and p in preview_set:
            connections_manny.append((p, i))

    print("Rendering Multi-Camera + 3D FBX vs GT...")
    
    fig = plt.figure(figsize=(12, 7), facecolor='white')
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2])
    ax_cam1 = fig.add_subplot(gs[0, 0])
    ax_cam2 = fig.add_subplot(gs[0, 1])
    ax_cam3 = fig.add_subplot(gs[0, 2])
    ax_3d = fig.add_subplot(gs[1, :], projection='3d')
    
    def update(f):
        ax_cam1.clear(); ax_cam2.clear(); ax_cam3.clear(); ax_3d.clear()
        
        # 3D plot GT
        for p, c in CONNECTIONS_COCO:
            if p < len(gt[f]) and c < len(gt[f]):
                ax_3d.plot([gt[f,p,0], gt[f,c,0]], [gt[f,p,2], gt[f,c,2]], [gt[f,p,1], gt[f,c,1]], color='green', alpha=0.5, linewidth=2)
        ax_3d.plot([], [], [], color='green', label='GT (COCO)')
        
        # 3D plot Manny Rig
        for p, c in connections_manny:
            ax_3d.plot([manny_pos[f,p,0], manny_pos[f,c,0]], [manny_pos[f,p,2], manny_pos[f,c,2]], [manny_pos[f,p,1], manny_pos[f,c,1]], color='blue', alpha=0.9, linewidth=2)
        ax_3d.plot([], [], [], color='blue', label="Candidate B (Manny Rig)")

        ax_3d.set_xlim(-150, 150)
        ax_3d.set_ylim(-150, 150)
        ax_3d.set_zlim(0, 200)
        ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Z (Depth)'); ax_3d.set_zlabel('Y (Up)')
        ax_3d.legend(loc="upper right")
        
        # 2D plots
        plot_skeleton_2d(ax_cam1, kpts[f, 0], scores[f, 0], "Cam 00_11")
        plot_skeleton_2d(ax_cam2, kpts[f, 1], scores[f, 1], "Cam 00_12")
        plot_skeleton_2d(ax_cam3, kpts[f, 2], scores[f, 2], "Cam 00_23")
        
        fig.suptitle(f"Candidate B: Retargeted FBX Rig vs Ground Truth | Frame {f}", fontsize=14, fontweight='bold')
        
    ani = animation.FuncAnimation(fig, update, frames=range(start_f, end_f), interval=1000/30.0)
    out_path = OUT_VIS / "manny_retargeted_action.mp4"
    ani.save(str(out_path), writer='ffmpeg', dpi=100)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Also save as gif for the artifact
    out_gif = OUT_VIS / "manny_retargeted_action.gif"
    print(f"Converting to GIF: {out_gif}")
    os.system(f"ffmpeg -y -i {out_path} -vf 'fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' {out_gif} 2> NUL")

if __name__ == "__main__":
    main()
