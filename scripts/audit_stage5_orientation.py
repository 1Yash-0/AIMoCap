"""Decompose error per-frame and test orientation-based swapping."""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

SEQ = "171204_pose3"
LAGS = {"00_03": 0, "00_04": 0, "00_28": 18, "00_24": -1}
cams = list(LAGS.keys())

COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
LR_PAIRS = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
BODY_JOINTS = list(range(5, 17))

def swap_lr(kpts, scores):
    k_new, s_new = kpts.copy(), scores.copy()
    for l, r in LR_PAIRS:
        k_new[l], k_new[r] = kpts[r].copy(), kpts[l].copy()
        s_new[l], s_new[r] = scores[r], scores[l]
    return k_new, s_new

def get_camera_center(R, t):
    # x_cam = R * x_world + t  => x_world = R^T * (x_cam - t)
    # Origin in cam space is [0,0,0]
    return -R.T @ t

def get_subject_forward(gt_kpts):
    # Use Nose (index 0) - Neck (Wait! GT kpts are in COCO order because we use COCO_TO_PAN)
    # COCO: Nose=0, LShoulder=5, RShoulder=6
    # Let's use (LShoulder + RShoulder)/2 for Neck
    if np.isnan(gt_kpts[0,0]) or np.isnan(gt_kpts[5,0]) or np.isnan(gt_kpts[6,0]):
        return None
    neck = (gt_kpts[5] + gt_kpts[6]) / 2.0
    nose = gt_kpts[0]
    fwd = nose - neck
    fwd[1] = 0 # Project to X-Z plane
    n = np.linalg.norm(fwd)
    if n < 1e-5: return None
    return fwd / n

def main():
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    K_list = [calib[cn].K.astype(np.float64) for cn in cams]
    extrinsics = []
    cam_centers = {}
    for cn in cams:
        R = calib[cn].R.astype(np.float64)
        t = calib[cn].t.astype(np.float64).reshape(3, 1)
        extrinsics.append((R, t))
        cam_centers[cn] = get_camera_center(R, t).flatten()
        
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"

    num_frames = 200
    all_kpts2d = []
    all_scores = []
    valid_gt = np.full((num_frames, 17, 3), np.nan)
    
    print("Extracting 2D and GT...")
    for i, f in enumerate(range(4700, 4900)):
        # Load GT
        fpath = gt_dir / f"body3DScene_{f:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                valid_gt[i] = k[COCO_TO_PAN, :3]
                
        # 2D Detections
        fkpts = []
        fsc = []
        for cn in cams:
            vf = f - LAGS[cn]
            ip = frames_dir / f"hd_{cn}" / f"{vf:08d}.jpg"
            kpt, sc = np.full((17, 2), np.nan), np.zeros(17)
            if ip.exists():
                fr = cv2.imread(str(ip))
                p = model.estimate(fr, pick="largest")
                if p:
                    kpt, sc = p[0].keypoints[:17], p[0].scores[:17]
            
            # BLIND SWAP for 00_03 (The static assumption)
            if cn == "00_03":
                kpt, sc = swap_lr(kpt, sc)
                
            fkpts.append(kpt)
            fsc.append(sc)
            
        all_kpts2d.append(fkpts)
        all_scores.append(fsc)

    all_kpts2d = np.array(all_kpts2d)
    all_scores = np.array(all_scores)

    print("Triangulating...")
    diag = triangulate_sequence_with_diagnostics(
        all_kpts2d, all_scores, K_list, extrinsics, min_conf=0.4, reproj_threshold_px=25.0
    )
    pts3d = diag.points3d
    
    # Orientation & Error Analysis
    print("\nDecomposing Error and Orientation...")
    frame_errors = []
    dot_prods = []
    spike_joints = []
    
    for i, f in enumerate(range(4700, 4900)):
        gt = valid_gt[i]
        pred = pts3d[i]
        
        fwd = get_subject_forward(gt)
        if fwd is None:
            frame_errors.append(np.nan)
            dot_prods.append(np.nan)
            spike_joints.append("")
            continue
            
        # Subject pelvis in X-Z
        pelvis = (gt[11] + gt[12]) / 2.0
        
        # Camera vector
        c03_pos = cam_centers["00_03"]
        v_cam = c03_pos - pelvis
        v_cam[1] = 0
        v_cam = v_cam / np.linalg.norm(v_cam)
        
        # Dot product ( >0 means camera is in FRONT of subject )
        dot = np.dot(fwd, v_cam)
        dot_prods.append(dot)
        
        # Errors
        val = np.isfinite(gt[:,0]) & np.isfinite(pred[:,0])
        if not np.any(val[BODY_JOINTS]):
            frame_errors.append(np.nan)
            spike_joints.append("")
            continue
            
        body_val = [j for j in BODY_JOINTS if val[j]]
        errs = np.linalg.norm(gt[body_val] - pred[body_val], axis=1) * 10 # mm
        mpjpe = np.mean(errs)
        frame_errors.append(mpjpe)
        
        if mpjpe > 100: # Spike threshold 10cm
            worst_j = body_val[np.argmax(errs)]
            spike_joints.append(f"J{worst_j}({np.max(errs):.0f}mm)")
        else:
            spike_joints.append("")

    # Output cluster analysis
    print("\n--- Frame Analysis (Static Swap on 00_03) ---")
    high_err_frames = []
    for i, f in enumerate(range(4700, 4900)):
        dot = dot_prods[i]
        if dot > 0: # Print any frame where subject is facing camera 00_03
            orientation = "FRONT" if dot > 0 else "BACK"
            mpjpe = frame_errors[i]
            if np.isnan(mpjpe):
                print(f"Frame {f:3d} | Body MPJPE: REJECTED BY TRIANGULATOR | 00_03 is {orientation:5s} (dot={dot:5.2f})")
            else:
                print(f"Frame {f:3d} | Body MPJPE: {mpjpe:5.1f}mm | 00_03 is {orientation:5s} (dot={dot:5.2f}) | Worst: {spike_joints[i]}")
            
    # Generating GIFs
    print("\nGenerating Rotating GIFs...")
    # Pick frames based on orientation:
    # We want one where 00_03 is BACK (e.g. 150), one where it's FRONT, one SIDE-ON (dot ~ 0)
    target_indices = []
    # Find BACK (dot < -0.8)
    for i, dot in enumerate(dot_prods):
        if not np.isnan(dot) and dot < -0.8:
            target_indices.append(i); break
    # Find SIDE (abs(dot) < 0.2)
    for i, dot in enumerate(dot_prods):
        if not np.isnan(dot) and abs(dot) < 0.2:
            target_indices.append(i); break
    # Find FRONT (dot > 0.8)
    for i, dot in enumerate(dot_prods):
        if not np.isnan(dot) and dot > 0.8:
            target_indices.append(i); break
            
    # Add one high error frame to see the mess
    if high_err_frames:
        target_indices.append(high_err_frames[0])
        
    target_indices = list(set(target_indices))
    if 0 not in target_indices: target_indices.append(0) # 4700
    print(f"Selected frame indices for GIF: {target_indices}")
    
    def create_gif(idx, real_f, out_path):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        gt_f = valid_gt[idx]
        pred_f = pts3d[idx]
        
        def plot_skel(ax, kpts, color, label):
            val = ~np.isnan(kpts[:, 0])
            xs, ys, zs = kpts[:, 0], kpts[:, 2], -kpts[:, 1]
            ax.scatter(xs[val], ys[val], zs[val], c=color, label=label, s=30)
            for p, c in BONES:
                if val[p] and val[c]:
                    ax.plot([xs[p], xs[c]], [ys[p], ys[c]], [zs[p], zs[c]], c=color, lw=2)
                    
        plot_skel(ax, gt_f, 'green', 'Ground Truth')
        plot_skel(ax, pred_f, 'red', 'Triangulated (Static Swap 00_03)')
        
        ax.set_title(f"Frame {real_f} | dot={dot_prods[idx]:.2f}")
        ax.legend()
        ax.set_xlim(-200, -100); ax.set_ylim(-50, 50); ax.set_zlim(0, 200)
        ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("-Y (Height)")
        
        def update(frame):
            ax.view_init(elev=15., azim=frame)
            return fig,
            
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 10), blit=False)
        ani.save(out_path, writer=PillowWriter(fps=15))
        plt.close(fig)
        
    for idx in target_indices:
        real_f = 4700 + idx
        out = ROOT / f"rot_frame_{real_f:04d}.gif"
        create_gif(idx, real_f, out)
        print(f"Saved {out}")

if __name__ == "__main__":
    main()
