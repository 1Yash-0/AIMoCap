"""Empirical sweep of 2D Left/Right mislabeling vs Camera Orientation."""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = "171204_pose3"
LAGS = {"00_03": 0, "00_04": 0, "00_28": 18, "00_24": -1}
cams = list(LAGS.keys())
COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

# Pair names and indices
PAIRS = {
    "Eye": (1, 2), "Ear": (3, 4), "Shoulder": (5, 6),
    "Elbow": (7, 8), "Wrist": (9, 10), "Hip": (11, 12),
    "Knee": (13, 14), "Ankle": (15, 16)
}

def get_camera_center(R, t): return -R.T @ t

def get_torso_normal(gt_kpts):
    # User requested shoulder-line / hip-line normal
    # L_shoulder=5, R_shoulder=6, L_hip=11, R_hip=12
    for j in [5, 6, 11, 12]:
        if np.isnan(gt_kpts[j, 0]): return None
        
    mid_shoulder = (gt_kpts[5] + gt_kpts[6]) / 2.0
    mid_hip = (gt_kpts[11] + gt_kpts[12]) / 2.0
    spine = mid_shoulder - mid_hip # Points UP
    shoulder_line = gt_kpts[5] - gt_kpts[6] # L - R (Points +X)
    
    # In Panoptic (Y-down, Z-forward), +X cross -Y = -Z (Backwards)
    # Wait, Spine is UP (-Y), Shoulder_line is +X.
    # cross(Spine, Shoulder_line) = cross(-Y, X) = +Z (Forwards)
    fwd = np.cross(spine, shoulder_line)
    fwd[1] = 0
    n = np.linalg.norm(fwd)
    if n < 1e-5: return None
    return fwd / n

def get_nose_neck_forward(gt_kpts):
    if np.isnan(gt_kpts[0,0]) or np.isnan(gt_kpts[5,0]) or np.isnan(gt_kpts[6,0]): return None
    neck = (gt_kpts[5] + gt_kpts[6]) / 2.0
    fwd = gt_kpts[0] - neck
    fwd[1] = 0
    n = np.linalg.norm(fwd)
    if n < 1e-5: return None
    return fwd / n

def main():
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    cam_centers = {}
    for cn in cams:
        R = calib[cn].R.astype(np.float64)
        t = calib[cn].t.astype(np.float64).reshape(3, 1)
        cam_centers[cn] = get_camera_center(R, t).flatten()

    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    
    # We will sample frames across the sequence to get a full range of orientations
    sample_frames = list(range(0, 5000, 50))
    
    # Store empirical results
    # Format: { pair_name: { cn: { 'dots': [], 'dx': [], 'is_hallucinated': [] } } }
    results = {p: {cn: {'dots': [], 'dx': [], 'is_hallucinated': []} for cn in cams} for p in PAIRS.keys()}
    
    print("Sweeping empirical 2D detections for all 5000 frames...")
    count_processed = 0
    for f in range(5000):
        fpath = gt_dir / f"body3DScene_{f:08d}.json"
        if not fpath.exists(): continue
        with open(fpath) as fp: d = json.load(fp)
        if not d.get("bodies"): continue
        gt = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)[COCO_TO_PAN, :3]
        
        fwd = get_torso_normal(gt)
        if fwd is None: continue
        
        nose_fwd = get_nose_neck_forward(gt)
        if nose_fwd is not None and np.dot(fwd, nose_fwd) < 0:
            fwd = -fwd
            
        pelvis = (gt[11] + gt[12]) / 2.0
        
        count_processed += 1
        
        for cn in cams:
            v_cam = cam_centers[cn] - pelvis
            v_cam[1] = 0
            v_cam = v_cam / np.linalg.norm(v_cam)
            dot = np.dot(fwd, v_cam)
            
            video_frame = f - LAGS[cn]
            ip = frames_dir / f"hd_{cn}" / f"{video_frame:08d}.jpg"
            if not ip.exists(): continue
            fr = cv2.imread(str(ip))
            p = model.estimate(fr, pick="largest")
            if not p: continue
            kpt = p[0].keypoints[:17]
            sc = p[0].scores[:17]
            
            if f == 150 and cn == "00_03":
                print(f"FRAME 150 CAM 00_03 RAW ARMS:")
                print(f"  Shoulder L: {kpt[5,0]:.1f}, R: {kpt[6,0]:.1f}")
                print(f"  Elbow    L: {kpt[7,0]:.1f}, R: {kpt[8,0]:.1f}")
                print(f"  Wrist    L: {kpt[9,0]:.1f}, R: {kpt[10,0]:.1f}")
            
            for p_name, (l_idx, r_idx) in PAIRS.items():
                if sc[l_idx] > 0.4 and sc[r_idx] > 0.4:
                    dx = kpt[r_idx, 0] - kpt[l_idx, 0]
                    is_hallucinated = (dot < -0.2 and dx < 0) or (dot > 0.2 and dx > 0)
                    
                    results[p_name][cn]['dots'].append(dot)
                    results[p_name][cn]['dx'].append(dx)
                    results[p_name][cn]['is_hallucinated'].append(is_hallucinated)

    print(f"Processed {count_processed} frames out of 5000.")

    # Analyze thresholds
    print("\n--- Empirical Mislabeling Analysis ---")
    for p_name in PAIRS.keys():
        print(f"\nJoint Pair: {p_name}")
        for cn in ["00_03", "00_04", "00_28", "00_24"]:
            data = results[p_name][cn]
            if not data['dots']: continue
            dots = np.array(data['dots'])
            hallucinations = np.array(data['is_hallucinated'])
            
            # Buckets
            bins = np.linspace(-1.0, 1.0, 11) # 10 buckets
            inds = np.digitize(dots, bins)
            
            summary = []
            for b in range(1, len(bins)):
                mask = (inds == b)
                if np.any(mask):
                    hall_rate = np.mean(hallucinations[mask]) * 100
                    count = np.sum(mask)
                    summary.append(f"[{bins[b-1]:5.1f} to {bins[b]:5.1f}]: {hall_rate:5.1f}% fail ({count:2d})")
                    
            print(f"  Camera {cn}:")
            for s in summary:
                print(f"    {s}")

if __name__ == "__main__":
    main()
