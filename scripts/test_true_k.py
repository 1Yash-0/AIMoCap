import json
import numpy as np
from pathlib import Path
from aimocap.calib.extrinsics import calibrate_all, align_to_floor
from aimocap.calib.io import load_calibration

# 1. Load ground truth
gt_path = 'data/panoptic/171204_pose1/calibration_171204_pose1.json'
with open(gt_path) as f:
    gt = json.load(f)

gt_K = {}
gt_centers = {}
target_cams = ['00_00', '00_01', '00_02']

for cam in gt['cameras']:
    if cam['name'] in target_cams:
        gt_K[cam['name']] = np.array(cam['K'])
        R = np.array(cam['R'])
        t = np.array(cam['t']).reshape(3, 1)
        C = (-R.T @ t).flatten()
        gt_centers[cam['name']] = C

print("--- Ground Truth Centers (cm) ---")
for name in target_cams:
    print(f"{name}: {gt_centers[name]}")

# 2. Setup K_list
K_list = [
    gt_K['00_00'],
    gt_K['00_01'],
    gt_K['00_02']
]

# 3. Load our 2D keypoints
data = np.load('outputs/panoptic_multipose.npz')
keypoints = data['keypoints']
if isinstance(keypoints, np.ndarray) and keypoints.dtype == object:
    keypoints = np.stack(keypoints)
scores = data['scores']
if isinstance(scores, np.ndarray) and scores.dtype == object:
    scores = np.stack(scores)

print(f"Keypoints shape: {keypoints.shape}")

# 4. Run our calibration using TRUE K
print("\nRunning calibrate_all with TRUE K...")
extrinsics = calibrate_all(keypoints, scores, K_list, min_conf=0.5)

# Align to floor
print("Running align_to_floor...")
extrinsics = align_to_floor(extrinsics, K_list, keypoints, scores)

# 5. Extract our centers (OpenCV space, unscaled)
our_centers_cv = []
for R, t in extrinsics:
    C = (-R.T @ t).flatten()
    our_centers_cv.append(C)

# 6. Apply scale (we will compute scale by triangulating and comparing to anthropometric standard)
from aimocap.triangulate.engine import triangulate_sequence
from aimocap.calib.scale import apply_metric_scale

skeleton3d = triangulate_sequence(keypoints, scores, K_list, extrinsics)
skeleton3d, extrinsics_scaled, sf = apply_metric_scale(skeleton3d, extrinsics)

print(f"\nComputed Metric Scale Factor: {sf:.3f}")

our_centers_scaled_internal = []
for R, t in extrinsics_scaled:
    C_cv = (-R.T @ t).flatten()
    # Convert to Y-up internal
    C_int = np.copy(C_cv)
    C_int[1] = -C_int[1]
    C_int[2] = -C_int[2]
    our_centers_scaled_internal.append(C_int)

print("\n--- Our Centers (Meters, internal Y-up space) ---")
for i, name in enumerate(target_cams):
    print(f"Cam {i} ({name}): {our_centers_scaled_internal[i]}")

# 7. Compare using Procrustes Alignment
# We must align our coordinate system to the GT coordinate system before measuring error.
# We will find scale s, rotation R, and translation t such that: s * R @ our_m + t ~= gt_m
gt_pts = np.array([gt_centers[name] / 100.0 for name in target_cams])
our_pts = np.array(our_centers_scaled_internal)

def procrustes(X, Y):
    # align X to Y
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_c = X - X_mean
    Y_c = Y - Y_mean
    
    # norm
    normX = np.linalg.norm(X_c)
    normY = np.linalg.norm(Y_c)
    X_c /= normX
    Y_c /= normY
    
    U, S, Vt = np.linalg.svd(Y_c.T @ X_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
        
    scale = normY / normX
    t = Y_mean - scale * (R @ X_mean.T).T
    
    return scale, R, t

s_align, R_align, t_align = procrustes(our_pts, gt_pts)

print("\n--- Comparison (Meters, after Procrustes alignment) ---")
aligned_pts = (s_align * (R_align @ our_pts.T)).T + t_align

for i, name in enumerate(target_cams):
    gt_m = gt_pts[i]
    our_m = aligned_pts[i]
    error = np.linalg.norm(gt_m - our_m)
    rel_error = error / np.linalg.norm(gt_m)
    print(f"{name}:")
    print(f"  GT:   {gt_m}")
    print(f"  Ours: {our_m}")
    print(f"  Err:  {error:.3f}m ({rel_error*100:.1f}%)")

