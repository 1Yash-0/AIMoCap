"""Compare our grid-search calibration output against Panoptic ground truth."""
import json
import numpy as np

# 1. Load ground truth
gt_path = 'data/panoptic/171204_pose1/calibration_171204_pose1.json'
with open(gt_path) as f:
    gt = json.load(f)

target_cams = ['00_00', '00_01', '00_02']
gt_R = {}
gt_t = {}
gt_K = {}
for cam in gt['cameras']:
    if cam['name'] in target_cams:
        gt_R[cam['name']] = np.array(cam['R'])
        gt_t[cam['name']] = np.array(cam['t']).reshape(3, 1)
        gt_K[cam['name']] = np.array(cam['K'])

# 2. Load our calibration
with open('outputs/panoptic_calib_new.json') as f:
    ours = json.load(f)

our_R = {}
our_t = {}
our_K = {}
for cam in ours['cameras']:
    our_R[cam['id']] = np.array(cam['R'])
    our_t[cam['id']] = np.array(cam['t']).reshape(3, 1)
    our_K[cam['id']] = np.array(cam['K'])

# 3. Compare focal lengths
print("=== FOCAL LENGTH COMPARISON ===")
for i, name in enumerate(target_cams):
    gt_f = gt_K[name][0, 0]
    our_f = our_K[i][0, 0]
    print(f"Cam {name}: GT focal={gt_f:.1f}  Ours={our_f:.1f}  Error={abs(gt_f - our_f)/gt_f*100:.1f}%")

# 4. Compare camera centers (position in world space)
print("\n=== CAMERA CENTER COMPARISON ===")
gt_centers = []
our_centers = []
for i, name in enumerate(target_cams):
    gt_C = (-gt_R[name].T @ gt_t[name]).flatten()
    our_C = (-our_R[i].T @ our_t[i]).flatten()
    gt_centers.append(gt_C)
    our_centers.append(our_C)

gt_centers = np.array(gt_centers)
our_centers = np.array(our_centers)

# 5. Procrustes alignment (our coordinate system is different from GT)
def procrustes(X, Y):
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_c = X - X_mean
    Y_c = Y - Y_mean
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
    t = Y_mean - scale * (R @ X_mean)
    return scale, R, t

s, R_align, t_align = procrustes(our_centers, gt_centers)
aligned = (s * (R_align @ our_centers.T)).T + t_align

print("After Procrustes alignment:")
for i, name in enumerate(target_cams):
    gt_m = gt_centers[i]
    our_m = aligned[i]
    error = np.linalg.norm(gt_m - our_m)
    gt_dist = np.linalg.norm(gt_m)
    print(f"  {name}: GT={gt_m}  Ours={our_m}  Err={error:.2f}cm ({error/gt_dist*100:.1f}%)")

# 6. Compare inter-camera distances (coordinate-system-independent)
print("\n=== INTER-CAMERA DISTANCE COMPARISON (cm) ===")
print("(This is independent of coordinate system alignment)")
for i in range(len(target_cams)):
    for j in range(i+1, len(target_cams)):
        gt_dist = np.linalg.norm(gt_centers[i] - gt_centers[j])
        our_dist = np.linalg.norm(our_centers[i] - our_centers[j])
        ratio = our_dist / gt_dist if gt_dist > 0 else float('inf')
        print(f"  {target_cams[i]} <-> {target_cams[j]}: GT={gt_dist:.1f}cm  Ours={our_dist:.4f} (unscaled)  Ratio={ratio:.6f}")

# 7. Check if distance RATIOS are preserved (shape of camera triangle)
print("\n=== DISTANCE RATIO PRESERVATION (shape test) ===")
gt_dists = []
our_dists = []
for i in range(len(target_cams)):
    for j in range(i+1, len(target_cams)):
        gt_dists.append(np.linalg.norm(gt_centers[i] - gt_centers[j]))
        our_dists.append(np.linalg.norm(our_centers[i] - our_centers[j]))

gt_ratios = np.array(gt_dists) / gt_dists[0]
our_ratios = np.array(our_dists) / our_dists[0]
print(f"  GT ratios:  {gt_ratios}")
print(f"  Our ratios: {our_ratios}")
print(f"  Ratio error: {np.abs(gt_ratios - our_ratios) / gt_ratios * 100}%")

# 8. Reprojection test: triangulate a few points with GT calib vs our calib
print("\n=== REPROJECTION ERROR TEST ===")
data = np.load('outputs/panoptic_multipose.npz')
keypoints = data['keypoints']  # (F, C, 133, 2)
scores = data['scores']

from aimocap.triangulate.engine import triangulate_n_views

# Pick frame 50, keypoint 0 (nose)
f_idx = 50
k_idx = 0

# Triangulate with GT
gt_P = []
gt_pts2d = []
for i, name in enumerate(target_cams):
    if scores[f_idx, i, k_idx] > 0.5:
        P = gt_K[name] @ np.hstack((gt_R[name], gt_t[name]))
        gt_P.append(P)
        gt_pts2d.append(keypoints[f_idx, i, k_idx])

if len(gt_P) >= 2:
    pt3d_gt = triangulate_n_views(np.array(gt_pts2d), gt_P)
    # Reproject
    reproj_errors_gt = []
    for idx in range(len(gt_P)):
        proj = gt_P[idx] @ np.append(pt3d_gt, 1)
        proj = proj[:2] / proj[2]
        err = np.linalg.norm(proj - gt_pts2d[idx])
        reproj_errors_gt.append(err)
    print(f"  GT calib reprojection error (nose, frame 50): {np.mean(reproj_errors_gt):.2f}px")

# Triangulate with ours
our_P = []
our_pts2d = []
for i in range(len(target_cams)):
    if scores[f_idx, i, k_idx] > 0.5:
        P = our_K[i] @ np.hstack((our_R[i], our_t[i]))
        our_P.append(P)
        our_pts2d.append(keypoints[f_idx, i, k_idx])

if len(our_P) >= 2:
    pt3d_ours = triangulate_n_views(np.array(our_pts2d), our_P)
    reproj_errors_ours = []
    for idx in range(len(our_P)):
        proj = our_P[idx] @ np.append(pt3d_ours, 1)
        proj = proj[:2] / proj[2]
        err = np.linalg.norm(proj - our_pts2d[idx])
        reproj_errors_ours.append(err)
    print(f"  Our calib reprojection error (nose, frame 50): {np.mean(reproj_errors_ours):.2f}px")

# Do a full-sequence reprojection comparison
print("\n=== FULL SEQUENCE MEAN REPROJECTION ERROR ===")
for label, K_dict, R_dict, t_dict in [
    ("GT", gt_K, gt_R, gt_t),
    ("Ours", our_K, our_R, our_t)
]:
    all_errors = []
    for fi in range(0, keypoints.shape[0], 10):  # every 10th frame
        for ki in range(17):  # body keypoints only
            P_list = []
            pts2d = []
            for ci in range(len(target_cams)):
                if scores[fi, ci, ki] > 0.5:
                    if label == "GT":
                        name = target_cams[ci]
                        P = K_dict[name] @ np.hstack((R_dict[name], t_dict[name]))
                    else:
                        P = K_dict[ci] @ np.hstack((R_dict[ci], t_dict[ci]))
                    P_list.append(P)
                    pts2d.append(keypoints[fi, ci, ki])
            if len(P_list) >= 2:
                pt3d = triangulate_n_views(np.array(pts2d), P_list)
                for idx in range(len(P_list)):
                    proj = P_list[idx] @ np.append(pt3d, 1)
                    proj = proj[:2] / proj[2]
                    all_errors.append(np.linalg.norm(proj - pts2d[idx]))
    print(f"  {label}: Mean reproj error = {np.mean(all_errors):.2f}px  Median = {np.median(all_errors):.2f}px  (N={len(all_errors)} measurements)")
