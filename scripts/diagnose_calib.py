"""Honest diagnostic: compare our computed calibration to Panoptic ground truth."""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Load ground truth
with open('data/panoptic/171204_pose1/calibration_171204_pose1.json') as f:
    gt = json.load(f)

gt_centers = {}
for cam in gt['cameras']:
    if cam['name'] in ['00_00', '00_01', '00_02']:
        R = np.array(cam['R'])
        t = np.array(cam['t'])
        C = (-R.T @ t).flatten()
        gt_centers[cam['name']] = C

print("=== GROUND TRUTH camera centers (cm) ===")
for name, C in gt_centers.items():
    print(f"  {name}: X={C[0]:8.1f}  Y={C[1]:8.1f}  Z={C[2]:8.1f}")

# 2. Load our computed calibration
with open('outputs/panoptic_calib.json') as f:
    ours = json.load(f)

our_centers_cv = []
for cam in ours['cameras']:
    R = np.array(cam['R'])
    t = np.array(cam['t']).reshape(3,1)
    C = (-R.T @ t).flatten()
    our_centers_cv.append(C)

print("\n=== OUR computed camera centers (unscaled, OpenCV space) ===")
for i, C in enumerate(our_centers_cv):
    print(f"  Cam {i}: X={C[0]:8.4f}  Y={C[1]:8.4f}  Z={C[2]:8.4f}")

# 3. Load skeleton to see scale
skel = np.load('outputs/panoptic_skeleton3d.npz')['skeleton3d']
valid = skel[~np.isnan(skel).any(axis=-1)]
print(f"\n=== SKELETON stats (after 7.935x scale) ===")
print(f"  Y range: {valid[:,1].min():.2f} to {valid[:,1].max():.2f}")
print(f"  X range: {valid[:,0].min():.2f} to {valid[:,0].max():.2f}")
print(f"  Z range: {valid[:,2].min():.2f} to {valid[:,2].max():.2f}")

# 4. Convert our centers to internal Y-up space and apply the same 7.935 scale
scale = 7.935
our_centers_internal = []
for C in our_centers_cv:
    ci = np.copy(C)
    ci[1] = -ci[1]  # flip Y
    ci[2] = -ci[2]  # flip Z
    ci *= scale
    our_centers_internal.append(ci)

print(f"\n=== OUR camera centers (Y-up internal, scaled by {scale}) ===")
for i, C in enumerate(our_centers_internal):
    print(f"  Cam {i}: X={C[0]:8.2f}  Y={C[1]:8.2f}  Z={C[2]:8.2f}")

# 5. Compare to GT
# GT is in cm. Our is in meters * scale = meters. GT is cm, so convert GT to meters.
print(f"\n=== COMPARISON (GT in meters, Ours in meters) ===")
gt_names = ['00_00', '00_01', '00_02']
for i, name in enumerate(gt_names):
    gt_m = gt_centers[name] / 100.0  # cm -> m
    ours_m = our_centers_internal[i]
    print(f"  Cam {i} ({name}):")
    print(f"    GT:   X={gt_m[0]:7.2f}  Y={gt_m[1]:7.2f}  Z={gt_m[2]:7.2f}")
    print(f"    Ours: X={ours_m[0]:7.2f}  Y={ours_m[1]:7.2f}  Z={ours_m[2]:7.2f}")

# 6. Render a single-frame PNG showing skeleton + cameras with CORRECT scale
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot skeleton frame 0
from aimocap.pose.keypoints import SKELETON_133
pts = skel[0]
for s, e in SKELETON_133:
    p1, p2 = pts[s], pts[e]
    if not np.isnan(p1).any() and not np.isnan(p2).any():
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], color='blue', linewidth=2)

# Plot cameras (correctly scaled)
for i, C in enumerate(our_centers_internal):
    ax.scatter(C[0], C[2], C[1], c='red', marker='^', s=150)
    ax.text(C[0], C[2], C[1], f"  C{i}", color='red', fontsize=12)

# Floor grid
ax.set_xlabel('X (Right)')
ax.set_ylabel('Z (Back)')
ax.set_zlabel('Y (Up)')
ax.set_title('Diagnostic: Skeleton + Cameras (both scaled)')
ax.view_init(elev=20., azim=-45)

plt.tight_layout()
plt.savefig('outputs/diagnostic_frame0.png', dpi=100)
print(f"\nSaved diagnostic PNG to outputs/diagnostic_frame0.png")
