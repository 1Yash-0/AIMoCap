import numpy as np, json, math
from pathlib import Path
import sys
import cv2

ROOT = Path(r'e:\Chaos\Projects\aimocap_re')
sys.path.insert(0, str(ROOT))
from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

SEQ = '171204_pose3'
LAGS = {'00_03': 0, '00_04': 0, '00_28': 18, '00_24': -1}
cams = list(LAGS.keys())
COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
JOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]

print('Loading GT...')
gt_dir = ROOT / f'data/panoptic/{SEQ}/hdPose3d_stage1_coco19'
frames = list(range(150, 450))
valid_gt = []
for f in frames:
    fpath = gt_dir / f'body3DScene_{f:08d}.json'
    if fpath.exists():
        with open(fpath) as fp: d = json.load(fp)
        if d.get('bodies'):
            k = np.array(d['bodies'][0]['joints19']).reshape(19, 4)
            pts = k[COCO_TO_PAN, :3]
            scores = k[COCO_TO_PAN, 3]
            pts[scores == 0] = np.nan
            valid_gt.append(pts)
        else: valid_gt.append(np.full((17, 3), np.nan))
    else: valid_gt.append(np.full((17, 3), np.nan))
valid_gt = np.array(valid_gt)

print('Loading 2D & Triangulating...')
model = PoseEstimator()
frames_dir = ROOT / f'data/panoptic/{SEQ}/sync_frames'
calib = load_panoptic_calib(ROOT / f'data/panoptic/{SEQ}/calibration_{SEQ}.json')
K_list = [calib[cn].K.astype(np.float64) for cn in cams]
extrinsics = [(calib[cn].R.astype(np.float64), calib[cn].t.astype(np.float64).reshape(3, 1)) for cn in cams]
P_list = [K_list[c] @ np.hstack(extrinsics[c]) for c in range(len(cams))]

all_kpts2d, all_scores, all_bboxes = [], [], []
for f in frames:
    kpts, scores = np.full((len(cams), 17, 2), np.nan), np.zeros((len(cams), 17))
    bboxes = np.full((len(cams), 4), np.nan)
    for ci, cn in enumerate(cams):
        ip = frames_dir / f'hd_{cn}' / f'{(f - LAGS[cn]):08d}.jpg'
        if ip.exists():
            fr = cv2.imread(str(ip))
            p = model.estimate(fr, pick='largest')
            if p:
                kpts[ci], scores[ci] = p[0].keypoints[:17], p[0].scores[:17]
                bboxes[ci] = p[0].bbox
    all_kpts2d.append(kpts)
    all_scores.append(scores)
    all_bboxes.append(bboxes)

diag = triangulate_sequence_with_diagnostics(np.array(all_kpts2d), np.array(all_scores), K_list, extrinsics, min_conf=0.4, reproj_threshold_px=25.0)
pts3d = diag.points3d

# Calculate Root-Aligned MPJPE
valid_mask = ~np.isnan(pts3d[:, 11, 0]) & ~np.isnan(valid_gt[:, 11, 0])
pred_root = (pts3d[:, 11] + pts3d[:, 12]) / 2.0
gt_root = (valid_gt[:, 11] + valid_gt[:, 12]) / 2.0
pred_centered = pts3d - pred_root[:, None, :]
gt_centered = valid_gt - gt_root[:, None, :]
errs = np.linalg.norm(pred_centered - gt_centered, axis=2) # Shape: (frames, 17)
# Mean over body joints (5-17)
body_errs = errs[:, 5:17]
mean_body_errs = np.nanmean(body_errs, axis=1)

# Mask invalid frames
mean_body_errs[~valid_mask] = -1
worst_frame_idx = np.nanargmax(mean_body_errs)
worst_frame_err = mean_body_errs[worst_frame_idx]
worst_f = frames[worst_frame_idx]

print(f'\n--- WORST FRAME FOUND ---')
print(f'Frame: {worst_f} (Index: {worst_frame_idx}), Mean Body Root-Aligned MPJPE: {worst_frame_err:.2f} cm')

print(f'\n--- PER-JOINT ERROR AT FRAME {worst_f} ---')
frame_errs_j = errs[worst_frame_idx]
worst_j_indices = np.argsort(frame_errs_j)[::-1]
for j in worst_j_indices:
    if j >= 5 and not np.isnan(frame_errs_j[j]):
        print(f'{JOINT_NAMES[j]:>12}: {frame_errs_j[j]:.2f} cm')

print(f'\n--- BBOX STATS AT FRAME {worst_f} ---')
frame_bboxes = all_bboxes[worst_frame_idx]
for ci, cn in enumerate(cams):
    if not np.isnan(frame_bboxes[ci, 0]):
        w = frame_bboxes[ci, 2] - frame_bboxes[ci, 0]
        h = frame_bboxes[ci, 3] - frame_bboxes[ci, 1]
        ar = h / w if w > 0 else 0
        print(f'{cn:>8}: w={w:.1f}, h={h:.1f}, aspect (h/w)={ar:.2f}')
    else:
        print(f'{cn:>8}: No bbox detected')

print(f'\n--- BASELINE ANGLE AND 2D LABELS FOR WORST JOINT ---')
worst_j = worst_j_indices[0] # The absolute worst joint overall
print(f'Focusing on worst joint: {JOINT_NAMES[worst_j]}')
def get_camera_ray(C_idx, point3d):
    # point3d is in world coordinates. Camera center is -R^T * t
    R = extrinsics[C_idx][0]
    t = extrinsics[C_idx][1]
    C_pos = -R.T @ t
    ray = point3d - C_pos.flatten()
    return ray / np.linalg.norm(ray)

# Which cameras had confident detections?
conf_cams = []
for ci in range(len(cams)):
    if all_scores[worst_frame_idx][ci, worst_j] >= 0.4:
        conf_cams.append(ci)

print(f'Confident views: {[cams[ci] for ci in conf_cams]}')
if len(conf_cams) >= 2:
    angles = []
    # Find max baseline angle between any two confident cameras
    import itertools
    for c1, c2 in itertools.combinations(conf_cams, 2):
        ray1 = get_camera_ray(c1, valid_gt[worst_frame_idx, worst_j])
        ray2 = get_camera_ray(c2, valid_gt[worst_frame_idx, worst_j])
        dot = np.clip(np.dot(ray1, ray2), -1.0, 1.0)
        angles.append(math.degrees(math.acos(dot)))
    print(f'Max baseline angle between confident views: {max(angles):.1f}°')
else:
    print('Not enough confident views to form a baseline.')

# L/R Swap Check
gt_pt_worst = valid_gt[worst_frame_idx, worst_j]
# Find the opposite joint
if worst_j % 2 == 1:
    opp_j = worst_j + 1 # left to right
else:
    opp_j = worst_j - 1 # right to left
gt_pt_opp = valid_gt[worst_frame_idx, opp_j]

def project(pt3d, P):
    X = np.append(pt3d, 1.0)
    p = P @ X
    return p[:2] / p[2]

for ci in conf_cams:
    p2d = all_kpts2d[worst_frame_idx][ci, worst_j]
    gt2d_worst = project(gt_pt_worst, P_list[ci])
    gt2d_opp = project(gt_pt_opp, P_list[ci])
    
    dist_to_correct = np.linalg.norm(p2d - gt2d_worst)
    dist_to_opp = np.linalg.norm(p2d - gt2d_opp)
    
    print(f'Cam {cams[ci]}:')
    print(f'  Pred 2D: {p2d}')
    print(f'  GT 2D (correct): {gt2d_worst} (dist: {dist_to_correct:.1f}px)')
    print(f'  GT 2D (opposite): {gt2d_opp} (dist: {dist_to_opp:.1f}px)')
    if dist_to_opp < dist_to_correct:
        print(f'  >> SWAP DETECTED! Prediction is closer to {JOINT_NAMES[opp_j]}')
