import numpy as np, json, math
from pathlib import Path
import sys
import cv2

ROOT = Path(r'e:\Chaos\Projects\aimocap_re')
sys.path.insert(0, str(ROOT))
from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = '171204_pose3'
LAGS = {'00_03': 0, '00_04': 0, '00_28': 18, '00_24': -1}
cams = list(LAGS.keys())
COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
JOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
HIP_KNEE_IDX = [11, 12, 13, 14]

def project(pt3d, P):
    X = np.append(pt3d, 1.0)
    p = P @ X
    if p[2] <= 0: return np.full(2, np.nan)
    return p[:2] / p[2]

gt_dir = ROOT / f'data/panoptic/{SEQ}/hdPose3d_stage1_coco19'
frames = list(range(150, 450))

print('Loading GT...')
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

print('Loading 2D...')
model = PoseEstimator()
frames_dir = ROOT / f'data/panoptic/{SEQ}/sync_frames'
calib = load_panoptic_calib(ROOT / f'data/panoptic/{SEQ}/calibration_{SEQ}.json')
K_list = [calib[cn].K.astype(np.float64) for cn in cams]
extrinsics = [(calib[cn].R.astype(np.float64), calib[cn].t.astype(np.float64).reshape(3, 1)) for cn in cams]
P_list = [K_list[c] @ np.hstack(extrinsics[c]) for c in range(len(cams))]

# Data collection for aspect ratio sweep
aspect_ratios = []
hip_knee_errs_2d = []
camera_aspects = {cn: [] for cn in cams}

# Specific check for frame 390
f390_idx = frames.index(390)

for f_idx, f in enumerate(frames):
    kpts, scores = np.full((len(cams), 17, 2), np.nan), np.zeros((len(cams), 17))
    bboxes = np.full((len(cams), 4), np.nan)
    for ci, cn in enumerate(cams):
        ip = frames_dir / f'hd_{cn}' / f'{(f - LAGS[cn]):08d}.jpg'
        if ip.exists():
            p = model.estimate(cv2.imread(str(ip)), pick='largest')
            if p:
                kpts[ci], scores[ci] = p[0].keypoints[:17], p[0].scores[:17]
                bboxes[ci] = p[0].bbox
                
                # Check frame 390
                if f == 390 and cn == '00_28':
                    print(f'\n--- FRAME 390, CAM 00_28 SPECIFIC CHECK ---')
                    print(f'l_hip confidence: {scores[ci, 11]:.3f}')
                    print(f'r_hip confidence: {scores[ci, 12]:.3f}')
                    
                w = bboxes[ci, 2] - bboxes[ci, 0]
                h = bboxes[ci, 3] - bboxes[ci, 1]
                ar = h / w if w > 0 else 0
                camera_aspects[cn].append(ar)
                
                # Calculate 2D errors for hips/knees
                errs_sum = 0
                valid_count = 0
                for j in HIP_KNEE_IDX:
                    if scores[ci, j] >= 0.4 and not np.isnan(valid_gt[f_idx, j, 0]):
                        gt2d = project(valid_gt[f_idx, j], P_list[ci])
                        if not np.isnan(gt2d[0]):
                            err = np.linalg.norm(kpts[ci, j] - gt2d)
                            errs_sum += err
                            valid_count += 1
                if valid_count > 0:
                    aspect_ratios.append(ar)
                    hip_knee_errs_2d.append(errs_sum / valid_count)

# Bin the aspect ratios to find the threshold empirically
print(f'\n--- EMPIRICAL SWEEP: ASPECT RATIO VS 2D HIP/KNEE ERROR ---')
ar_arr = np.array(aspect_ratios)
err_arr = np.array(hip_knee_errs_2d)

bins = [0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 3.0]
for i in range(len(bins)-1):
    mask = (ar_arr >= bins[i]) & (ar_arr < bins[i+1])
    if mask.sum() > 0:
        mean_err = np.mean(err_arr[mask])
        median_err = np.median(err_arr[mask])
        print(f'AR [{bins[i]:.1f} - {bins[i+1]:.1f}): {mask.sum():>4} instances -> Mean 2D Err: {mean_err:>6.1f}px, Median: {median_err:>6.1f}px')

# Determine threshold (e.g., where mean error explodes)
# By looking at the bins, we'll pick a threshold and count
threshold = 1.7  # We'll refine this based on the printout, but let's test 1.7 for coverage

print(f'\n--- COVERAGE CHECK WITH AR > {threshold} THRESHOLD ---')
good_cams_per_frame = []
for f_idx in range(len(frames)):
    good_cams = 0
    for cn in cams:
        if f_idx < len(camera_aspects[cn]):
            ar = camera_aspects[cn][f_idx]
            if ar > threshold:
                good_cams += 1
    good_cams_per_frame.append(good_cams)

counts = {0:0, 1:0, 2:0, 3:0, 4:0}
for c in good_cams_per_frame:
    if c in counts: counts[c] += 1
    
print(f'Over {len(frames)} frames:')
for c in range(5):
    print(f'Frames with {c} good cameras: {counts[c]} ({counts[c]/len(frames)*100:.1f}%)')

