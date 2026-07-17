import numpy as np, json, math
from pathlib import Path
import sys
import cv2

ROOT = Path(r'e:\Chaos\Projects\aimocap_re')
sys.path.insert(0, str(ROOT))
from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.math.triangulate import triangulate_robust

SEQ = '171204_pose3'
LAGS = {'00_03': 0, '00_04': 0, '00_28': 18, '00_24': -1}
cams = list(LAGS.keys())
COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

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
cam_centers = [-extrinsics[c][0].T @ extrinsics[c][1] for c in range(len(cams))]

def angle_between_cams(ci, cj, target_pt):
    ray_i = target_pt - cam_centers[ci].flatten()
    ray_j = target_pt - cam_centers[cj].flatten()
    ray_i /= np.linalg.norm(ray_i)
    ray_j /= np.linalg.norm(ray_j)
    dot = np.clip(np.sum(ray_i * ray_j), -1.0, 1.0)
    return math.degrees(math.acos(dot))

# Data collection
two_cam_frames = []
lr_swap_incidents = []

THRESHOLD = 1.7

for f_idx, f in enumerate(frames):
    kpts, scores = np.full((len(cams), 17, 2), np.nan), np.zeros((len(cams), 17))
    aspect_ratios = np.zeros(len(cams))
    
    for ci, cn in enumerate(cams):
        ip = frames_dir / f'hd_{cn}' / f'{(f - LAGS[cn]):08d}.jpg'
        if ip.exists():
            p = model.estimate(cv2.imread(str(ip)), pick='largest')
            if p:
                kpts[ci], scores[ci] = p[0].keypoints[:17], p[0].scores[:17]
                w = p[0].bbox[2] - p[0].bbox[0]
                h = p[0].bbox[3] - p[0].bbox[1]
                aspect_ratios[ci] = h / w if w > 0 else 0
                
    # Identify surviving cameras for the lower body
    surviving_cams = [ci for ci in range(len(cams)) if aspect_ratios[ci] > THRESHOLD]
    
    # 1. 2-Camera Baseline Check
    if len(surviving_cams) == 2:
        c1, c2 = surviving_cams
        gt_hip = (valid_gt[f_idx, 11] + valid_gt[f_idx, 12]) / 2.0
        if not np.isnan(gt_hip[0]):
            baseline_ang = angle_between_cams(c1, c2, gt_hip)
            
            # Triangulate l_hip with just these 2 cameras
            pt_lhip = np.full(3, np.nan)
            if scores[c1, 11] >= 0.4 and scores[c2, 11] >= 0.4:
                pt_lhip = triangulate_robust(kpts[[c1, c2], 11], [P_list[c1], P_list[c2]], scores[[c1, c2], 11])
                
            err = np.nan
            if np.isfinite(pt_lhip).all() and np.isfinite(valid_gt[f_idx, 11]).all():
                err = float(np.linalg.norm(pt_lhip - valid_gt[f_idx, 11]))
                
            two_cam_frames.append({
                'frame': f,
                'cams': (cams[c1], cams[c2]),
                'angle': baseline_ang,
                'l_hip_err': err
            })
            
    # 2. L/R Swap Check on SURVIVING cameras
    for ci in surviving_cams:
        for hip_idx, opp_idx, name in [(11, 12, 'l_hip'), (12, 11, 'r_hip')]:
            if scores[ci, hip_idx] >= 0.4 and not np.isnan(valid_gt[f_idx, hip_idx, 0]) and not np.isnan(valid_gt[f_idx, opp_idx, 0]):
                p2d = kpts[ci, hip_idx]
                gt2d_correct = project(valid_gt[f_idx, hip_idx], P_list[ci])
                gt2d_opp = project(valid_gt[f_idx, opp_idx], P_list[ci])
                
                if not np.isnan(gt2d_correct[0]) and not np.isnan(gt2d_opp[0]):
                    dist_correct = np.linalg.norm(p2d - gt2d_correct)
                    dist_opp = np.linalg.norm(p2d - gt2d_opp)
                    
                    if dist_opp < dist_correct and dist_opp < 30.0:
                        # Log if it's closer to the opposite hip AND reasonably close to it (not just floating in space)
                        lr_swap_incidents.append({
                            'frame': f,
                            'cam': cams[ci],
                            'joint': name,
                            'dist_correct': float(dist_correct),
                            'dist_opp': float(dist_opp)
                        })

# Summarize 2-Camera baseline results
print(f'\n--- 2-CAMERA SURVIVOR FRAMES (AR > {THRESHOLD}) ---')
print(f'Total 2-camera frames: {len(two_cam_frames)}')

# Bin by baseline angle
bins = [0, 15, 30, 45, 60, 90, 180]
for i in range(len(bins)-1):
    in_bin = [x for x in two_cam_frames if bins[i] <= x['angle'] < bins[i+1]]
    if in_bin:
        errs = [x['l_hip_err'] for x in in_bin if not math.isnan(x['l_hip_err'])]
        if errs:
            print(f'Angle [{bins[i]:>2}° - {bins[i+1]:>3}°]: {len(in_bin):>2} frames -> Mean 3D Err: {np.mean(errs):.1f} cm (Max: {np.max(errs):.1f} cm)')
        else:
            print(f'Angle [{bins[i]:>2}° - {bins[i+1]:>3}°]: {len(in_bin):>2} frames -> All NaNs (could not triangulate)')

# List the specific narrow baseline frames if any
narrow = [x for x in two_cam_frames if x['angle'] < 30.0]
if narrow:
    print(f'\nNarrow baseline (< 30°) pairs found:')
    for x in narrow:
        print(f"  Frame {x['frame']}: {x['cams']}, angle={x['angle']:.1f}°, l_hip error={x['l_hip_err']:.1f} cm")
else:
    print('\nNo narrow baseline (< 30°) pairs found among the surviving 2-camera frames.')

# Summarize L/R Swap Check
print(f'\n--- L/R SWAP CHECK ON AR-SURVIVING CAMERAS ---')
if not lr_swap_incidents:
    print('No clear L/R hip swaps detected on surviving cameras.')
else:
    print(f'Found {len(lr_swap_incidents)} instances of L/R hip swap on surviving cameras:')
    for x in lr_swap_incidents[:10]: # Print first 10
        print(f"  Frame {x['frame']} on {x['cam']}: {x['joint']} predicted closer to opposite GT ({x['dist_opp']:.1f}px) than correct GT ({x['dist_correct']:.1f}px)")
    if len(lr_swap_incidents) > 10:
        print(f'  ... and {len(lr_swap_incidents)-10} more.')
