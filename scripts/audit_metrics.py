import numpy as np, json
from pathlib import Path
import sys

ROOT = Path(r'e:\Chaos\Projects\aimocap_re')
sys.path.insert(0, str(ROOT))
from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics
import cv2

SEQ = '171204_pose3'
LAGS = {'00_03': 0, '00_04': 0, '00_28': 18, '00_24': -1}
cams = list(LAGS.keys())
COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def get_jitter(pts3d):
    valid_mask = ~np.isnan(pts3d)
    diffs = pts3d[1:] - pts3d[:-1]
    valid_diff = valid_mask[1:] & valid_mask[:-1]
    jitters = []
    for i in range(len(diffs)):
        frame_j = []
        for j in range(5, 17):
            if valid_diff[i, j, 0]: frame_j.append(np.linalg.norm(diffs[i, j]))
        if frame_j: jitters.append(np.mean(frame_j))
    return np.array(jitters)

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

all_kpts2d, all_scores = [], []
for f in frames:
    kpts, scores = np.full((len(cams), 17, 2), np.nan), np.zeros((len(cams), 17))
    for ci, cn in enumerate(cams):
        ip = frames_dir / f'hd_{cn}' / f'{(f - LAGS[cn]):08d}.jpg'
        if ip.exists():
            fr = cv2.imread(str(ip))
            p = model.estimate(fr, pick='largest')
            if p:
                kpts[ci], scores[ci] = p[0].keypoints[:17], p[0].scores[:17]
    all_kpts2d.append(kpts)
    all_scores.append(scores)

diag = triangulate_sequence_with_diagnostics(np.array(all_kpts2d), np.array(all_scores), K_list, extrinsics, min_conf=0.4, reproj_threshold_px=25.0)
pts3d = diag.points3d

# Calculate Jitter
raw_jitter = get_jitter(pts3d) / 10.0 # to cm
print(f'Raw Jitter -> Median: {np.median(raw_jitter):.2f} cm/frame, p95: {np.percentile(raw_jitter, 95):.2f} cm/frame, Mean: {np.mean(raw_jitter):.2f} cm/frame')

# Calculate Root Offset
valid_mask = ~np.isnan(pts3d[:, 11, 0]) & ~np.isnan(valid_gt[:, 11, 0])
pred_root = (pts3d[:, 11] + pts3d[:, 12]) / 2.0
gt_root = (valid_gt[:, 11] + valid_gt[:, 12]) / 2.0
offsets = np.linalg.norm(pred_root[valid_mask] - gt_root[valid_mask], axis=1) / 10.0
print(f'Absolute Root Offset -> Median: {np.median(offsets):.2f} cm, p95: {np.percentile(offsets, 95):.2f} cm, Mean: {np.mean(offsets):.2f} cm')
