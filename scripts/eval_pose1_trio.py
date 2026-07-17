import numpy as np, json, cv2
from pathlib import Path
import sys

ROOT = Path(r'e:\Chaos\Projects\aimocap_re')
sys.path.insert(0, str(ROOT))
from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics
from aimocap.math.coords import internal_to_opencv
from aimocap.retarget.engine import retarget_to_fbx
from aimocap.math.filter import filter_skeleton3d

SEQ         = '171204_pose1'
CAMS        = ['00_08', '00_09', '00_26']
START_FRAME = 149   # skip first 5s of setup
N_FRAMES    = 300

print('=== Step 1: Loading Extracted 2D Poses ===')
frames_dir = ROOT / 'data/panoptic' / SEQ / 'sync_frames'
calib      = load_panoptic_calib(ROOT / f'data/panoptic/{SEQ}/calibration_{SEQ}.json')
K_list     = [calib[cn].K.astype(np.float64) for cn in CAMS]
extrinsics = [(calib[cn].R.astype(np.float64), calib[cn].t.astype(np.float64).reshape(3, 1)) for cn in CAMS]
model      = PoseEstimator()

all_kpts2d, all_scores = [], []
for f in range(N_FRAMES):
    kpts   = np.full((len(CAMS), 17, 2), np.nan)
    scores = np.zeros((len(CAMS), 17))
    for ci, cam in enumerate(CAMS):
        ip = frames_dir / f'hd_{cam}' / f'{f:08d}.jpg'
        if ip.exists():
            fr = cv2.imread(str(ip))
            p  = model.estimate(fr, pick='largest')
            if p:
                kpts[ci]   = p[0].keypoints[:17]
                scores[ci] = p[0].scores[:17]
    all_kpts2d.append(kpts)
    all_scores.append(scores)
    if (f+1) % 50 == 0: print(f'  Pose {f+1}/{N_FRAMES}')

print('\n=== Step 2: Triangulating ===')
diag   = triangulate_sequence_with_diagnostics(
    np.array(all_kpts2d), np.array(all_scores), K_list, extrinsics,
    min_conf=0.4, reproj_threshold_px=25.0)
pts3d  = diag.points3d

print('\n=== Step 3: Filtering & Retargeting ===')
# Filter the skeleton (removes jitter)
filtered_pts = filter_skeleton3d(pts3d)

out_npz = ROOT / 'outputs' / 'pose1_trio.npz'
out_npz.parent.mkdir(exist_ok=True)
np.savez(out_npz, skeleton3d=filtered_pts)

out_bvh = ROOT / 'outputs' / 'pose1_trio.bvh'
retarget_to_fbx(
    triangulated_npz=str(out_npz),
    fbx_rig_path=str(ROOT / 'Manny.FBX'),
    output_bvh=str(out_bvh)
)
print(f'\nDone! Output saved to:')
print(f'  - {out_npz}')
print(f'  - {out_bvh}')
