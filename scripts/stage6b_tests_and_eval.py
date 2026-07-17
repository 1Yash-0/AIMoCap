import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import cv2
from aimocap.motion import (
    CameraModel,
    CanonicalSkeleton,
    SequentialCanonicalFitter,
    MultiViewObservations,
    build_multiview_observations,
    WindowedSequenceOptimizer,
    estimate_bone_lengths_robust,
    stitch_windows,
    opencv_to_canonical,
    canonical_to_opencv
)
from aimocap.calib.io import load_cameras

def run_synthetic_tests():
    print("Running synthetic tests...")
    # Coordinate round trip
    pts = np.random.rand(10, 3)
    pts_c = opencv_to_canonical(pts)
    pts_r = canonical_to_opencv(pts_c)
    assert np.allclose(pts, pts_r), "Coordinate round trip failed"
    
    # Determinant classification is built into CameraModel
    # Try corrupt camera
    try:
        CameraModel(name="bad", K=np.eye(3), R=np.eye(3)*-1, t=np.zeros(3), dist=np.zeros(5), image_size=None)
        assert False, "Should have rejected R with det -1"
    except ValueError:
        pass
        
    print("Synthetic tests passed.")

def run_real_eval():
    print("Loading data...")
    # Final B arrays
    gate1 = np.load('outputs/phase_b_gate1/gate1_arrays.npz')
    b_stage6 = gate1['b_stage6']
    gt = gate1['gt'] / 10.0  # mm to cm if gt was in mm?
    
    # Observations
    obs_data = np.load('outputs/canonical_dataset/canonical_detector_pose_observations.npz')
    kpts = obs_data['kpts']
    scores = obs_data['scores']
    
    # Calib
    cameras = load_cameras('data/panoptic/171204_pose1/calibration_171204_pose1.json', camera_names=['00_00', '00_01', '00_02'])
    img_sizes = [c.image_size for c in cameras]
    
    print("Building MultiViewObservations...")
    fps = 30.0
    obs = build_multiview_observations(kpts, scores, cameras, img_sizes, fps)
    
    print("Estimating robust bone lengths...")
    bl, report = estimate_bone_lengths_robust(obs)
    
    print("Initializing Sequential Fitter...")
    fitter = SequentialCanonicalFitter(fps=fps)
    bvh_pos_raw = CanonicalSkeleton.build_positions_from_coco(obs.points3d)
    x0_seq = fitter.optimize_sequence(bvh_pos_raw, bl)
    
    # Eval seq fitter vs GT
    # Get fk pos from x0_seq
    pos_seq, _ = WindowedSequenceOptimizer(cameras, fps, bl)._fk(x0_seq, len(x0_seq))
    
    # Compare
    gt_canon = CanonicalSkeleton.build_positions_from_coco(gt)
    b_stage6_canon = CanonicalSkeleton.build_positions_from_coco(b_stage6)
    
    def calc_mpjpe(pred, target):
        errs = []
        for f in range(len(pred)):
            valid = np.isfinite(target[f]).all(axis=1) & np.isfinite(pred[f]).all(axis=1)
            if valid.sum() > 3:
                root_offset = pred[f, 0] - target[f, 0]
                err = np.linalg.norm((pred[f, valid] - root_offset) - target[f, valid], axis=1).mean()
                errs.append(err)
        return np.mean(errs) * 10.0 if errs else 999.0
        
    fitter_mpjpe = calc_mpjpe(pos_seq, gt_canon)
    b_mpjpe = calc_mpjpe(b_stage6_canon, gt_canon)
    
    print(f"Candidate B RC-MPJPE: {b_mpjpe:.1f} mm")
    print(f"Sequential Fitter RC-MPJPE: {fitter_mpjpe:.1f} mm")
    
    # Now run WindowedSequenceOptimizer
    print("Running Windowed Sequence Optimizer (test on first 300 frames for speed)...")
    opt = WindowedSequenceOptimizer(cameras, fps, bl, b_stage6)
    
    win_len = 60
    overlap = 20
    stride = win_len - overlap
    
    F_opt = min(300, len(x0_seq))
    
    windows = []
    indices = []
    
    for start in range(0, F_opt, stride):
        end = min(start + win_len, F_opt)
        idx = np.arange(start, end)
        if len(idx) < 10: break
        
        x0_win = x0_seq[idx]
        x_opt = opt.optimize_window(obs, idx, x0_win.flatten())
        windows.append(x_opt.reshape(len(idx), -1))
        indices.append(idx)
        
    print("Stitching windows...")
    final_x = stitch_windows(windows, indices, F_opt, x0_seq.shape[1])
    
    final_pos, _ = opt._fk(final_x, F_opt)
    
    opt_mpjpe = calc_mpjpe(final_pos, gt_canon[:F_opt])
    print(f"Window Optimizer RC-MPJPE: {opt_mpjpe:.1f} mm")
    
    metrics = {
        "bone_lengths": report,
        "mpjpe": {
            "Candidate_B": round(b_mpjpe, 1),
            "Sequential_Fitter": round(fitter_mpjpe, 1),
            "Windowed_Optimizer": round(opt_mpjpe, 1)
        }
    }
    
    with open('outputs/stage6b_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    run_synthetic_tests()
    run_real_eval()
