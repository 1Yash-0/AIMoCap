import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import numpy as np
from aimocap.motion import CanonicalSkeleton, MotionOptimizer

def evaluate_stage6b():
    # 1. Load data
    pts3d_clean = np.load('outputs/stage4_2_knee_rescue/pts3d_clean.npy')
    gt = np.load('outputs/stage4_2_knee_rescue/gt_kpts.npy')
    
    # 2. Build canonical positions
    bvh_pos = CanonicalSkeleton.build_positions_from_coco(pts3d_clean)
    
    # We treat all frames as measured for bone length estimation since stage 4.2 fills gaps robustly
    recon_mask = np.zeros(bvh_pos.shape[0], dtype=bool)
    
    # 3. Estimate Bone Lengths
    opt = MotionOptimizer(fps=29.97)
    bl = opt.estimate_bone_lengths(bvh_pos, recon_mask)
    
    # 4. Optimize Sequence (FK Solve)
    root_pos, local_quats, fk_pos = opt.optimize_sequence(bvh_pos, bl)
    
    # 5. Evaluate vs GT
    gt_bvh = CanonicalSkeleton.build_positions_from_coco(gt)
    errs = []
    for f in range(gt.shape[0]):
        valid = np.isfinite(gt_bvh[f]).all(axis=1)
        if valid.sum() > 3:
            # We align by root for a fair comparison since GT and our coords might differ in global translation
            root_offset = fk_pos[f, 0] - gt_bvh[f, 0]
            err = np.linalg.norm((fk_pos[f, valid] - root_offset) - gt_bvh[f, valid], axis=1).mean()
            errs.append(err)
            
    mean_err_mm = np.mean(errs) * 10.0
    
    # 6. Check Rotation Smoothness (Angular Velocity)
    # Convert quats to Rotation
    from scipy.spatial.transform import Rotation
    max_ang_vels = []
    for j in range(1, CanonicalSkeleton.num_joints()):
        quats_j = local_quats[:, j, :]
        rots = Rotation.from_quat(quats_j)
        # Compute delta angles
        deltas = []
        for i in range(1, len(rots)):
            delta = rots[i-1].inv() * rots[i]
            deltas.append(delta.magnitude())
        max_ang_vels.append(np.percentile(deltas, 95) * np.rad2deg(1.0) * opt.fps) # deg/sec
        
    avg_smoothness = np.mean(max_ang_vels)
    
    # Save results
    metrics = {
        "bone_lengths": {CanonicalSkeleton.NAMES[i]: round(bl[i], 3) for i in range(len(bl))},
        "mean_mpjpe_mm_vs_gt": round(mean_err_mm, 2),
        "angular_velocity_p95_deg_sec": round(avg_smoothness, 2)
    }
    
    with open('outputs/stage6b_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    evaluate_stage6b()
