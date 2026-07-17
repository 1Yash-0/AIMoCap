"""Test script: Swap L/R only for upper body on camera 00_03."""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

SEQ = "171204_pose3"
FRAME = 150
LAGS = {"00_03": 0, "00_04": 0, "00_28": 18, "00_24": -1}
cams = list(LAGS.keys())

# L/R swap pairs ONLY for upper body (Eyes, Ears, Shoulders, Elbows, Wrists)
LR_PAIRS_UPPER = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)
]

def swap_lr_upper(kpts, scores):
    kpts_new = kpts.copy()
    scores_new = scores.copy()
    for l, r in LR_PAIRS_UPPER:
        kpts_new[l], kpts_new[r] = kpts[r].copy(), kpts[l].copy()
        scores_new[l], scores_new[r] = scores[r], scores[l]
    return kpts_new, scores_new

def main():
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    K_list = [calib[cn].K.astype(np.float64) for cn in cams]
    extrinsics = []
    for cn in cams:
        R = calib[cn].R.astype(np.float64)
        t = calib[cn].t.astype(np.float64).reshape(3, 1)
        extrinsics.append((R, t))

    all_kpts2d = []
    all_scores = []
    for ci, cn in enumerate(cams):
        video_frame = FRAME - LAGS[cn]
        img_path = frames_dir / f"hd_{cn}" / f"{video_frame:08d}.jpg"
        kpts = np.full((17, 2), np.nan)
        scores = np.zeros(17)
        if img_path.exists():
            fr = cv2.imread(str(img_path))
            p = model.estimate(fr, pick="largest")
            if p:
                kpts = p[0].keypoints[:17]
                scores = p[0].scores[:17]
                
                # MANUALLY SWAP UPPER BODY 00_03
                if cn == "00_03":
                    kpts, scores = swap_lr_upper(kpts, scores)

        all_kpts2d.append(kpts)
        all_scores.append(scores)

    all_kpts2d = np.array([all_kpts2d])
    all_scores = np.array([all_scores])

    diag = triangulate_sequence_with_diagnostics(
        all_kpts2d, all_scores, K_list, extrinsics, min_conf=0.4, reproj_threshold_px=25.0
    )
    pred3d = diag.points3d[0]
    
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    fpath = gt_dir / f"body3DScene_{FRAME:08d}.json"
    with open(fpath) as fp: d = json.load(fp)
    k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
    COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
    gt3d = k[COCO_TO_PAN, :3]
    
    valid = np.isfinite(gt3d[:, 0]) & np.isfinite(pred3d[:, 0])
    errs = np.linalg.norm(gt3d[valid] - pred3d[valid], axis=1) * 10
    print(f"Frame 150 MPJPE after Upper-Body L/R swap on 00_03: {errs.mean():.1f} mm")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    BONES = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    def plot_skel(kpts, color, label):
        val = ~np.isnan(kpts[:, 0])
        xs, ys, zs = kpts[:, 0], kpts[:, 2], -kpts[:, 1]
        ax.scatter(xs[val], ys[val], zs[val], c=color, label=label, s=30)
        for p, c in BONES:
            if val[p] and val[c]:
                ax.plot([xs[p], xs[c]], [ys[p], ys[c]], [zs[p], zs[c]], c=color, lw=2)
                
    plot_skel(gt3d, 'green', 'Ground Truth')
    plot_skel(pred3d, 'red', 'Triangulated (Upper Swapped)')
    
    ax.legend()
    ax.set_xlim(-200, -100); ax.set_ylim(-50, 50); ax.set_zlim(0, 200)
    out = ROOT / "swapped_upper_viz.png"
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
