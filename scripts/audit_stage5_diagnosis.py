"""Stage 5: Robust Triangulation Diagnosis — corrected coordinate + joint mapping"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

SEQ = "171204_pose3"
LAGS = {
    "00_03": 0,
    "00_04": 0,
    "00_28": 18,
    "00_24": -1
}

# Correct COCO-17 → Panoptic joints19 index mapping
# Panoptic joints19:
#   0=Neck, 1=Nose, 2=BodyCenter, 3=lShoulder, 4=lElbow, 5=lWrist,
#   6=rShoulder, 7=rElbow, 8=rWrist, 9=lHip, 10=lKnee, 11=lAnkle,
#   12=rHip, 13=rKnee, 14=rAnkle, 15=lEye, 16=lEar, 17=rEye, 18=rEar
# COCO-17:
#   0=Nose, 1=LEye, 2=REye, 3=LEar, 4=REar, 5=LShoulder, 6=RShoulder,
#   7=LElbow, 8=RElbow, 9=LWrist, 10=RWrist, 11=LHip, 12=RHip,
#   13=LKnee, 14=RKnee, 15=LAnkle, 16=RAnkle
COCO_TO_PAN = [
    1,   # COCO 0 (Nose)       -> Pan 1 (Nose)
    15,  # COCO 1 (LEye)       -> Pan 15 (lEye)
    17,  # COCO 2 (REye)       -> Pan 17 (rEye)
    16,  # COCO 3 (LEar)       -> Pan 16 (lEar)
    18,  # COCO 4 (REar)       -> Pan 18 (rEar)
    3,   # COCO 5 (LShoulder)  -> Pan 3 (lShoulder)
    9,   # COCO 6 (RShoulder)  -> Pan 9 (rShoulder)
    4,   # COCO 7 (LElbow)     -> Pan 4 (lElbow)
    10,  # COCO 8 (RElbow)     -> Pan 10 (rElbow)
    5,   # COCO 9 (LWrist)     -> Pan 5 (lWrist)
    11,  # COCO 10 (RWrist)    -> Pan 11 (rWrist)
    6,   # COCO 11 (LHip)      -> Pan 6 (lHip)
    12,  # COCO 12 (RHip)      -> Pan 12 (rHip)
    7,   # COCO 13 (LKnee)     -> Pan 7 (lKnee)
    13,  # COCO 14 (RKnee)     -> Pan 13 (rKnee)
    8,   # COCO 15 (LAnkle)    -> Pan 8 (lAnkle)
    14,  # COCO 16 (RAnkle)    -> Pan 14 (rAnkle)
]

# COCO-17 bone connections
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (0, 5), (0, 6),
    (5, 7), (7, 9),                   # Left arm
    (6, 8), (8, 10),                  # Right arm
    (5, 11), (6, 12),                 # Torso
    (11, 13), (13, 15),               # Left leg
    (12, 14), (14, 16)                # Right leg
]

def plot_skeleton_3d(ax, kpts, color, label):
    valid = ~np.isnan(kpts[:, 0])
    xs = kpts[:, 0]
    ys = kpts[:, 2]  # OpenCV Z -> Matplotlib Y
    zs = -kpts[:, 1] # OpenCV Y -> Matplotlib Z (flipped so up is positive)
    ax.scatter(xs[valid], ys[valid], zs[valid], c=color, label=label, s=30)
    for p, c in BONES:
        if p < len(kpts) and c < len(kpts) and valid[p] and valid[c]:
            ax.plot([xs[p], xs[c]],
                    [ys[p], ys[c]],
                    [zs[p], zs[c]], c=color, linewidth=1.5)

def main():
    print(f"Stage 5 Corrected Diagnosis for {SEQ}")
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")

    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    gt_frames = {}
    for fi in range(500):
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                # Reorder from Panoptic order to COCO-17 order using CORRECT mapping
                gt_frames[fi] = k[COCO_TO_PAN, :3]

    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"

    cams = list(LAGS.keys())
    K_list = [calib[cn].K.astype(np.float64) for cn in cams]
    extrinsics = []
    for cn in cams:
        R = calib[cn].R.astype(np.float64)
        t = calib[cn].t.astype(np.float64).reshape(3, 1)
        extrinsics.append((R, t))

    all_kpts2d = []
    all_scores = []
    valid_gt = []

    print("Running 2D inference on 200 frames (temporally synchronized)...")
    for fi in tqdm(range(200)):
        kpts = np.full((len(cams), 17, 2), np.nan)
        scores = np.zeros((len(cams), 17))

        # fi is the GT frame index.
        # Each camera's corresponding video frame is: fi - lag[camera]
        # (lag = GT_frame - video_frame, so video_frame = GT_frame - lag)
        for ci, cn in enumerate(cams):
            video_frame = fi - LAGS[cn]  # correct video frame for this GT instant
            if video_frame < 0:
                continue

            img_path = frames_dir / f"hd_{cn}" / f"{video_frame:08d}.jpg"
            if img_path.exists():
                fr = cv2.imread(str(img_path))
                p = model.estimate(fr, pick="largest")
                if p:
                    kpts[ci] = p[0].keypoints[:17]
                    scores[ci] = p[0].scores[:17]

        all_kpts2d.append(kpts)
        all_scores.append(scores)
        valid_gt.append(gt_frames.get(fi, np.full((17, 3), np.nan)))

    print("Triangulating with Gate 5 policy...")
    diag = triangulate_sequence_with_diagnostics(
        np.array(all_kpts2d),
        np.array(all_scores),
        K_list,
        extrinsics,
        min_conf=0.4,
        reproj_threshold_px=25.0
    )
    pts3d = diag.points3d  # Raw world coords, NO conversion needed — same space as GT

    # --- MPJPE ---
    errors_all, errors_body, errors_face = [], [], []
    face_joints = {0, 1, 2, 3, 4}  # nose, eyes, ears
    for f in range(200):
        gt = valid_gt[f]
        pred = pts3d[f]
        for j in range(17):
            if np.isfinite(gt[j, 0]) and np.isfinite(pred[j, 0]):
                err_mm = np.linalg.norm(gt[j] - pred[j]) * 10.0  # cm -> mm
                errors_all.append(err_mm)
                if j in face_joints:
                    errors_face.append(err_mm)
                else:
                    errors_body.append(err_mm)

    print("\n=== Results ===")
    if errors_all:
        print(f"All joints  — MPJPE: {np.mean(errors_all):.1f} mm  Median: {np.median(errors_all):.1f} mm  P95: {np.percentile(errors_all,95):.1f} mm")
    if errors_body:
        print(f"Body only   — MPJPE: {np.mean(errors_body):.1f} mm  Median: {np.median(errors_body):.1f} mm  P95: {np.percentile(errors_body,95):.1f} mm")
    if errors_face:
        print(f"Face only   — MPJPE: {np.mean(errors_face):.1f} mm  Median: {np.median(errors_face):.1f} mm  P95: {np.percentile(errors_face,95):.1f} mm")

    # --- 3D Skeleton Plot ---
    f_viz = 150
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    gt_f = valid_gt[f_viz]
    pred_f = pts3d[f_viz]

    if not np.isnan(gt_f).all():
        plot_skeleton_3d(ax, gt_f, 'green', 'Ground Truth')
    if not np.isnan(pred_f).all():
        plot_skeleton_3d(ax, pred_f, 'red', 'Triangulated (Gate 5)')

    ax.set_title(f"3D Skeleton — Frame {f_viz}\nBody MPJPE: {np.mean(errors_body):.1f}mm")
    ax.legend()
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Z (depth, cm)"); ax.set_zlabel("Height (-Y, cm)")
    ax.set_xlim(-200, -100)
    ax.set_ylim(-50, 50)
    ax.set_zlim(0, 200)
    out = ROOT / "stage5_corrected_viz.png"
    plt.savefig(str(out), dpi=120, bbox_inches='tight')
    print(f"\nSaved visualization to {out}")

if __name__ == "__main__":
    main()
