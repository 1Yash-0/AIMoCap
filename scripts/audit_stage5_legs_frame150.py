"""Audit legs for Frame 150 to check for rendering artifacts vs actual swapped labels."""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

SEQ = "171204_pose3"
FRAME = 150
LAGS = {"00_03": 0, "00_04": 0, "00_28": 18, "00_24": -1}
cams = list(LAGS.keys())

COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def get_camera_center(R, t): return -R.T @ t

def get_subject_forward(gt_kpts):
    if np.isnan(gt_kpts[0,0]) or np.isnan(gt_kpts[5,0]) or np.isnan(gt_kpts[6,0]): return None
    neck = (gt_kpts[5] + gt_kpts[6]) / 2.0
    fwd = gt_kpts[0] - neck
    fwd[1] = 0
    n = np.linalg.norm(fwd)
    if n < 1e-5: return None
    return fwd / n

def main():
    model = PoseEstimator()
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    K_list = []
    extrinsics = []
    cam_centers = {}
    for cn in cams:
        K_list.append(calib[cn].K.astype(np.float64))
        R = calib[cn].R.astype(np.float64)
        t = calib[cn].t.astype(np.float64).reshape(3, 1)
        extrinsics.append((R, t))
        cam_centers[cn] = get_camera_center(R, t).flatten()

    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    fpath = gt_dir / f"body3DScene_{FRAME:08d}.json"
    with open(fpath) as fp: d = json.load(fp)
    gt = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)[COCO_TO_PAN, :3]
    
    fwd = get_subject_forward(gt)
    pelvis = (gt[11] + gt[12]) / 2.0
    
    print(f"--- FRAME {FRAME} TRUE SUBJECT ORIENTATION ---")
    for cn in cams:
        v_cam = cam_centers[cn] - pelvis
        v_cam[1] = 0
        v_cam = v_cam / np.linalg.norm(v_cam)
        dot = np.dot(fwd, v_cam)
        orient = "FRONT" if dot > 0 else "BACK"
        print(f"Cam {cn:5s}: dot = {dot:5.2f} ({orient})")
        
    print("\n--- RAW 2D DETECTIONS (Before any swap) ---")
    all_kpts2d = []
    all_scores = []
    for cn in cams:
        video_frame = FRAME - LAGS[cn]
        ip = frames_dir / f"hd_{cn}" / f"{video_frame:08d}.jpg"
        kpt = np.full((17, 2), np.nan)
        sc = np.zeros(17)
        if ip.exists():
            fr = cv2.imread(str(ip))
            p = model.estimate(fr, pick="largest")
            if p:
                kpt, sc = p[0].keypoints[:17], p[0].scores[:17]
        
        print(f"\nCamera {cn}:")
        print(f"  Hip   L: {kpt[11]} | R: {kpt[12]} (Score: {sc[11]:.2f} / {sc[12]:.2f})")
        print(f"  Knee  L: {kpt[13]} | R: {kpt[14]} (Score: {sc[13]:.2f} / {sc[14]:.2f})")
        print(f"  Ankle L: {kpt[15]} | R: {kpt[16]} (Score: {sc[15]:.2f} / {sc[16]:.2f})")
        all_kpts2d.append(kpt)
        all_scores.append(sc)

    # Triangulate completely raw (no swaps)
    all_kpts2d = np.array([all_kpts2d])
    all_scores = np.array([all_scores])
    diag = triangulate_sequence_with_diagnostics(
        all_kpts2d, all_scores, K_list, extrinsics, min_conf=0.4, reproj_threshold_px=25.0
    )
    pred3d = diag.points3d[0]
    
    print("\n--- TRIANGULATED 3D JOINTS (Raw, No Swap) ---")
    print(f"  Knee  L Valid? {not np.isnan(pred3d[13, 0])}")
    print(f"  Knee  R Valid? {not np.isnan(pred3d[14, 0])}")
    print(f"  Knee  L: {pred3d[13]}")
    print(f"  Knee  R: {pred3d[14]}")
    print(f"  Ankle L: {pred3d[15]}")
    print(f"  Ankle R: {pred3d[16]}")
    
    # Let's see if full swap crosses them
    print("\n--- TRIANGULATED 3D JOINTS (Full Swap on 00_03) ---")
    all_kpts2d_full = all_kpts2d.copy()
    all_scores_full = all_scores.copy()
    LR_PAIRS = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    for l, r in LR_PAIRS:
        all_kpts2d_full[0, 0, l], all_kpts2d_full[0, 0, r] = all_kpts2d[0, 0, r].copy(), all_kpts2d[0, 0, l].copy()
        all_scores_full[0, 0, l], all_scores_full[0, 0, r] = all_scores[0, 0, r], all_scores[0, 0, l]
        
    diag_f = triangulate_sequence_with_diagnostics(
        all_kpts2d_full, all_scores_full, K_list, extrinsics, min_conf=0.4, reproj_threshold_px=25.0
    )
    pred3d_f = diag_f.points3d[0]
    print(f"  Knee  L Valid? {not np.isnan(pred3d_f[13, 0])}")
    print(f"  Knee  R Valid? {not np.isnan(pred3d_f[14, 0])}")
    print(f"  Knee  L: {pred3d_f[13]}")
    print(f"  Knee  R: {pred3d_f[14]}")
    print(f"  Ankle L: {pred3d_f[15]}")
    print(f"  Ankle R: {pred3d_f[16]}")

if __name__ == "__main__":
    main()
