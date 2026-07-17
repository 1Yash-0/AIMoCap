import sys, json, cv2, csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.pose.infer import PoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics

SEQ = "171204_pose1"
CAMS = ["00_11", "00_12", "00_23"]
FRAMES_NUM = 1800
OFFSET = 150

COCO_TO_PAN = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def _project(pts3d, K, R, t):
    if np.isnan(pts3d).all(): return np.full((17, 2), np.nan)
    proj, _ = cv2.projectPoints(pts3d.astype(np.float64), R.astype(np.float64), t.astype(np.float64), K.astype(np.float64), None)
    return proj.reshape(-1, 2)

def write_csv(data_list, out_path):
    if not data_list: return
    keys = data_list[0].keys()
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_list)

def main():
    out_dir = ROOT / "outputs" / "diag_pose1_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    print("Loading Ground Truth...")
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    gt_frames = {}
    for fi in range(OFFSET, OFFSET + FRAMES_NUM):
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies") and len(d["bodies"]) > 0:
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                gt_frames[fi - OFFSET] = k[COCO_TO_PAN, :3] 

    model = PoseEstimator()
    vid_dir = ROOT / f"data/panoptic/{SEQ}/hdVideos"
    
    K_list = [calib[cn].K.astype(np.float64) for cn in CAMS]
    extrinsics = []
    centers = []
    for cn in CAMS:
        R = calib[cn].R.astype(np.float64)
        t = calib[cn].t.astype(np.float64).reshape(3, 1)
        extrinsics.append((R, t))
        centers.append((-R.T @ t).flatten())
        
    all_kpts2d = np.full((FRAMES_NUM, len(CAMS), 17, 2), np.nan, dtype=np.float32)
    all_scores = np.zeros((FRAMES_NUM, len(CAMS), 17), dtype=np.float32)
    
    print("Running 2D inference on video streams...")
    for ci, cn in enumerate(CAMS):
        print(f"  Camera {cn}")
        cap = cv2.VideoCapture(str(vid_dir / f"hd_{cn}.mp4"))
        for fi in tqdm(range(FRAMES_NUM)):
            ret, frame = cap.read()
            if not ret: break
            p = model.estimate(frame, pick="largest")
            if p:
                all_kpts2d[fi, ci] = p[0].keypoints[:17]
                all_scores[fi, ci] = p[0].scores[:17]
        cap.release()
        
    print("Triangulating Sequence...")
    diag = triangulate_sequence_with_diagnostics(
        all_kpts2d, all_scores, K_list, extrinsics,
        min_conf=0.4, reproj_threshold_px=25.0, min_aspect_ratio=1.8
    )
    
    print("Building structured logs...")
    log_a, log_b, log_c, log_d, log_e = [], [], [], [], []
    
    for f in tqdm(range(FRAMES_NUM)):
        global_frame = f + OFFSET
        gt = gt_frames.get(f, np.full((17, 3), np.nan))
        pts3d_raw = diag.points3d[f]
        
        gt_cv = gt
        
        n_cams_passing_ar = 0
        cams_passing = []
        for c in range(len(CAMS)):
            cn = CAMS[c]
            conf = all_scores[f, c]
            valid_mask = conf >= 0.4
            j_above = valid_mask.sum()
            det_conf = float(np.mean(conf[valid_mask])) if j_above > 0 else 0.0
            
            w, h, ar, passed = 0.0, 0.0, 0.0, False
            if j_above > 2:
                pts = all_kpts2d[f, c][valid_mask]
                w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
                h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
                if w > 0: ar = h / w
                passed = ar >= 1.8
                if passed: 
                    n_cams_passing_ar += 1
                    cams_passing.append(c)
                    
            log_a.append({
                'frame': global_frame, 'camera': cn, 'det_conf': det_conf,
                'bbox_h': h, 'bbox_w': w, 'bbox_ar': ar, 'kp_extent_ar': ar,
                'passed_ar_gate': passed, 'joints_above_min_conf': j_above
            })
            
            R, t = extrinsics[c]
            K = K_list[c]
            gt_proj = _project(gt_cv, K, R, t)
            
            for j in range(17):
                is_valid_det = conf[j] >= 0.4
                px = all_kpts2d[f, c, j]
                g_px = gt_proj[j]
                px_err = float(np.linalg.norm(px - g_px)) if is_valid_det and not np.isnan(g_px[0]) else None
                
                out_bbox = False
                if is_valid_det:
                    out_bbox = (px[0] < 20 or px[0] > 1900 or px[1] < 20 or px[1] > 1060)
                
                used_inlier = bool(diag.inlier_mask[f, j, c])
                
                log_b.append({
                    'frame': global_frame, 'camera': cn, 'joint': j,
                    'px_x': float(px[0]) if is_valid_det else None,
                    'px_y': float(px[1]) if is_valid_det else None,
                    'conf': float(conf[j]),
                    'gt_proj_x': float(g_px[0]) if not np.isnan(g_px[0]) else None, 
                    'gt_proj_y': float(g_px[1]) if not np.isnan(g_px[1]) else None,
                    'px_error': px_err,
                    'outside_bbox': out_bbox,
                    'used_as_inlier': used_inlier
                })
                
        max_baseline = 0.0
        for c1, c2 in combinations(range(len(CAMS)), 2):
            if c1 in cams_passing and c2 in cams_passing:
                v1 = centers[c1]; v2 = centers[c2]
                cos_t = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                ang = float(np.degrees(np.arccos(np.clip(cos_t, -1, 1))))
                
                n_contrib = sum(1 for j in range(17) if diag.inlier_mask[f, j, c1] and diag.inlier_mask[f, j, c2])
                log_c.append({
                    'frame': global_frame, 'cam_a': CAMS[c1], 'cam_b': CAMS[c2],
                    'baseline_angle_deg': ang, 'n_joints_contributed': n_contrib
                })
                max_baseline = max(max_baseline, ang)
            else:
                log_c.append({
                    'frame': global_frame, 'cam_a': CAMS[c1], 'cam_b': CAMS[c2],
                    'baseline_angle_deg': 0.0, 'n_joints_contributed': 0
                })
                
        body_joints = [j for j in range(17) if j not in {0,1,2,3,4}]
        body_errs = []
        gt_yup = np.copy(gt_cv)
        if not np.isnan(gt_yup).all():
            gt_yup[:, 1] = -gt_yup[:, 1] 
            
        for j in range(17):
            pred = pts3d_raw[j]
            g3d = gt_yup[j]
            inls = int(diag.num_inliers[f, j])
            
            err = float(np.linalg.norm(pred - g3d)*10.0) if not np.isnan(pred[0]) and not np.isnan(g3d[0]) else None
            if j in body_joints and err is not None: body_errs.append(err)
            
            log_d.append({
                'frame': global_frame, 'joint': j,
                'raw_x': float(pred[0]) if not np.isnan(pred[0]) else None, 
                'raw_y': float(pred[1]) if not np.isnan(pred[1]) else None, 
                'raw_z': float(pred[2]) if not np.isnan(pred[2]) else None,
                'inlier_count': inls,
                'min_baseline_deg': None, 'max_baseline_deg': max_baseline, 
                'bone_lengths_*': None, 'gate6_reject': False,
                'gt_x': float(g3d[0]) if not np.isnan(g3d[0]) else None, 
                'gt_y': float(g3d[1]) if not np.isnan(g3d[1]) else None, 
                'gt_z': float(g3d[2]) if not np.isnan(g3d[2]) else None,
                'abs_error_mm': err,
                'clean_x': float(pred[0]) if not np.isnan(pred[0]) else None, 
                'clean_y': float(pred[1]) if not np.isnan(pred[1]) else None, 
                'clean_z': float(pred[2]) if not np.isnan(pred[2]) else None,
                'gap_filled': False,
                'clean_error_mm': err
            })
            
        root = pts3d_raw[11] if not np.isnan(pts3d_raw[11][0]) else np.full(3, np.nan)
        dist = float(np.sqrt(root[0]**2 + root[2]**2)) if not np.isnan(root[0]) else None
        mpjpe = float(np.mean(body_errs)) if body_errs else None
        
        log_e.append({
            'frame': global_frame,
            'root_x': float(root[0]) if not np.isnan(root[0]) else None, 
            'root_y': float(root[1]) if not np.isnan(root[1]) else None, 
            'root_z': float(root[2]) if not np.isnan(root[2]) else None,
            'dist_from_center_cm': dist,
            'subject_orientation_x': None, 'subject_orientation_z': None,
            'motion_vel_cm_frame': None,
            'root_offset_raw_cm': None, 'root_offset_clean_cm': None,
            'mpjpe_body_raw_mm': mpjpe, 'mpjpe_body_clean_mm': mpjpe,
            'low_quality_flag': bool(diag.low_quality_mask[f]),
            'max_baseline_angle_deg': max_baseline,
            'n_cameras_passing_ar': n_cams_passing_ar
        })
        
    print("Writing CSVs...")
    write_csv(log_a, out_dir / "camera_frame_log.csv")
    write_csv(log_b, out_dir / "keypoint_2d_log.csv")
    write_csv(log_c, out_dir / "camera_pair_geometry_log.csv")
    write_csv(log_d, out_dir / "triangulation_3d_log.csv")
    write_csv(log_e, out_dir / "frame_summary_log.csv")
    print("Done!")

if __name__ == '__main__':
    main()
