"""Stage 6a.5 — Pose-Model and Crop Benchmark Under Occlusion"""

import sys
from pathlib import Path
import json
import itertools
import numpy as np
import cv2
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aimocap  # Ensures CUDA DLLs load
from aimocap.pose.infer import PoseEstimator
from aimocap.pose.infer_rtmpose import RTMPoseEstimator
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.math.triangulate import triangulate_weighted_dlt, triangulate_robust

# ── Paths & Config ────────────────────────────────────────────────────────
CALIB_JSON  = ROOT / "data" / "panoptic" / "171204_pose1" / "calibration_171204_pose1.json"
GT_DIR      = ROOT / "data" / "panoptic" / "171204_pose1" / "hdPose3d_stage1_coco19"
VIDEO_DIR   = ROOT / "data" / "panoptic" / "171204_pose1" / "hdVideos"
OUT_DIR     = ROOT / "outputs" / "stage6a_5_pose_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAM_NAMES_4 = ["00_26", "00_29", "00_30", "00_01"]
N_FRAMES    = 300
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
CRITICAL_JOINTS = [("l_knee", 13), ("r_knee", 14), ("l_ankle", 15), ("r_ankle", 16), ("l_wrist", 9), ("r_wrist", 10), ("l_hip", 11), ("r_hip", 12)]

# ── Loaders ────────────────────────────────────────────────────────
def load_gt():
    gt_files = sorted(GT_DIR.glob("body3DScene_*.json"))
    if not gt_files: return np.zeros((0, 17, 3))
    max_idx = int(gt_files[-1].stem.split("_")[-1])
    kpts = np.full((max_idx + 1, 19, 4), np.nan)
    for fpath in gt_files:
        fi = int(fpath.stem.split("_")[-1])
        with open(fpath) as fp:
            d = json.load(fp)
        if not d.get("bodies"): continue
        kpts[fi] = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
    return kpts[:, COCO17_TO_PAN19, :3] # Panoptic is in cm!

def load_calib():
    calib = load_panoptic_calib(CALIB_JSON)
    cams = []
    for cn in CAM_NAMES_4:
        c = calib[cn]
        P = c.K.astype(np.float64) @ np.hstack([c.R.astype(np.float64), c.t.reshape(3, 1).astype(np.float64)])
        cams.append({"name": cn, "K": c.K.astype(np.float64), "R": c.R.astype(np.float64), 
                     "t": c.t.reshape(3, 1).astype(np.float64), "distCoef": c.dist_coef, "P": P})
    return cams

# ── Padding Logic ────────────────────────────────────────────────────────
def pad_bbox(bbox, pad_frac, w, h):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * pad_frac / 2.0, bh * pad_frac / 2.0
    return [max(0, x1-dx), max(0, y1-dy), min(w, x2+dx), min(h, y2+dy)]

class PaddedPoseEstimator(PoseEstimator):
    def __init__(self, pad_frac=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pad_frac = pad_frac
    def _detect(self, frame):
        bboxes = super()._detect(frame)
        if self.pad_frac > 0 and bboxes:
            h, w = frame.shape[:2]
            bboxes = [pad_bbox(b, self.pad_frac, w, h) for b in bboxes]
        return bboxes

class PaddedRTMPoseEstimator(RTMPoseEstimator):
    def __init__(self, pad_frac=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pad_frac = pad_frac
    def estimate(self, frame, pick="largest"):
        h, w = frame.shape[:2]
        bboxes = self._wholebody.det_model(frame)
        if self.pad_frac > 0 and len(bboxes) > 0:
            bboxes = np.array([pad_bbox(b, self.pad_frac, w, h) for b in bboxes])
        kpts, scores = self._wholebody.pose_model(frame, bboxes=bboxes)
        if kpts is None or len(kpts) == 0: return []
        from aimocap.pose.infer import Pose2D
        poses = []
        for i in range(len(kpts)):
            b = self._bbox_from_kpts(kpts[i])
            poses.append(Pose2D(kpts[i].astype(np.float32), scores[i].astype(np.float32), b))
        return [max(poses, key=lambda p: (p.bbox[2]-p.bbox[0])*(p.bbox[3]-p.bbox[1]))] if pick=="largest" and poses else poses

# ── Triangulation ────────────────────────────────────────────────────────
def ransac_triangulate(k_sub, P_sub, s_sub):
    n_v = len(P_sub)
    if n_v < 2: return np.full(3, np.nan)
    if n_v == 2: return triangulate_weighted_dlt(k_sub, P_sub, s_sub)
    
    best_err, best_pt = np.inf, None
    for pair in itertools.combinations(range(n_v), 2):
        pt = triangulate_weighted_dlt(k_sub[list(pair)], P_sub[list(pair)], s_sub[list(pair)])
        pts3d_h = np.append(pt, 1.0)
        proj = (P_sub @ pts3d_h)
        proj = proj[:, :2] / proj[:, 2:3]
        errs = np.linalg.norm(proj - k_sub, axis=1)
        med_err = np.median(errs)
        if med_err < best_err:
            best_err, best_pt = med_err, pt
    return best_pt

def calc_rc_mpjpe(pred, gt):
    rp = (pred[11] + pred[12]) / 2.0
    rg = (gt[11] + gt[12]) / 2.0
    diff = (pred - rp) - (gt - rg)
    return np.nanmean(np.linalg.norm(diff, axis=1) * 10.0)

# ── 0. Discrepancy & Regression ────────────────────────────────────────────────────────
def test_regression(gt, cams):
    print("="*80 + "\\n0. Discrepancy Resolution & Regression Test\\n" + "="*80)
    print("Discrepancy: The 112.1mm vs 98.3mm left-knee errors in Stage 6a.4 were 'Huber Robust' vs 'Weighted DLT'.")
    print("Both were computed on held-out frames 100-299.\\n")
    
    errs = []
    for fi in range(N_FRAMES):
        if fi + 149 >= len(gt) or np.isnan(gt[fi + 149, 15]).any(): continue
        gt_pt = gt[fi + 149, 15]
        
        pts2d = []
        Ps = []
        for c in cams:
            rvec, _ = cv2.Rodrigues(c["R"])
            proj_d, _ = cv2.projectPoints(np.array([gt_pt]), rvec, c["t"], c["K"], c["distCoef"])
            corr = cv2.undistortPoints(proj_d, c["K"], c["distCoef"], P=c["K"])
            pts2d.append(corr[0, 0])
            Ps.append(c["P"])
        pt3d = triangulate_weighted_dlt(np.array(pts2d), np.array(Ps), np.ones(4))
        errs.append(np.linalg.norm(pt3d - gt_pt) * 10.0)
        
    med = np.median(errs)
    print(f"Regression Test (Distorted GT -> Undistort -> DLT) Median Error: {med:.4f} mm")
    assert med < 1.0, f"Regression test failed! Median error {med} >= 1mm"
    
    # Clear Pair Upper Bound
    errs_clear = []
    c_idxs = [2, 3] # 00_30 and 00_01
    for fi in range(100, N_FRAMES):
        if fi + 149 >= len(gt) or np.isnan(gt[fi + 149, 13]).any(): continue
        gt_pt = gt[fi + 149, 13]
        pts2d = []
        Ps = []
        for i in c_idxs:
            c = cams[i]
            rvec, _ = cv2.Rodrigues(c["R"])
            proj_d, _ = cv2.projectPoints(np.array([gt_pt]), rvec, c["t"], c["K"], c["distCoef"])
            corr = cv2.undistortPoints(proj_d, c["K"], c["distCoef"], P=c["K"])
            pts2d.append(corr[0, 0])
            Ps.append(c["P"])
        pt3d = triangulate_weighted_dlt(np.array(pts2d), np.array(Ps), np.ones(2))
        errs_clear.append(np.linalg.norm(pt3d - gt_pt) * 10.0)
    print(f"Clear Pair (00_30 + 00_01) Left Knee Oracle Median: {np.median(errs_clear):.2f} mm")

# ── Main Run ────────────────────────────────────────────────────────
def main():
    gt = load_gt()
    cams = load_calib()
    test_regression(gt, cams)
    
    print("\\n" + "="*80 + "\\n1. 2D Inference & Evaluation\\n" + "="*80)
    models = {
        "CIGPose_YOLO": PaddedPoseEstimator(pad_frac=0.0),
        "CIGPose_Pad15": PaddedPoseEstimator(pad_frac=0.15),
        "RTMPose_YOLO": PaddedRTMPoseEstimator(pad_frac=0.0),
        "RTMPose_Pad15": PaddedRTMPoseEstimator(pad_frac=0.15),
    }
    
    # Store predictions: dict[model][cam][frame] = Pose2D
    preds = {m: {c["name"]: [None]*N_FRAMES for c in cams} for m in models}
    
    for c_idx, c in enumerate(cams):
        vid_path = VIDEO_DIR / f"hd_{c['name']}.mp4"
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened(): raise RuntimeError(f"Could not open {vid_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 149)
        print(f"Running inference on {c['name']}...", end="", flush=True)
        t0 = time.time()
        for fi in range(N_FRAMES):
            ret, frame = cap.read()
            if not ret: break
            for m_name, model in models.items():
                p = model.estimate(frame, pick="largest")
                if p: preds[m_name][c["name"]][fi] = p[0]
        print(f" done in {time.time()-t0:.1f}s.")
        cap.release()
        
    print("\\n2D Evaluation against GT (Diagnostic set: 0-99):")
    for m_name in models:
        print(f"\\n--- {m_name} ---")
        for c in cams:
            for j_name, jid in [("l_knee", 13), ("l_ankle", 15)]:
                errs_px = []
                confs = []
                for fi in range(100):
                    f = 149 + fi
                    if f >= len(gt) or np.isnan(gt[f, jid]).any(): continue
                    p = preds[m_name][c["name"]][fi]
                    if not p or p.scores[jid] < 0.5: continue # Only evaluate confident predictions
                    
                    gt_pt = gt[f, jid]
                    rvec, _ = cv2.Rodrigues(c["R"])
                    proj, _ = cv2.projectPoints(np.array([gt_pt]), rvec, c["t"], c["K"], c["distCoef"])
                    gt_2d = proj[0, 0]
                    
                    err = np.linalg.norm(p.keypoints[jid] - gt_2d)
                    errs_px.append(err)
                    confs.append(p.scores[jid])
                    
                if errs_px:
                    errs_px = np.array(errs_px)
                    med = np.median(errs_px)
                    p95 = np.percentile(errs_px, 95)
                    gt40 = np.mean(errs_px > 40) * 100
                    print(f"  {c['name']} {j_name}: n={len(errs_px):2d} | Med={med:5.1f}px | p95={p95:5.1f}px | >40px={gt40:5.1f}% | Conf={np.mean(confs):.2f}")
                else:
                    print(f"  {c['name']} {j_name}: n= 0")

    print("\\n" + "="*80 + "\\n2. 3D Benchmark (Held-out 100-299)\\n" + "="*80)
    methods = ["DLT", "Robust", "RANSAC"]
    
    print(f"{'Model':<15} | {'Tri Method':<8} | {'l_knee (m/p95)':<15} | {'l_ankle (m/p95)':<15} | {'RC-MPJPE':<8}")
    for m_name in models:
        for method in methods:
            errs_lk = []
            errs_la = []
            rcs = []
            
            for fi in range(100, N_FRAMES):
                f = 149 + fi
                if f >= len(gt) or np.isnan(gt[f, 13]).any(): continue
                
                pts3d_pred = np.full((17, 3), np.nan)
                for jid in range(17):
                    gate = 0.35 if jid in [11, 12, 13, 14, 15, 16] else 0.5
                    kpts2d = []
                    Ps = []
                    scores = []
                    for c in cams:
                        p = preds[m_name][c["name"]][fi]
                        if p and p.scores[jid] > gate:
                            # Undistort
                            p_d = p.keypoints[jid]
                            corr = cv2.undistortPoints(np.array([[p_d]], dtype=np.float32), c["K"], c["distCoef"], P=c["K"])
                            kpts2d.append(corr[0, 0])
                            Ps.append(c["P"])
                            scores.append(p.scores[jid])
                    
                    if len(Ps) >= 2:
                        k2 = np.array(kpts2d)
                        P2 = np.array(Ps)
                        s2 = np.array(scores)
                        if method == "DLT":
                            pts3d_pred[jid] = triangulate_weighted_dlt(k2, P2, s2)
                        elif method == "Robust":
                            pts3d_pred[jid] = triangulate_robust(k2, P2, s2)
                        elif method == "RANSAC":
                            pts3d_pred[jid] = ransac_triangulate(k2, P2, s2)
                            
                # Metrics
                if not np.isnan(pts3d_pred[13]).any() and not np.isnan(gt[f, 13]).any():
                    errs_lk.append(np.linalg.norm(pts3d_pred[13] - gt[f, 13]) * 10.0)
                if not np.isnan(pts3d_pred[15]).any() and not np.isnan(gt[f, 15]).any():
                    errs_la.append(np.linalg.norm(pts3d_pred[15] - gt[f, 15]) * 10.0)
                if not np.isnan(pts3d_pred).all():
                    rcs.append(calc_rc_mpjpe(pts3d_pred, gt[f]))
                    
            m_lk = f"{np.median(errs_lk):.1f}/{np.percentile(errs_lk,95):.1f}" if errs_lk else "N/A"
            m_la = f"{np.median(errs_la):.1f}/{np.percentile(errs_la,95):.1f}" if errs_la else "N/A"
            rc = f"{np.nanmean(rcs):.1f}" if rcs else "N/A"
            print(f"{m_name:<15} | {method:<8} | {m_lk:<15} | {m_la:<15} | {rc:<8}")

if __name__ == "__main__":
    main()
