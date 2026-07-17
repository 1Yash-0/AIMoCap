import sys
import json
import hashlib
import numpy as np
import scipy.linalg
from pathlib import Path
from datetime import datetime

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
NPZ_PATH = ROOT / "outputs/phase_b_gate1/gate1_arrays.npz"
SPOT_CHECKS_PATH = ROOT / "outputs/phase_b_matcher_recovery/real_event_spot_checks.json"
OUT_JSON = ROOT / "outputs/phase_b_matcher_recovery/coord_alignment.json"

BODY_J = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
FPS = 30.0

def sha256_file(path):
    h = hashlib.sha256()
    if not path.exists(): return ""
    with open(path, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

def umeyama(X, Y):
    """
    Estimates similarity transform (s, R, t) that maps X to Y.
    Y = s * X * R^T + t
    Here X is GT, Y is B. Wait, "mapping GT -> B" means B approx s * R * GT + t.
    Actually standard is Y approx c * R @ X.T + t.
    Let's stick to row vectors: Y approx c * X @ R.T + t
    """
    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)
    X0 = X - mu_x
    Y0 = Y - mu_y
    
    var_x = np.mean(np.sum(X0**2, axis=1))
    cov = (Y0.T @ X0) / X.shape[0]
    
    U, S, Vt = np.linalg.svd(cov)
    
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.eye(3)
    D[2, 2] = d
    
    R = U @ D @ Vt
    c = 1.0 / var_x * np.trace(np.diag(S) @ D)
    t = mu_y - c * (R @ mu_x)
    
    return c, R, t

def main():
    with open(__file__, 'rb') as f: script_sha = hashlib.sha256(f.read()).hexdigest()
    
    prov = {
        "script_path": __file__,
        "script_sha256": script_sha,
        "npz_path": str(NPZ_PATH),
        "npz_sha256": sha256_file(NPZ_PATH),
        "timestamp": datetime.utcnow().isoformat(),
        "joint_list": BODY_J,
        "units": "cm",
    }
    
    arrs = np.load(NPZ_PATH, allow_pickle=True)
    pos_b = arrs["b_stage6"].astype(np.float64)
    pos_c = arrs["c_stage6"].astype(np.float64)
    pos_gt = arrs["gt"] / 10.0
    
    b_pts = []
    gt_pts = []
    for f in range(1800):
        for j in BODY_J:
            b = pos_b[f, j]
            g = pos_gt[f, j]
            if np.isfinite(b).all() and np.isfinite(g).all():
                b_pts.append(b)
                gt_pts.append(g)
    
    b_pts = np.array(b_pts)
    gt_pts = np.array(gt_pts)
    
    # 1. POSITION-LEVEL AXIS CORRELATION
    print("=== 1. POSITION-LEVEL AXIS CORRELATION ===")
    pearson_mat = np.zeros((3, 3))
    b_means = np.mean(b_pts, axis=0)
    b_stds = np.std(b_pts, axis=0)
    g_means = np.mean(gt_pts, axis=0)
    g_stds = np.std(gt_pts, axis=0)
    
    for i in range(3):
        for j in range(3):
            cov = np.mean((b_pts[:, i] - b_means[i]) * (gt_pts[:, j] - g_means[j]))
            pearson_mat[i, j] = cov / (b_stds[i] * g_stds[j] + 1e-9)
            
    print("Correlation Matrix (Rows: Bx, By, Bz. Cols: GTx, GTy, GTz):")
    print(np.round(pearson_mat, 3))
    
    mapping = []
    axis_names = ['x', 'y', 'z']
    for i in range(3):
        best_j = np.argmax(np.abs(pearson_mat[i]))
        sign = "+" if pearson_mat[i, best_j] > 0 else "-"
        mapping.append(f"B{axis_names[i]}={sign}GT{axis_names[best_j]}")
        print(f"B{axis_names[i]} best match: {sign}GT{axis_names[best_j]} (r={pearson_mat[i, best_j]:.3f})")
    
    inferred_mapping = ", ".join(mapping)
    print(f"Inferred Signed Axis Mapping: {inferred_mapping}")
    
    # 2. RIGID/SIMILARITY ALIGNMENT
    print("\n=== 2. RIGID/SIMILARITY ALIGNMENT ===")
    valid_full_frames = []
    for f in range(1800):
        if np.isfinite(pos_b[f, BODY_J]).all() and np.isfinite(pos_gt[f, BODY_J]).all():
            valid_full_frames.append(f)
            
    print(f"Frames with full finite body joints: {len(valid_full_frames)}")
    
    if len(valid_full_frames) > 0:
        stacked_b = pos_b[valid_full_frames][:, BODY_J, :].reshape(-1, 3)
        stacked_gt = pos_gt[valid_full_frames][:, BODY_J, :].reshape(-1, 3)
        
        scale, R, t = umeyama(stacked_gt, stacked_b)
        
        pred_b = scale * (stacked_gt @ R.T) + t
        rmse = np.sqrt(np.mean(np.sum((stacked_b - pred_b)**2, axis=1)))
        
        print("Rotation matrix R (GT -> B):")
        print(np.round(R, 3))
        print(f"Scale s: {scale:.4f}")
        print(f"Translation t magnitude: {np.linalg.norm(t):.4f} cm")
        print(f"Residual RMSE: {rmse:.4f} cm")
        
        is_perm = np.all(np.abs(np.abs(R) - 1.0) < 0.15) or np.all(np.max(np.abs(R), axis=1) > 0.85)
        print(f"R approximately signed axis-permutation? {is_perm}")
        print(f"s approximately 1.0? {abs(scale - 1.0) < 0.1}")
    else:
        scale, R, t, rmse, is_perm = 1.0, np.eye(3), np.zeros(3), 999.0, False
        print("Not enough frames for Umeyama.")
        
    # 3. EIGHT-FLIP DIRECTION SWEEP
    print("\n=== 3. EIGHT-FLIP DIRECTION SWEEP ===")
    flips = [
        (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
        (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)
    ]
    
    with open(SPOT_CHECKS_PATH) as f:
        spot_checks = json.load(f)
        
    print("Flip -> Median Direction Cosine:")
    flip_results = {}
    best_flip = None
    best_med = -1.0
    
    for flip in flips:
        cosines = []
        for s in spot_checks:
            j = s["b_event"]["joint"]
            b_s = s["b_event"]["start"]
            b_e = s["b_event"]["end"]
            g_s = s["gt_event"]["start"]
            g_e = s["gt_event"]["end"]
            
            # Recompute GT accel
            vel_g = (pos_gt[1:, j] - pos_gt[:-1, j]) * FPS
            acc_g = (vel_g[1:] - vel_g[:-1]) * FPS
            
            b_vec = np.array(s["b_event"]["peak_vector_mps2"])
            
            # Apply flip to pos_gt effectively flips accel_gt
            flip_arr = np.array(flip)
            g_peak_frame = s["gt_event"]["peak_frame"]
            if g_peak_frame < len(acc_g):
                g_vec = acc_g[g_peak_frame] * flip_arr
                nx = np.linalg.norm(b_vec)
                ny = np.linalg.norm(g_vec)
                c = np.dot(b_vec, g_vec) / (nx * ny) if nx > 0 and ny > 0 else 0
                cosines.append(c)
                
        med = float(np.median(cosines))
        flip_str = f"x{'+' if flip[0]>0 else '-'} y{'+' if flip[1]>0 else '-'} z{'+' if flip[2]>0 else '-'}"
        print(f"{flip_str}: {med:.4f} {'<--- OVER 0.5' if med > 0.5 else ''}")
        flip_results[flip_str] = med
        if med > best_med:
            best_med = med
            best_flip = flip_str
            
    # 4. INTERPRETATION GATE
    print("\n=== 4. INTERPRETATION GATE ===")
    if is_perm and rmse < 10.0 and best_med > 0.5:
        print("LIKELY COORDINATE MISMATCH")
        print(f"Corrective Transform: R=\n{np.round(R, 2)}")
        print("Evidence: phase_b_gate1_audit.py applied gt_mm[:,:,1] *= -1; gt_mm[:,:,2] *= -1; gt_mm *= 10.0 which inverted Y and Z axes.")
    elif rmse >= 10.0 and best_med <= 0.5:
        print("LIKELY UNRELATED TIME-COINCIDENT EVENTS")
    else:
        print("MIXED EVIDENCE: Check R matrix and cosines.")
        
    # 5. FIVE FAILING-PAIR DUMPS
    print("\n=== 5. FIVE FAILING-PAIR DUMPS ===")
    fails = [s for s in spot_checks if not s["eligibility"]["5"]][:5]
    for idx, s in enumerate(fails):
        j = s["b_event"]["joint"]
        bp = s["b_event"]["peak_frame"]
        gp = s["gt_event"]["peak_frame"]
        bv = np.array(s["b_event"]["peak_vector_mps2"])
        gv = np.array(s["gt_event"]["peak_vector_mps2"])
        
        bpos = pos_b[bp+1, j] if bp+1 < 1800 else np.zeros(3)
        gpos = pos_gt[gp+1, j] if gp+1 < 1800 else np.zeros(3)
        
        print(f"Pair {idx+1}: Joint {j}")
        print(f"  B Peak Frame: {bp}, GT Peak Frame: {gp}")
        print(f"  B Accel Vec: [{bv[0]:.2f}, {bv[1]:.2f}, {bv[2]:.2f}], GT Accel Vec: [{gv[0]:.2f}, {gv[1]:.2f}, {gv[2]:.2f}]")
        print(f"  B Raw Pos: [{bpos[0]:.2f}, {bpos[1]:.2f}, {bpos[2]:.2f}], GT Raw Pos: [{gpos[0]:.2f}, {gpos[1]:.2f}, {gpos[2]:.2f}]")
        
    out = {
        "provenance": prov,
        "axis_correlation": pearson_mat.tolist(),
        "inferred_mapping": inferred_mapping,
        "umeyama": {
            "scale": scale,
            "R": R.tolist(),
            "t": t.tolist(),
            "rmse_cm": rmse,
            "is_permutation": bool(is_perm)
        },
        "eight_flip_cosines": flip_results,
        "failing_pairs": fails
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
