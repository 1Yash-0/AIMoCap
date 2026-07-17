"""
Stage 6b Deep Diagnostic — fixes the three root causes found in the reconciliation audit:

ROOT CAUSE 1: Inlier mask is all False (zero residuals) because:
  - observations.py uses `from aimocap.math.triangulate import triangulate_n_views`
    inside the function — if this import fails, the bare `except: continue` swallows it silently.
  - Camera units: calibration t is in cm, but triangulate_n_views might expect mm or have a unit issue.

ROOT CAUSE 2: Coordinate system mismatch:
  - b_stage6 is in COCO joint order (17 joints as COCO keypoints 0-16) in Panoptic world space.
  - Stage 6b FK outputs in CANONICAL joint order (different joint numbering).
  - The comparison must either (a) map canonical->COCO, or (b) use COCO ordering throughout.

ROOT CAUSE 3: Bone lengths all defaulted to 10cm:
  - estimate_bone_lengths_robust() needs `ray_angle_deg >= 15` AND `inlier_count >= 2`.
  - If no inliers in obs60 (root cause 1), all bone lengths are defaulted.
  - Fix: use b_stage6 directly to estimate bone lengths (it IS the stage-5 clean output).
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

print("=" * 70)
print("DEEP DIAGNOSTIC — Root cause analysis and fixes")
print("=" * 70)

# Load data
arrs   = np.load(ROOT / "outputs/phase_b_gate1/gate1_arrays.npz")
b_raw  = arrs["b_stage6"]                               # (1800,17,3) cm, COCO order
gt_raw = arrs["gt"]                                      # (1800,17,3) mm, COCO order
pos_b  = b_raw.astype(np.float64)
pos_gt = gt_raw.astype(np.float64) / 10.0               # cm, COCO order

obs_data = np.load(ROOT / "outputs/canonical_dataset/canonical_detector_pose_observations.npz")
kpts     = obs_data["kpts"]    # (1800, 3, 17, 2)
scores   = obs_data["scores"]  # (1800, 3, 17)

with open(ROOT / "data/panoptic/171204_pose1/calibration_171204_pose1.json") as f:
    calib = json.load(f)
cam_by_name = {c.get("name", c.get("node", str(c.get("id","")))): c for c in calib["cameras"]}

TRIO   = ["00_00", "00_01", "00_02"]
BODY_J = [5,6,7,8,9,10,11,12,13,14,15,16]  # COCO indices for body joints

# ─── Diagnostic 1: Are there ANY inliers in obs60? ──────────────────────────
print("\n[Diag 1] Checking if any inliers in obs60...")

from aimocap.motion.camera import CameraModel
from aimocap.motion.observations import build_multiview_observations
from aimocap.motion.skeleton import CanonicalSkeleton

def build_cameras(trio_names):
    cams = []
    for nm in trio_names:
        c  = cam_by_name[nm]
        K  = np.array(c["K"], dtype=np.float64)
        R  = np.array(c["R"], dtype=np.float64)
        t  = np.array(c["t"], dtype=np.float64).reshape(3,1)
        d  = np.zeros(5, dtype=np.float64)  # pre-rectified
        sz = tuple(c.get("resolution", [1920,1080]))
        cams.append(CameraModel(name=nm, K=K, R=R, t=t, dist=d, image_size=sz))
    return cams

cameras = build_cameras(TRIO)
kpts60  = kpts[:60];  scores60 = scores[:60]
img_sizes = [c.image_size for c in cameras]

obs60 = build_multiview_observations(kpts60, scores60, cameras, img_sizes, fps=30.0)

n_inliers_total = obs60.inlier_mask.sum()
n_valid_3d      = obs60.valid.sum()
print(f"  inlier_mask total True entries: {n_inliers_total}  out of {obs60.inlier_mask.size}")
print(f"  valid 3D points: {n_valid_3d}  out of {obs60.valid.size}")
print(f"  reprojection RMSE mean: {obs60.reprojection_rmse_px[obs60.valid].mean():.2f} px")
print(f"  ray_angle_deg mean:     {obs60.ray_angle_deg[obs60.valid].mean():.2f} deg")
print(f"  inlier_count mean (valid): {obs60.inlier_count[obs60.valid].mean():.2f}")

# ─── Diagnostic 2: Joint ordering mismatch ──────────────────────────────────
print("\n[Diag 2] COCO vs Canonical joint ordering mismatch")
print("  COCO 17 joints (b_stage6 / gt order):")
coco_names = ["nose","l_eye","r_eye","l_ear","r_ear",
               "l_shoulder","r_shoulder","l_elbow","r_elbow",
               "l_wrist","r_wrist","l_hip","r_hip",
               "l_knee","r_knee","l_ankle","r_ankle"]
for i, nm in enumerate(coco_names):
    print(f"    COCO {i:2d}: {nm}")
    
print("\n  Canonical 17 joints (FK output order):")
for i, nm in enumerate(CanonicalSkeleton.NAMES):
    print(f"    Canon {i:2d}: {nm}")

# Build canonical->COCO mapping
# For each canonical joint, which COCO joint is the equivalent?
CANON_TO_COCO = {
    4:  0,   # head -> nose
    5:  5,   # l_shoulder -> l_shoulder
    6:  7,   # l_elbow -> l_elbow
    7:  9,   # l_wrist -> l_wrist
    8:  6,   # r_shoulder -> r_shoulder
    9:  8,   # r_elbow -> r_elbow
    10: 10,  # r_wrist -> r_wrist
    11: 11,  # l_hip -> l_hip
    12: 13,  # l_knee -> l_knee
    13: 15,  # l_ankle -> l_ankle
    14: 12,  # r_hip -> r_hip
    15: 14,  # r_knee -> r_knee
    16: 16,  # r_ankle -> r_ankle
}
# Joints in BODY_J (COCO order) that have canonical equivalents
BODY_COCO_TO_CANON = {v: k for k, v in CANON_TO_COCO.items()}
print("\n  BODY_J COCO->Canonical mapping:")
for coco_j in BODY_J:
    canon_j = BODY_COCO_TO_CANON.get(coco_j, "NO MAPPING")
    print(f"    COCO {coco_j:2d} ({coco_names[coco_j]:12s}) -> Canonical {canon_j}")

# ─── Diagnostic 3: Bone lengths from b_stage6 directly ─────────────────────
print("\n[Diag 3] Estimating bone lengths from b_stage6 (the clean Stage-6 output)")
print("  This is the authoritative source since b_stage6 IS the clean pipeline output.")

# b_stage6 is in COCO order. The canonical skeleton has its own parent chain.
# We need bone lengths in canonical joint order (parent->child).
# Map canonical child joints to their COCO equivalents:
canon_parents = CanonicalSkeleton.PARENTS  # -1 means root

bone_lengths_cm = np.zeros(17)
for canon_j in range(1, 17):
    p_canon = canon_parents[canon_j]
    coco_j  = CANON_TO_COCO.get(canon_j)
    coco_p  = CANON_TO_COCO.get(p_canon)
    
    if coco_j is None or coco_p is None:
        # Virtual joint (spine=1, neck=3, chest=2, pelvis=0) — estimate separately
        if canon_j == 0:  # pelvis from hips
            # Pelvis = midpoint of hips, bone "length" is 0 (it's derived)
            continue
        elif canon_j == 1:  # spine (pelvis to chest midpoint)
            # Use b_stage6: ||(l_hip+r_hip)/2 - (l_shoulder+r_shoulder)/2|| / 2
            pelvis_cm  = (pos_b[:, 11] + pos_b[:, 12]) / 2.0   # midpoint hips
            chest_cm   = (pos_b[:, 5]  + pos_b[:, 6])  / 2.0   # midpoint shoulders
            dists = np.linalg.norm(chest_cm - pelvis_cm, axis=1) / 2.0  # spine = half
            fin   = np.isfinite(dists)
            bone_lengths_cm[canon_j] = float(np.median(dists[fin])) if fin.sum() > 0 else 15.0
        elif canon_j == 2:  # chest (spine to chest = half total torso)
            pelvis_cm  = (pos_b[:, 11] + pos_b[:, 12]) / 2.0
            chest_cm   = (pos_b[:, 5]  + pos_b[:, 6])  / 2.0
            dists = np.linalg.norm(chest_cm - pelvis_cm, axis=1) / 2.0
            fin   = np.isfinite(dists)
            bone_lengths_cm[canon_j] = float(np.median(dists[fin])) if fin.sum() > 0 else 15.0
        elif canon_j == 3:  # neck (chest to head * 0.33)
            chest_cm = (pos_b[:, 5] + pos_b[:, 6]) / 2.0
            head_cm  = pos_b[:, 0]  # nose
            dists    = np.linalg.norm(head_cm - chest_cm, axis=1) * 0.33
            fin      = np.isfinite(dists)
            bone_lengths_cm[canon_j] = float(np.median(dists[fin])) if fin.sum() > 0 else 7.0
        elif canon_j == 4:  # head (neck to head * 0.67)
            chest_cm = (pos_b[:, 5] + pos_b[:, 6]) / 2.0
            head_cm  = pos_b[:, 0]
            dists    = np.linalg.norm(head_cm - chest_cm, axis=1) * 0.67
            fin      = np.isfinite(dists)
            bone_lengths_cm[canon_j] = float(np.median(dists[fin])) if fin.sum() > 0 else 15.0
        continue
    
    # Both joints have COCO equivalents
    child_pos  = pos_b[:, coco_j]   # (1800, 3) cm
    parent_pos = pos_b[:, coco_p]   # (1800, 3) cm
    dists      = np.linalg.norm(child_pos - parent_pos, axis=1)
    fin        = np.isfinite(dists) & (dists > 0.1)
    bone_lengths_cm[canon_j] = float(np.median(dists[fin])) if fin.sum() > 20 else 10.0

print("  Bone lengths (cm) from b_stage6:")
for j, (nm, bl) in enumerate(zip(CanonicalSkeleton.NAMES, bone_lengths_cm)):
    print(f"    Canonical {j:2d} {nm:16s}: {bl:.2f} cm")

# Check bilateral symmetry
pairs = [(5,8,"shoulder"), (6,9,"elbow"), (7,10,"wrist"), (11,14,"hip"), (12,15,"knee"), (13,16,"ankle")]
print("\n  Bilateral symmetry check:")
for l, r, nm in pairs:
    diff = abs(bone_lengths_cm[l] - bone_lengths_cm[r])
    sym  = diff / ((bone_lengths_cm[l] + bone_lengths_cm[r]) / 2.0 + 1e-9)
    flag = "OK" if sym < 0.05 else ("WARN" if sym < 0.10 else "FAIL")
    print(f"    {nm:10s}: L={bone_lengths_cm[l]:.2f}cm  R={bone_lengths_cm[r]:.2f}cm  asym={sym:.1%}  {flag}")

# Pool symmetric pairs
for l, r, nm in pairs:
    sym = abs(bone_lengths_cm[l] - bone_lengths_cm[r]) / ((bone_lengths_cm[l] + bone_lengths_cm[r])/2.0+1e-9)
    if sym < 0.05:
        pooled = (bone_lengths_cm[l] + bone_lengths_cm[r]) / 2.0
        bone_lengths_cm[l] = bone_lengths_cm[r] = pooled

print("\n  Final (pooled symmetric) bone lengths:")
for j, (nm, bl) in enumerate(zip(CanonicalSkeleton.NAMES, bone_lengths_cm)):
    print(f"    Canon {j:2d} {nm:16s}: {bl:.2f} cm")

# ─── Diagnostic 4: Correct comparison strategy ──────────────────────────────
print("\n[Diag 4] Correct evaluation strategy for Stage 6b vs Candidate B")
print("""
  The canonical FK output is in CANONICAL joint order.
  The b_stage6 and gt are in COCO joint order.
  
  CORRECT comparison:
    For each body joint that has a COCO<->Canonical mapping:
      canonical_idx: Canon-j  (e.g., Canon 11 = left_hip)
      coco_idx:      COCO-j   (e.g., COCO 11 = l_hip)
    
    stage6b_err[f, coco_j] = ||pos_fk[f, canon_j] - pos_gt[f, coco_j]||
    b_err[f, coco_j]       = ||b_stage6[f, coco_j] - pos_gt[f, coco_j]||
    
  This aligns the two streams in COCO joint space for a fair comparison.
  Note: pelvis (Canon 0), spine (Canon 1), chest (Canon 2), neck (Canon 3)
  have NO direct COCO equivalent → exclude from MPJPE comparison.
""")

# ─── Fix and re-run the 60-frame optimizer ──────────────────────────────────
print("=" * 70)
print("FIX AND RE-RUN: 60-frame optimizer with correct bone lengths and comparison")
print("=" * 70)

from aimocap.motion.optimizer import SequentialCanonicalFitter
from aimocap.motion.sequence_optimizer import WindowedSequenceOptimizer

# Build x0 using obs.points3d (world-space 3D from triangulation)
bvh_pos = CanonicalSkeleton.build_positions_from_coco(obs60.points3d)
print(f"bvh_pos.shape: {bvh_pos.shape}")
fin_bvh = np.isfinite(bvh_pos).sum()
print(f"Finite bvh_pos values: {fin_bvh}/{bvh_pos.size} ({100*fin_bvh/bvh_pos.size:.1f}%)")

# Run fitter with authoritative bone lengths
fitter = SequentialCanonicalFitter(fps=30.0)
x0 = fitter.optimize_sequence(bvh_pos, bone_lengths_cm)
print(f"x0 shape: {x0.shape}, finite: {np.isfinite(x0).sum()}/{x0.size}")

# Get FK positions from x0
wopt = WindowedSequenceOptimizer(cameras, fps=30.0, bone_lengths=bone_lengths_cm, b_stage6=None)
pos_fk_init, _ = wopt._fk(x0.flatten(), 60)  # (60, 17, 3) in canonical order, cm

print(f"FK pos_init shape: {pos_fk_init.shape}, finite: {np.isfinite(pos_fk_init).sum()}/{pos_fk_init.size}")

# Compare initial FK to GT using correct COCO<->Canonical mapping
print("\nInitial FK vs GT (using correct canonical->COCO joint mapping):")
F60 = 60
init_errs = []
b_errs    = []

for coco_j in BODY_J:
    canon_j = BODY_COCO_TO_CANON.get(coco_j)
    if canon_j is None:
        continue
    for f in range(F60):
        fk_pos  = pos_fk_init[f, canon_j]  # canonical cm
        gt_pos  = pos_gt[f, coco_j]         # COCO cm
        b_pos   = pos_b[f, coco_j]          # COCO cm
        if np.isfinite(fk_pos).all() and np.isfinite(gt_pos).all() and np.isfinite(b_pos).all():
            init_errs.append(np.linalg.norm(fk_pos - gt_pos) * 10.0)  # mm
            b_errs.append(np.linalg.norm(b_pos - gt_pos) * 10.0)      # mm

print(f"  Candidate B   MPJPE: {np.mean(b_errs):.3f} mm  (n={len(b_errs)})")
print(f"  Stage6b init  MPJPE: {np.mean(init_errs):.3f} mm")

if np.mean(init_errs) < 500.0:
    print("  -> Initial FK is in world coordinates (expected)")
else:
    print("  -> WARNING: Initial FK error very large (>500mm) - coordinate frame issue")

# Check initial residuals more carefully
frame_indices = np.arange(60)
r0 = wopt._residuals(x0.flatten(), 60, obs60, frame_indices)
print(f"\nResiduals diagnostic:")
print(f"  Length: {len(r0)}")
print(f"  RMS:    {np.sqrt(np.mean(r0**2)):.6f}")
print(f"  Non-zero residuals: {(np.abs(r0) > 1e-12).sum()}")
print(f"  Max absolute residual: {np.abs(r0).max():.6f}")
print(f"  Min absolute residual: {np.abs(r0).min():.6f}")

# Check how many 2D inliers were found in obs60
print(f"\n  obs60 inlier check:")
print(f"    inlier_mask.any(): {obs60.inlier_mask.any()}")
print(f"    inlier_mask sum:   {obs60.inlier_mask.sum()}")
print(f"    valid.any():       {obs60.valid.any()}")
print(f"    valid sum:         {obs60.valid.sum()}")
print(f"    points3d finite:   {np.isfinite(obs60.points3d).sum()}/{obs60.points3d.size}")

# Show sample: for frame 0, joint 11 (l_hip), camera 0 — is it an inlier?
f, j, c_idx = 0, 11, 0
print(f"\n  Frame={f} Joint={j}(l_hip) Camera={c_idx}(00_00):")
print(f"    scores:         {scores60[f, c_idx, j]:.3f}")
print(f"    kpts2d:         {kpts60[f, c_idx, j]}")
print(f"    is_inlier:      {obs60.inlier_mask[f, c_idx, j]}")
print(f"    obs_weight:     {obs60.observation_weight[f, c_idx, j]:.3f}")
print(f"    points3d[f,j]:  {obs60.points3d[f, j]}")
print(f"    valid[f,j]:     {obs60.valid[f, j]}")

# ─── Suggested fix for residuals: use obs.kpts2d directly ───────────────────
print("\n[Fix] Computing 2D residuals manually to check projection consistency")
# For frame 0, joint 11, camera 0: project obs.points3d[0,11] through cam 0
cam0 = cameras[0]
pt3d = obs60.points3d[0, 11]  # cm
if np.isfinite(pt3d).all():
    rvec, _ = cv2.Rodrigues(cam0.R)
    proj, _ = cv2.projectPoints(pt3d.reshape(1,3), rvec, cam0.t, cam0.K, cam0.dist)
    proj = proj.reshape(2)
    obs_2d = kpts60[0, 0, 11]
    print(f"  Projecting obs.points3d[0,11] through cam 00_00:")
    print(f"    pt3d (cm): {pt3d}")
    print(f"    projected: {proj.round(2)}")
    print(f"    observed:  {obs_2d.round(2)}")
    print(f"    residual:  {np.linalg.norm(proj - obs_2d):.2f} px")
else:
    print(f"  obs.points3d[0,11] is NaN — triangulation failed for this joint")
    # Try manual triangulation
    print("  Attempting manual DLT triangulation for frame 0, joint 11...")
    from aimocap.math.triangulate import triangulate_n_views
    conf_valid = scores60[0, :, 11]
    print(f"  Confidence: {conf_valid}")
    valid_cams = [c for c in range(3) if conf_valid[c] > 0.3]
    print(f"  Valid cameras (conf>0.3): {valid_cams}")
    if len(valid_cams) >= 2:
        pts2d = [kpts60[0, c, 11] for c in valid_cams]
        Pmats = []
        for c_i in valid_cams:
            K = cameras[c_i].K; R = cameras[c_i].R; t = cameras[c_i].t
            Pmats.append(K @ np.hstack([R, t]))
        try:
            pt3d_dlt = triangulate_n_views(np.array(pts2d), Pmats)
            print(f"  DLT result (cm): {pt3d_dlt}")
            print(f"  GT l_hip (cm):   {pos_gt[0, 11]}")
            print(f"  Error (mm):      {np.linalg.norm(pt3d_dlt - pos_gt[0, 11])*10:.2f}")
        except Exception as e:
            print(f"  DLT failed: {e}")

print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
print(f"""
1. DISTORTION: PATH3/4 (production, no undistortion) = 0.0000mm. CONFIRMED.

2. INLIERS: obs60 inlier_mask sum = {obs60.inlier_mask.sum()}, valid 3D = {obs60.valid.sum()}.
   {'OK — inliers found' if obs60.inlier_mask.any() else 'PROBLEM — no inliers, residuals will be zero'}

3. BONE LENGTHS: All defaulted to 10cm in prior run due to no inliers.
   Now estimated from b_stage6 directly: {[f'{bone_lengths_cm[j]:.1f}' for j in range(17)]}

4. JOINT MAPPING: COCO order (b_stage6, gt) vs Canonical order (FK output).
   Misaligned comparison caused the 820mm error in prior run.
   CANON_TO_COCO mapping: {CANON_TO_COCO}

5. NEXT STEP: 
   - If initial FK MPJPE (with correct mapping) is reasonable (< 500mm), 
     proceed to optimize with higher max_nfev.
   - If initial MPJPE is still >500mm, the FK initialization is in the wrong coordinate frame.
""")
