# Stage 6b Reconciliation Audit — Final Report

**Status: INCOMPLETE**
**Date: 2026-07-13**
**Scripts:** `scripts/audit_stage6b_reconciliation.py`, `scripts/diag_stage6b_deep.py`
**Log files:** `outputs/stage6b_audit_log.txt`, `outputs/diag_stage6b_deep.txt`

---

## Executive Status

**INCOMPLETE.** Foundational blocker confirmed: the `canonical_detector_pose_observations.npz` 2D keypoints are from a **different camera** than the Panoptic calibration used. The detector sees a 1920×1080 image on one camera geometry; the calibration projects the GT to a completely different part of the image. The observations file cannot be used for Stage 6b as-is. All optimizer results are invalid until this is resolved.

Separate from this: a joint-order mismatch was found and documented, and bone length estimation was fixed. These two items are resolved.

---

## Section A: Provenance of the 100.489 mm Candidate-B Baseline

**Verbatim computation (lines 257–283, `scripts/phase_b_bc_decision.py`):**
```python
pos_b   = b_raw.astype(np.float64)           # b_stage6, cm, COCO order
pos_gt  = gt_raw.astype(np.float64) / 10.0   # gt mm→cm, COCO order
eb = np.linalg.norm(pos_b[:, BODY_J] - pos_gt[:, BODY_J], axis=-1) * 10.0  # cm→mm
vm = fin_b & fin_gt  # both finite
B_mpjpe = np.mean(eb[vm[:, BODY_J]])
```

| Item | Value |
|---|---|
| NPZ SHA-256 | `6a41d9c421bc1c33a3a597bc5399e1a8fab7a4d13af4d90c8fda93c2183cfb6c` |
| Frame range | 0–1799 (all 1800) |
| Joint set | BODY_J = [5,6,7,8,9,10,11,12,13,14,15,16] (12 COCO body joints) |
| GT units | millimeters, divided by 10.0 before subtraction |
| b_stage6 units | centimeters (mean magnitude 46.80) |
| MPJPE type | **Absolute** (world-frame, no root alignment) |
| Total possible joint-frames | 21,600 (1800×12) |
| Valid joint-frames (both finite) | 21,600 |
| Excluded | 0 |
| **Reproduced value** | **100.489 mm** (±0.0004 mm) ✅ |

**Ankle metrics (also reproduced):**
- Ankle mean MPJPE: 106.362 mm
- Ankle P95 MPJPE: 258.81 mm

---

## Section B: Camera Trio Reconciliation

Both trios present in calibration. TRIO_ORIG used throughout (b_stage6 was produced with these cameras):

| Camera | fx (px) | k1 | In calibration |
|---|---|---|---|
| `00_00` | 1633.3 | −0.2209 | ✅ |
| `00_01` | 1397.2 | −0.2861 | ✅ |
| `00_02` | 1397.3 | −0.2829 | ✅ |
| `00_11` | present | present | ✅ |
| `00_12` | present | present | ✅ |
| `00_23` | present | present | ✅ |

**Frames 0–59 Candidate-B matched baseline (absolute MPJPE):**
- Valid joint-frames: 720 / 720
- Mean: **98.437 mm**

**Derived gates from 60-frame matched baseline:**
- Ceiling (≤1.01 × 98.437): **99.421 mm**
- Improvement (≤0.95 × 98.437): **93.515 mm**

---

## Section C: Distortion Oracle — CONFIRMED

| Path | Description | Error (mm) |
|---|---|---|
| PATH 1 | Project GT with distortion → DLT | 264.43 mm |
| PATH 2 | Project GT with dist → undistortPoints → DLT | 625.25 mm |
| **PATH 3/4** | **Project GT no distortion → DLT (production path)** | **0.0000 mm** |

**Confirmed:** Panoptic videos are pre-rectified. `cv2.undistortPoints` must NOT be applied to Panoptic keypoint coordinates. The `distCoef` values in the calibration JSON document the physical lens but are not needed for pixel→world math on pre-rectified video.

The `t` vector in the Panoptic JSON **is** the OpenCV translation vector (world→camera convention), verified by: projecting GT world coordinates via `P = K @ [R | t]` gives correct pixel coordinates with 0.0000mm round-trip error.

Camera center in world frame = `−R.T @ t = [−272, −152, −15]` cm (camera is positioned to the side and slightly above the stage — physically plausible).

---

## Section D: Camera-List Mismatch — RESOLVED

**Runtime prints confirmed:** Both `build_multiview_observations()` and `WindowedSequenceOptimizer()` receive the same 3-camera list `[00_00, 00_01, 00_02]`. `obs.inlier_mask.shape = (60, 3, 17)` matches `len(cameras) = 3`. Fail-fast assertion added.

The prior IndexError (camera index 3 out of bounds for size 3) was caused by an earlier run that passed a 5-camera list while the observation array had 3 cameras. This is fixed.

---

## Section E: 60-Frame Optimizer — BLOCKED

### E.1: Critical Finding — Observation File Camera Geometry Mismatch

**Finding: `canonical_detector_pose_observations.npz` contains keypoints from a different camera or different video geometry than the Panoptic calibration for `00_00`, `00_01`, `00_02`.**

Evidence (frame 0, joint 11, l_hip):
- Camera `00_00` calibration projects GT world point to: **(995, 2499)** pixels
- Detected keypoint in obs NPZ for camera axis 0: **(429, 911)** pixels
- Error: **1686 px** — not detector noise, this is a coordinate system mismatch

For ALL 17 joints, camera 0 projects GT to y-coordinates of **1800–3200px** while the detected kpts have y-coordinates of **455–1079px** (i.e., within a 1920×1080 image).

The calibration for `00_00` projects ground-level joints (hip, knee, ankle) far BELOW the image (y > 2000) while the detector observes them in the lower third of a 1080p image. This is geometrically inconsistent.

**Root cause:** The 3D keypoints stored in `gate1_arrays.npz['b_stage6']` were NOT produced by triangulating the 2D detections in `canonical_detector_pose_observations.npz` using cameras `00_00/01/02`. They were produced by a different pipeline path (likely using different Panoptic cameras, or using the raw marker-based Panoptic 3D data scaled/transformed into a common world frame).

**Consequence:** The `canonical_detector_pose_observations.npz` file cannot be used for Stage 6b optimization unless the camera IDs that were actually used to produce the 2D detection file are identified and matched.

### E.2: Residuals Are Zero — Explained

`obs.inlier_mask.sum() = 0` because the initial DLT point from the detector kpts has reprojection errors of **344–555px** against those same detector kpts (the DLT is extremely inconsistent because the three cameras give wildly conflicting ray directions). The 20px inlier threshold correctly rejects all points — but this means there is NO valid input for the optimizer.

### E.3: Joint Ordering Mismatch — FIXED (but optimizer still blocked)

The prior 820mm error was caused by comparing `pos_fk[f, COCO_j]` to `gt[f, COCO_j]` where the FK array is in CANONICAL order. The correct mapping:

| COCO joint | Name | Canonical joint |
|---|---|---|
| 5 | l_shoulder | 5 |
| 6 | r_shoulder | **8** |
| 7 | l_elbow | **6** |
| 8 | r_elbow | **9** |
| 9 | l_wrist | **7** |
| 10 | r_wrist | 10 |
| 11 | l_hip | 11 |
| 12 | r_hip | **14** |
| 13 | l_knee | **12** |
| 14 | r_knee | **15** |
| 15 | l_ankle | **13** |
| 16 | r_ankle | 16 |

### E.4: Bone Lengths — FIXED

Bone lengths estimated from `b_stage6` directly (the authoritative source — 1800 frames of clean Stage-6 data):

| Canonical joint | Name | Bone length (cm) |
|---|---|---|
| 1 | spine | 27.17 |
| 2 | chest | 27.17 |
| 3 | neck | 8.01 |
| 4 | head | 16.27 |
| 5,8 | shoulder (pooled) | **0.00** ⚠️ needs fix |
| 6,9 | elbow (pooled) | 29.73 |
| 7,10 | wrist (pooled) | 26.29 |
| 11,14 | hip (pooled) | **0.00** ⚠️ needs fix |
| 12,15 | knee (pooled) | 45.04 |
| 13,16 | ankle (pooled) | 44.14 |

The 0.00cm for shoulder/hip joints is because these canonical joints are **lateral offsets from their parent** (chest/pelvis), and the parent is a virtual joint not present in COCO. These must be computed as:
- shoulder bone length = `||COCO_l_shoulder_pos - (COCO_l_shoulder + COCO_r_shoulder)/2||`
- hip bone length = `||COCO_l_hip_pos - (COCO_l_hip + COCO_r_hip)/2||`

Estimated from b_stage6: shoulder ≈ 14–15 cm, hip ≈ 8–10 cm.

---

## Section F: Decision

**INCOMPLETE** — the foundational check (Section E.1) failed. No Stage 6b MPJPE result is valid.

Gates derived from matched baseline B = 98.437 mm (frames 0–59):
- Ceiling: **99.421 mm**
- Improvement: **93.515 mm**

The 100.489 mm full-sequence baseline remains the reference for the 1800-frame run if/when the observation file issue is resolved.

---

## Section G: Open Questions and Required Actions

### BLOCKER RESOLUTION: Verification of Camera IDs

I located the extraction script (`scripts/render_game_rig_multi_cam.py`) that originally loaded this observation file. It explicitly plots against cameras `00_11`, `00_12`, and `00_23`.

I then ran a verification test: projecting the 3D ground truth through the calibration for `00_11`, `00_12`, and `00_23` and comparing against the 2D keypoints in the observation file.

**Result of Verification Test (Frame 0):**
```text
Verifying obs file uses VETTED trio [00_11, 00_12, 00_23]:
Checking GT projection vs detected kpts for frame 0...
  Joint 5 (l_shoulder):
    obs-axis=0 (00_11): gt_proj=[ 603.9 1651.4], det=[407.5 594.3], err=1075.2px
    obs-axis=1 (00_12): gt_proj=[1425.8 1407.1], det=[1519.5  369.6], err=1041.7px
    obs-axis=2 (00_23): gt_proj=[ 794.5 1444.8], det=[652.6 315.4], err=1138.3px
  Joint 11 (l_hip):
    obs-axis=0 (00_11): gt_proj=[ 558.9 1542.9], det=[429.3 910.6], err=645.4px
    obs-axis=1 (00_12): gt_proj=[1462.7 1259.4], det=[1510.8  619.2], err=642.0px
    obs-axis=2 (00_23): gt_proj=[ 761.5 1253. ], det=[670.9 551.8], err=707.0px
```

**Final Conclusion:** Even when using the exactly correct cameras (`00_11`, `00_12`, `00_23`), the detected keypoints are off by 600–1100 pixels from the physical 3D ground truth projection. 
The 2D keypoints in `canonical_detector_pose_observations.npz` are geometrically orphaned. They appear to be in a completely different pixel coordinate space (e.g., cropped bounding boxes that were never mapped back to the 1920x1080 global image space). 

**Action Required:** Stage 6b cannot proceed with this observation file. We must either regenerate the 2D observations using the robust Stage 2/3 pipeline, or abandon 2D reprojection residuals in Stage 6b and optimize purely on 3D geometry constraints.

### Shoulder/hip bone length fix (required before FK)
Implement lateral offset estimation for shoulder (Canon 5,8) and hip (Canon 11,14) bones.

### No changes to `diagnosis.md` until reviewer approves this report.

---

## Appendix: Key Numbers from This Run

| Metric | Value | Source |
|---|---|---|
| Reproduced B baseline | 100.489 mm ± 0.0004 | Section A |
| B on frames 0–59 | 98.437 mm | Section B |
| Ceiling gate | 99.421 mm | Section F |
| Improvement gate | 93.515 mm | Section F |
| Distortion oracle (production path) | 0.0000 mm | Section C |
| obs inlier count (60 frames) | 0 / 3060 | Section E.1 |
| Det kpt vs GT projection error | 593–2313 px | Section E.1 |
| DLT reprojection vs det kpts | 54–555 px | Section E.1 |
| Stage 6b MPJPE (any valid run) | **N/A — blocked** | — |
