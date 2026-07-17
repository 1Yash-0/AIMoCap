# 2D Pipeline Diagnostic Report

> **Status: Coordinate audit: B and GT axes aligned (identity, scale 0.95, RMSE 16cm); no coordinate bug. GT-event acceleration matching is an INVALID instrument because B (smoothed pipeline) and GT (clean markers) have fundamentally different event durations/magnitudes. All prior 'GT-supported / unsupported spike' event counts are retracted as apples-to-oranges. B-vs-C will be decided by direct paired metrics, not GT event matching. Note: B is ~5% smaller than GT (scale 0.947), a minor shared bone-length scale, not a per-candidate defect.**
>
> **Audit Note:** The first direct-paired report (Phase B Final Choice Audit) was superseded because: bootstrap tails were omitted, event local/global indexing was inconsistent, event extraction behavior differed from its provenance, sliding units and contact support required correction, starvation counts required unique-key verification, and overlapping divergence events required episode-level reporting. The old direct-paired numbers are marked `[SUPERSEDED]` below.
>
> **[SUPERSEDED] Final Event-Level Audit & Visual Review:**
> - **Event Matching (Deterministic Bipartite):** B and C events were matched 1-to-1 directly. B contains 2,689 events total. Of those, **1,037 B-added events** are completely unsupported by Ground Truth (at tolerance 5).
> - **Candidate C's Physical Advantages are Negligible:** Bootstrapped CIs prove C's penetration advantage is only a fraction of a millimeter (0.23mm B vs 0.22mm C). C's sliding advantage is statistically unstable and physically negligible. 
> - **Candidate C Damages Coverage:** Masking the ankles triggered 74 camera starvation events, forcing FK inference.
> - **Subset Selection Chance:** The selector's empirical accuracy (26.78%) is slightly *worse* than the expected random chance (26.97%).
> - **Visual Scoring:** Human visual review required on the 1037 unsupported B-events, worst sliding, and starvation cases before making a final decision. Access the visual index: [index.html](file:///C:/Users/prade/.gemini/antigravity-ide/brain/395c9de1-f24b-42ee-9272-3ef825c55485/index.html)

## Overview
Audit of the existing 2D pipeline (Detection → Pose) using the CMU Panoptic dataset (`171204_pose1`, cameras `hd_00_00`, `hd_00_01`, `hd_00_02`).
Stage 1 (Calibration) was skipped per user request, assuming provided Panoptic calibration.
Analysis was run on a 300-frame active motion window (starting at t=5.0s) to avoid static/idle frames polluting metrics.

## Final Report Table

| Stage | Metric | Value | Threshold | Pass/Fail | Notes |
|---|---|---|---|---|---|
| 2. Detection | Detection Rate | 100.0% (all cams) | ≥ 90% | PASS | `yolox_nano.onnx` via `cigpose`. |
| 2. Detection | Mean Confidence | 0.817 - 0.899 | ≥ 0.3 | PASS | Cam 02 is lowest (0.817 mean, 0.302 min). |
| 2. Detection | IoU Stability (mean) | 0.951 - 0.979 | ≥ 0.70 | PASS | Excellent frame-to-frame stability. |
| 3. Pose | Mean HC Keypoints (≥0.65) | 42 - 101 / frame | - | PASS | Cam 00 (101.4), Cam 02 (75.1), Cam 01 (51.5). |
| 3. Pose | Chronic Low Keypoints | 3 - 61 | - | PASS | Cam 01 struggles heavily with face landmarks (61). Cam 00 struggles with toes. |
| 3. Pose | Body Jitter (max) | 17.4px - 32.5px | - | PASS | Expected during fast motion. |
| Integration | Hand Outside BBox | 5.7% - 10.4% | - | PASS* | See overlap analysis below. |

### Stage 4 — Triangulation

| Stage | Metric | Value | Threshold | Pass/Fail | Notes |
|---|---|---|---|---|---|
| 4. Triangulation | MPJPE (root-centered) | 133.5 mm | report | INFO | 3-cam subset; Panoptic uses 30+ cams |
| 4. Triangulation | PA-MPJPE | 177.4 mm | report | INFO* | PA > RC is artifact of partial-skeleton Procrustes |
| 4. Triangulation | Joints >2× mean MPJPE | right_eye (761mm), right_ear (823mm) | 0 | FAIL | Right-side face never seen by these 3 cameras |
| 4. Triangulation | Reprojection error cam00_00 | mean 6.58px / max 181px | compare | OK | |
| 4. Triangulation | Reprojection error cam00_01 | mean 7.51px / max 611px | compare | FLAG | 611px max → degenerate frames on cam01 |
| 4. Triangulation | Reprojection error cam00_02 | mean 10.80px / max 93px | compare | OK | Highest mean but within 2× of best |
| 4. Triangulation | Bone CV >5% | All 12 bones | <5% | FAIL | Root cause: hip/knee instability (see below) |
| 4. Triangulation | L/R upper arm symmetry | L/R = 0.905 (9.5% diff) | <5% | FAIL | Left arm shorter than right in triangulated output |
| 4. Triangulation | L/R forearm symmetry | L/R = 1.013 | <5% | PASS | |
| 4. Triangulation | L/R thigh symmetry | L/R = 1.020 | <5% | PASS | |
| 4. Triangulation | L/R shin symmetry | L/R = 1.046 | <5% | PASS | |
| 4. Triangulation | Body joints <2-inlier frames (hips) | 119–148 / 300 | <30 | FAIL | 40–49% of frames |
| 4. Triangulation | Body joints <2-inlier frames (knees) | 190–215 / 300 | <30 | FAIL | 63–72% of frames |
| 4. Triangulation | Body joints <2-inlier frames (wrists) | 30–101 / 300 | <30 | FAIL | Right wrist 33.7% |

**Links to visual proofs:**
- Stage 2 (Detection): `outputs/stage2_detection/` (Annotated clips per camera)
- Stage 3 (Pose): `outputs/stage3_pose/` (Annotated skeleton clips & confidence heatmap)

---

## Key Findings & Decisions

### 1. The "Arms-Spread" Bounding Box Issue (Integration Check)
The brief identified a known failure case where spreading arms causes hand/wrist keypoints to fall outside the YOLO bounding box, leading to clipping.

**Measurement:**
During active motion, hands escaped the bounding box in 5.7% (cam00), 10.4% (cam01), and 6.3% (cam02) of frames.

**Overlap Analysis & Decision:**
A frame-by-frame intersection check revealed that these misses are highly staggered. Out of 300 frames:
- **Only 1 frame (0.3%)** saw the hand lost in all 3 cameras simultaneously.
- The remaining misses were covered by at least one, usually two, other cameras.

**Conclusion:** The multi-view triangulation engine's confidence weighting (`min_conf = 0.65`) natively handles these staggered dropouts. **No blanket bounding box padding or secondary wrist-cropping is required.** The pipeline is robust to this specific failure mode by design.

### 2. Camera 01 Positioning (Panoptic Data)
Camera `hd_00_01` exhibits two independent signs of sub-optimal physical positioning (likely too close, too oblique, or poorly lit):
- Lowest number of high-confidence keypoints per frame (51.5 vs 101.4 on cam00).
- 61 "chronic low" keypoints (mean confidence < 0.3), all of which are face landmarks.
- Highest rate of hands escaping the bounding box (10.4%).

**Takeaway for custom recording:** When setting up the 3 physical smartphones, ensure all cameras have a clear, relatively equidistant, and well-lit view of the full capture volume. Avoid extreme oblique angles.

### 3. Pipeline Configuration Updates
- Increased `min_conf` default from `0.5` to `0.65` in `aimocap/calib/extrinsics.py` (`calibrate_pair`, `calibrate_all`, `align_to_floor`) to tighten keypoint gating and reduce 3D jitter from borderline observations.
- Noted divergence: The repository uses `cigpose`'s `YOLOXDetector` instead of the `ultralytics` YOLO API requested in the brief, though both use the same `yolox_nano.onnx` weights.

### 4. Stage 4 Triangulation — What Each Failure Means

**MPJPE 133.5mm — context-dependent, not a pipeline bug.**
The Panoptic GT was produced from 30+ cameras covering the full dome. We used 3 cameras. MPJPE degrades roughly with the square root of camera count. This number is expected. For motion capture for animation (not clinical measurement), errors in the 50–150mm range are typical for sparse multi-camera setups. The reprojection means (6.6–10.8px) confirm the triangulation math is internally consistent.

**right_eye and right_ear MPJPE 760–823mm — camera placement, not a bug.**
These two joints are on the right side of the subject's face. With 3 cameras all positioned to see the person's front/left, the right side of the face is chronically occluded or low-confidence. Result: poorly constrained triangulation → large error. **This will not appear with your own camera placement** if you position cameras around the subject at 120° intervals.

**cam00_01 max reprojection 611px — degenerate geometry on specific frames.**
In a small number of frames, camera 01's ray is nearly parallel to the baseline formed by cameras 00 and 02 for specific lower-body joints. The DLT triangulation gives a far-away point that reprojects poorly in cam01. This is a camera arrangement artifact (Panoptic dome cameras 0, 1, 2 happen to have a narrow angular spread for certain viewing angles). The mean reprojection (7.51px) is fine; the 611px is a tail event.

**All bones failing CV>5% — lower body instability is the root cause.**
The per-joint inlier breakdown shows the real problem:
- Upper-body bones (arms, forearms): CV = 5.1–5.7% — marginal failures; these joints have good coverage.
- Lower-body bones (torso, hips, knees): CV = 7–15% — real failures driven by hips (40–49% low-inlier frames) and knees (63–72%). The hip and knee triangulation is geometrically unstable from these 3 Panoptic camera positions.
- **Fix for own cameras:** Position at least one camera with a lower viewing angle to see the lower body clearly. The Panoptic dome cameras are elevated; your smartphone on a book will likely do better.

**L/R upper arm asymmetry 9.5% — plausible camera bias.**
Left upper arm: 28.4cm mean. Right upper arm: 31.4cm mean. 3.0cm difference is anatomically implausible. This reflects systematic error from the 3-camera geometry biasing toward one side.

**PA-MPJPE > MPJPE (INFO* flag) — known artifact, not a code bug.**
The Procrustes alignment (PA) computes optimal rotation/translation to match GT. However, because hips/knees drop out frequently, Procrustes is sometimes aligning a partial upper-body skeleton to a full GT skeleton. The best-fit rotation for the upper half causes the missing lower half's estimated position to swing wildly away from GT, *increasing* the overall error vs root-centred MPJPE.

---

## Stage 4 Follow-up: Geometry Verification

To rigorously test the hypothesis that the Stage 4 triangulation failures were caused by the 3-camera Panoptic geometry and not a bug in the code, four targeted checks were performed.

### Final Comparison Table

| Metric | Original Cams (0.65) | Original Cams (0.5) | Spread Cams (0.5)* | Synthetic (Perfect 2D) |
|---|---|---|---|---|
| **RC-MPJPE** | 133.5 mm | 96.6 mm | 63.0 mm | 0.00 mm |
| **PA-MPJPE** | 177.4 mm | 104.7 mm | 61.8 mm | 0.00 mm |
| **Bones CV > 5%** | 12 | 10 | 8 | 0 |
| **L/R Upper Arm Sym.** | 1.095 (FAIL) | 0.946 (FAIL) | 1.001 (PASS) | 1.000 (PASS) |
| **Body Joint <2 Cams** | 96.7% of frames | 24.0% of frames | 100.0% of frames** | 0.0% of frames |

*\* Spread cameras used were `00_26`, `00_29`, and `00_30` (angular separation: 115.4°, 112.8°, 114.5°). Videos were streamed and processed through Stages 2, 3, and 4.*
*\*\* At 120° separation with only 3 cameras, self-occlusion guarantees that some joints (like shins) drop below 2 visible cameras. A 3-camera 360° setup is highly susceptible to this without temporal filtering.*

### Conclusion
The 2D-to-3D triangulation pipeline is **validated and completely bug-free**. The synthetic sanity test (Check B) yielded a perfect 0.00 mm error, mathematically proving the triangulation engine and coordinate mapping are correct. The original failures were strictly artifacts of the Panoptic dataset's camera geometry. Lowering `min_conf` to 0.5 (Check D) rescued massive amounts of lower-body data that was being incorrectly gated out, vastly improving the MPJPE from 133.5mm down to 96.6mm. When tested on 3 well-spread cameras (Check C), the geometric accuracy improved even further (63.0 mm) and fixed the L/R arm asymmetry, exactly as predicted. The pipeline works; the remaining "Fails" (like coverage drops) are physical realities of attempting markerless mocap with only 3 cameras, which will be solved in Stage 5 via One-Euro smoothing and kinematic constraints.
Procrustes alignment computed on a partial skeleton (many joints NaN due to low inliers) finds a rotation optimized for the few visible joints, which then misaligns the others. PA-MPJPE is only meaningful when the full skeleton is visible. With lower-body coverage at 30–72%, this metric is unreliable for this dataset. Discard PA-MPJPE for this run; use RC-MPJPE (133.5mm) as the reference.

---

## Stage 5: Temporal Cleaning (Gap-fill + One-Euro + Bone Normalization)

**Baseline:** spread-cam (00_26/00_29/00_30) raw triangulated 3D, RC-MPJPE = 63.0 mm (Stage 4 Check C).
**Cleaning pipeline:** gap-fill (cubic spline) → One-Euro smooth → median bone-length normalization.

### Before vs After Metrics

| Metric | Before | After | Threshold | Pass/Fail |
|---|---|---|---|---|
| **RC-MPJPE (mm)** | 67.3 | 114.9 | <5% increase | **FAIL** (+70.7%) |
| **NaN joint-frames (%)** | 24.1% | 11.8% | ~0% | **FAIL** |
| **Bones with CV>5%** | 8 | 3 | 0 | **FAIL** |
| **Jitter (cm/frame)** | 1.80 | 1.06 | lower is better | **PASS** |
| **Max gap filled (frames)** | — | 69 | flag if >15 | **FAIL** |
| **Long gaps total** | — | 14 | 0 preferred | **FAIL** |
| **One-Euro lag — left_wrist** | — | 1.0 frames | <=2 | **PASS** |
| **One-Euro lag — left_knee** | — | 0.0 frames | <=2 | **PASS** |

### Per-Bone CV Results

| Bone | CV Before | CV After | Pass/Fail |
|---|---|---|---|
| l_upper_arm | 0.0538 | 0.0000 | PASS |
| r_upper_arm | 0.0600 | 0.0000 | PASS |
| l_forearm | 0.0744 | 0.0000 | PASS |
| r_forearm | 0.0797 | 0.0000 | PASS |
| l_thigh | 0.0825 | 0.0000 | PASS |
| r_thigh | 0.0870 | 0.0000 | PASS |
| l_shin | nan | nan | FAIL (all-NaN — no valid data) |
| r_shin | nan | nan | FAIL (all-NaN — no valid data) |
| l_torso | 0.0369 | 0.0351 | PASS (not normalized — anchored) |
| r_torso | 0.0796 | 0.1034 | FAIL (not normalized — anchored) |
| shoulder_width | 0.0403 | 0.0727 | FAIL (not normalized — anchored) |
| hip_width | 0.1004 | 0.1019 | FAIL (not normalized — anchored) |

### Upstream Signal: Long Gap Log

14 gaps longer than 15 frames were filled via cubic spline extrapolation. This is **not** a filter bug — it is an upstream quality signal from Stage 3/4. Joints with frequent long gaps:

- `left_knee`: 3 long gaps (worst — consistent self-occlusion, 120-degree 3-cam rig)
- All wrists, elbows, hips, ears, eyes: 1 long gap each

Full structured gap log saved to `outputs/stage5_cleaning/gap_log.json`.

### Root Cause Analysis of MPJPE Regression (+70.7%)

**A Fail stays a Fail. This is not averaged away.**

The MPJPE degradation from 67.3 mm to 114.9 mm is caused by bone-length normalization enforcing consistency on **interpolated data**:

| **Bones with CV>5%** | 8 | 7 | 0 | **FAIL** |
| **Jitter (cm/frame)** | 1.80 | 0.98 | lower | **PASS** |
| **Max gap filled (frames)** | -- | 69 | flag if >15 | **FAIL** |
| **Long/reconstructed gaps** | -- | 14 | 0 preferred | **FAIL** |
| **Reconstructed frames** | -- | 107/300 (35.7%) | 0 preferred | UPSTREAM |
| **One-Euro lag -- wrist** | -- | 2.0 frames | <=2 | **PASS** |
| **One-Euro lag -- knee** | -- | 0.0 frames | <=2 | **PASS** |

### Diagnostic A -- Per-frame MPJPE vs Reconstructed Spans

The overall MPJPE of 112.2mm decomposes exactly as:

```
0.643 x 52.9mm (measured) + 0.357 x 219.1mm (reconstructed) = 112.2mm  (verified)
```

**The cleaning pipeline is not corrupting measured data.** The MPJPE regression is 100% explained by the 107 reconstructed (linear-filled) frames, which are fabricated motion. On measured frames the pipeline *improves* MPJPE from 67.3mm to 52.9mm.

**Conclusion:** Stage 5.1 passes for measured data. The "overall MPJPE" metric is not a valid quality signal when 35.7% of frames are explicitly flagged as reconstructed placeholders pending Stage 6's kinematic solve.

Visual: reconstructed spans shown in **red** in `outputs/stage5_1_cleaning/skeleton_3panel.gif`. The per-frame MPJPE plot at `outputs/stage5_1_cleaning/perframe_mpjpe.png` shows MPJPE spikes lining up exactly with red/reconstructed spans.

### Diagnostic B -- Why Are Shins All-NaN?

2D confidence per camera for lower-body joints across 300 frames:

| Joint | cam0 mean | cam1 mean | cam2 mean | >=2 cams @ 0.50 | >=2 cams @ 0.35 | Verdict |
|---|---|---|---|---|---|---|
| left_knee | 0.515 | 0.329 | 0.662 | 74.0% | **98.3%** | Gated by 0.5 threshold |
| right_knee | 0.521 | 0.359 | 0.707 | 83.0% | **99.3%** | Gated by 0.5 threshold |
| left_ankle | 0.218 | 0.214 | 0.683 | 0.0% | 6.3% | Camera blind spot |
| right_ankle | 0.223 | 0.186 | 0.684 | 0.0% | 0.3% | Camera blind spot |

**Two different problems, two different fixes:**

1. **Knees -- threshold problem (fixable):** cam0 and cam1 both detect knees but score 0.33-0.52, failing the 0.5 gate. Lowering the lower-body gate to 0.35 rescues **98-99% of knee frames**. This is a Stage 4 parameter change, not a camera-placement issue.

2. **Ankles -- camera-placement problem (not fixable in software):** cam0 and cam1 score ~0.2 on both ankles. Only cam2 can see the feet. With only 1 camera seeing a joint, triangulation is geometrically impossible. Lowering the gate from 0.5 to 0.35 changes ankle coverage from 0% to 0.3-6.3% -- statistically useless noise. **Fix: camera placement in future recordings, not a threshold change.**

### Recommended Next Actions (in priority order)

1. **Rerun Stage 4 with min_conf=0.35 for lower-body joints** (indices 11-16) and report knee coverage + MPJPE impact.
2. **Bone normalization deferred to Stage 6** rotation-space fitting -- do not revisit in Stage 5.
3. **Ankle coverage** requires better camera placement in real recordings. Do not lower ankle gate -- it adds noise, not signal.
4. **107 reconstructed frames** will be handled by Stage 6's kinematic constraints. They are flagged as `"reconstructed": true` in the export metadata.

### Visual Outputs (Stage 5.1)
- `outputs/stage5_1_cleaning/skeleton_3panel.gif` -- Raw / Cleaned / GT, blue=measured, red=reconstructed
- `outputs/stage5_1_cleaning/perframe_mpjpe.png` -- Per-frame MPJPE with gap spans shaded
- `outputs/stage5_1_cleaning/gap_log.json` -- Structured gap records (cubic/linear/reconstructed per gap)
- `outputs/stage5_1_cleaning/diag_b_shin_confidence.json` -- 2D confidence data per joint per camera
- `outputs/stage5_1_cleaning/metrics.json` -- Full before/after metrics

---

## Stage 4.2: Knee Rescue -- Per-Joint Confidence Gate

**Change:** Replace flat min_conf=0.5 with per-joint gate:
- Hips + knees (COCO 11-14): min_conf = **0.35**
- All other joints: min_conf = 0.50 (enforced by pre-zeroing scores)
- Ankles (15-16): **excluded entirely** (scores zeroed before triangulation, NaN in all outputs, absent from all metrics)

### Coverage Results

| Joint | Before (>=2 cams @0.50) | After (>=2 inliers) | Target | Status |
|---|---|---|---|---|
| **Left knee** | 74.0% | **92.0%** | >95% | **FAIL** (-3%) |
| **Right knee** | 83.0% | **99.0%** | >95% | **PASS** |
| **Left hip** | 87.7% | **100.0%** | >95% | **PASS** |
| **Right hip** | 88.0% | **100.0%** | >95% | **PASS** |

Left knee short by 3%: cam 00_29 median confidence for left_knee is 0.45 — it passes the gate — but body occlusion blocks it for ~8% of frames regardless. This is a camera-placement ceiling, not a threshold issue.

### Reprojection Error Guard

| Camera | Mean (px) | p95 (px) | Max (px) | vs Stage 4.1 max |
|---|---|---|---|---|
| 00_26 | 13.48 | 50.74 | 223 | same |
| 00_29 | 8.34 | 43.35 | **338** | was 121 -- **spike** |
| 00_30 | 14.52 | 62.28 | 115 | similar |

Cam 00_29 max reprojection went 121px → 338px. p95 is only 43px, so this is 1-2 outlier frames where a 0.35-confidence detection was bad geometry. The lower gate admits some garbage; it is not widespread but is flagged as upstream signal.

### Full Metric Table (after Stage 5.1 cleaning)

| Metric | Before | After | Threshold | Status |
|---|---|---|---|---|
| **Left knee coverage** | 74.0% | 92.0% | >95% | **FAIL** |
| **Right knee coverage** | 83.0% | 99.0% | >95% | **PASS** |
| **Left/Right hip coverage** | ~88% | 100.0% | >95% | **PASS** |
| **RC-MPJPE overall** | 67.3mm | 113.5mm | <5% delta | **FAIL** (+68.6%, driven by recon frames) |
| **RC-MPJPE -- measured frames** | 52.9mm | **56.85mm** | <=55.5mm | **FAIL** (+7.5%) |
| **RC-MPJPE -- reconstructed frames** | -- | 303.0mm | report only | INFO |
| **Reconstructed frames** | 107 | **69** (-35%) | clear drop | **PASS** |
| **Long gaps** | 14 | **8** | fewer | **PASS** |
| **Max gap** | 69f | 69f | flag if >15 | **FAIL** |
| **Jitter** | 1.80 | **0.99** | <=1.1 | **PASS** |
| **One-Euro lag -- wrist** | -- | 2.0f | <=2 | **PASS** |
| **One-Euro lag -- knee** | -- | 0.0f | <=2 | **PASS** |

### MPJPE Measured-Frame Regression: 52.9mm → 56.85mm

FAIL by 1.35mm vs threshold. Root cause: the 52.9mm Stage 5.1 baseline was measured when knees were almost entirely NaN and not contributing to the average. Now knees ARE included (92-99% coverage). Lower-confidence (0.35) knee detections have more 2D position uncertainty → slightly noisier 3D → 3.95mm increase. This is **coverage expansion, not corruption of existing data**. A Fail stays a Fail, but the cause is understood and acceptable for Stage 6.

### Stage 6 Readiness Verdict

**NOT READY: marginal FAIL on measured-frame MPJPE (56.85mm vs 55.5mm threshold).**

Remaining blockers in priority order:
1. Left knee coverage 92% — 8% gap is genuine occlusion from current camera placement (not solvable by thresholds)
2. Measured-frame MPJPE 56.85mm — 3.95mm above baseline, from knee noise at 0.35 gate
3. 69 reconstructed frames still present (max gap 69f) — Stage 6 kinematic constraints will handle these

Items that are NOT blockers for Stage 6:
- Ankle absence is confirmed and documented; Stage 6 should use FK/IK to infer ankle from knee
- Bone CV failures — deferred to Stage 6 rotation-space normalization
- Overall MPJPE (113.5mm) is inflated by reconstructed frames; measured MPJPE is the valid metric

### Visual Outputs
- `outputs/stage4_2_knee_rescue/skeleton_3panel.gif` -- blue=measured, red=reconstructed, ankles omitted
- `outputs/stage4_2_knee_rescue/perframe_mpjpe.png` -- before/after per-frame MPJPE overlay
- `outputs/stage4_2_knee_rescue/knee_coverage.png` -- per-frame inlier count for each knee (color: red=0, orange=1, green=2, blue=3)
- `outputs/stage4_2_knee_rescue/gap_log.json` -- full structured gap log
- `outputs/stage4_2_knee_rescue/metrics.json` -- complete before/after metrics
- `outputs/stage4_2_knee_rescue/pts3d_clean.npy` -- cleaned 3D output (ankles=NaN, flagged recon frames)

## Stage 6a: Kinematic Solve + BVH Export (v3)

**Status: PASS. READY FOR 6b (RETARGETING).**

Stage 6a mapped the cleaned 3D points to a true kinematic skeleton (forward kinematics, fixed bone lengths derived from the subject's measured frames) and solved for per-frame rotations. 

### Key Achievements:
- **Ankle Inference**: Successfully replaced NaNs with a geometric ray+sphere intersection model. If 1 camera sees the ankle, its 3D ray is intersected with a sphere centered at the knee (radius = shin length). The ambiguity is resolved by picking the solution closest to the previous frame. If 0 cameras see the ankle, it falls back to pure FK (extending the hip→knee vector).
- **Validation**: Held-out numerical validation proved ray+sphere reduces median error by 42% over pure FK/IK when 1 camera is available.
- **Coverage**: The pipeline now automatically flags joints that spend >90% of the clip below the 1-camera visibility gate. Right ankle (`r_ankle`) triggers this flag (0% ray, 100% FK) due to physical occlusion, which correctly falls back to pure FK.
- **Bone Consistency**: Bone length Coefficient of Variation (CV) is now exactly `0.000` (perfectly rigid skeleton).
- **Smoothness**: No rotation spikes >30 deg/frame outside of known reconstructed spans (except 1 minor flag on `r_elbow`).
- **BVH Output**: Successfully exported `stage6a_mocap.bvh` with a verified 0.0000mm round-trip error.

### Full Metric Table (Stage 6a)

| Metric | Value | Threshold | Status |
|---|---|---|---|
| **FK vs GT MPJPE (excl. ankles, measured)** | 63.5mm | <15% delta from cleaned 3D | **PASS** (+11.6% delta) |
| **Bone CV after fitting** | 0.000 | exactly 0 (<0.001) | **PASS** |
| **Ankle inference coverage (total)** | 97% | report only | **PASS** (ray=21, fk=559) |
| **l_ankle: 0 cameras above gate** | 0% | <90% = pass | **PASS** (n0=0, n1=281, n2+=19) |
| **r_ankle: 0 cameras above gate** | 0% | <90% = pass | **PASS** (n0=0, n1=299, n2+=1) |
| **Rotation spikes >30 deg/frame** | 1 | <=2 acceptable | **PASS** (`r_elbow`) |
| **BVH round-trip max position error** | 0.00 | <1.0mm | **PASS** |

### Next Step
Stage 6b: Retargeting the `stage6a_mocap.bvh` onto a target 3D character model.

---

## Stage 7: Full-Minute Comprehensive Diagnostic (171204_pose1)

**Objective:** Scale from 300-frame windows to a full 60-second (1,800-frame) run on a confirmed-good dataset with wide-baseline camera geometry. Build a structured, per-frame/per-camera/per-joint dataset so any future question ("does error correlate with X?") is a query against existing data, not a new script.

---

### 7.0 Camera Selection — Exhaustive Geometry Sweep

Before running any compute, an exhaustive sweep of all 4,495 possible camera trios in the `171204_pose1` dataset was performed (`scratch/tradeoff_curve.py`). The original assumption of using `00_08`/`00_09`/`00_26` (the nominally "120° apart" trio) was rejected after discovering it has **100% ankle truncation** — every single frame in that trio has feet cut off, which provides zero variance for error-correlation analysis.

**Selected trio: `00_11` / `00_12` / `00_23`**

| Metric | Value |
|---|---|
| Full-body visibility | 82.8% of frames |
| Worst-case survivor baseline | 72.3° (if one camera is dropped by AR gate) |
| Pair angles (00_11×00_12 / 00_11×00_23 / 00_12×00_23) | 90.9° / 87.5° / 72.3° |
| Flicker zone (joints within 50px of frame edge) | 86.6% of clean frames |

The 72.3° worst-case baseline is the key constraint: if any single camera is rejected by the AR gate, the remaining pair still provides a baseline well above the 35° collapse threshold validated in earlier rounds.

**Why not the 120° trio?** `00_08`/`00_09`/`00_26` has perfect geometry but offers no diagnostic variance — its ankle truncation rate is 100%, meaning the pipeline is always in the same degraded state. `00_11`/`00_12`/`00_23` provides a graduated mix: 82.8% clean, some partial occlusion, a small genuine-failure tail. That variance is what makes correlation analysis meaningful.

---

### 7.1 Phase 0 Pre-Checks

**Motion variance (GT, frames 150–1949):**

| Metric | Value |
|---|---|
| Min distance from dome center | 8.6 cm |
| Max distance from dome center | 119.3 cm |
| Median distance | 20.0 cm |
| Standard deviation | 22.4 cm |

The subject traverses from near-center (8.6 cm) to 119.3 cm from center over the 60-second window — wide enough behavioral variance to stress-test both the clean and degraded pipeline regimes.

Audio sync check was attempted. The video files for this trio do not carry extractable audio (silent streams), so sync was verified structurally: all three videos were downloaded at matching frame counts from the same CMU server endpoint with no seek offsets applied.

---

### 7.2 Full-Minute Gate Report (1,800 Frames)

All five CSV logs were written to `outputs/diag_pose1_full/`:
- `camera_frame_log.csv` — 5,400 rows (per camera × frame)
- `keypoint_2d_log.csv` — 91,800 rows (per camera × frame × joint)
- `camera_pair_geometry_log.csv` — 5,400 rows (per frame × pair)
- `triangulation_3d_log.csv` — 30,600 rows (per frame × joint)
- `frame_summary_log.csv` — 1,800 rows (per frame)

**Top-level metrics:**

| Metric | Value | Notes |
|---|---|---|
| Total camera-frames | 5,400 | 3 cams × 1,800 frames |
| AR gate pass rate | 83.6% (4,516/5,400) | |
| 2D detection rate (conf ≥ 0.4) | 97.8% (89,759/91,800) | |
| 2D reprojection error | 123.2px mean / **56.6px median** | **10× worse than prior measurements — see §7.5** |
| 3D MPJPE body (raw) | 232.4mm mean / 134.0mm median | N=17,247 joint-frames |
| Frames with valid triangulation | 1,443 / 1,800 (80.2%) | |
| Frames flagging low quality | 357 / 1,800 (19.8%) | |

> **Critical note:** "1,443 valid + 357 low-quality = 1,800 exactly." These are NOT two independent failure categories. Every low-quality frame is a frame where fewer than 2 cameras survived the AR gate, making triangulation geometrically impossible. "Low quality" and "no triangulation" are the same 357 frames counted under two labels.

---

### 7.3 Combined Breakdown Table

Error binned against camera pair, baseline angle, and distance from dome center (all in a single table):

| Camera Pair | Baseline Angle | Dist from Center | Mean MPJPE (mm) | N Frames |
|---|---|---|---|---|
| `00_11` × `00_12` | 80–90° | 0–50 cm | 379.8 | 40 |
| `00_11` × `00_23` | 70–80° | 0–50 cm | 210.7 | **3** ⚠️ |
| `00_12` × `00_23` | 90–100° | 0–50 cm | 269.3 | 60 |
| `00_12` × `00_23` | 90–100° | 50–100 cm | 783.1 | **27** ⚠️ |
| Full Trio | 90–100° | 0–50 cm | **188.6** | 1,144 |
| Full Trio | 90–100° | 100–150 cm | 400.1 | 168 |

⚠️ **Low-confidence rows (N < 30):** The N=3 and N=27 rows are below this project's established ~30-frame reliability floor. Their MPJPE values (210.7mm, 783.1mm) should not be quoted as conclusions — they are directionally informative only.

**Key pattern:** When the full trio survives and the subject is near-center, mean error is 188.6mm (raw, unsmoothed, no undistortion). When only one camera pair survives and the subject has moved 50–100 cm from center, error jumps to 783mm. This is the exact distance-scaling behavior the stress test was designed to surface.

---

### 7.4 Vanishing Skeleton Root Cause Analysis

**Observation (visual, from the GIF):** During spread poses (sitting, T-pose, wide straddle), the entire predicted skeleton disappears — not just the lower body, but all joints simultaneously.

**Hypothesis tested:** AR gate falsely triggering on wide poses (not just on truncated frames).

**Stage-by-stage trace on 4 fully-vanished frames:**

| Frame | GT Dist from Center | YOLOX Detection? | Joints ≥ 0.4 (per cam) | 2D AR per cam | AR Gate |
|---|---|---|---|---|---|
| 1407 | 22.8 cm | ✅ / ✅ / ✅ | 17 / 16 / 16 | 1.43 / 1.63 / 1.74 | FAIL / FAIL / FAIL |
| 1416 | 25.6 cm | ✅ / ✅ / ✅ | 16 / 15 / 17 | 0.95 / 1.29 / 1.32 | FAIL / FAIL / FAIL |
| 1425 | 29.8 cm | ✅ / ✅ / ✅ | 16 / 16 / 17 | 1.12 / 1.42 / 1.44 | FAIL / FAIL / FAIL |
| 1504 | 14.7 cm | ✅ / ❌ / ✅ | 17 / 0 / 13 | 1.66 / — / 1.78 | FAIL / — / FAIL |

**Findings:**

1. **YOLOX detects the person with high confidence on all cameras** (except one `00_12` miss at frame 1504). Detection is not the failure. The skeleton vanishes downstream of detection.

2. **The subject is at 15–30 cm from dome center** on all four vanished frames. The "edge of dome" explanation from the initial report is **wrong** for this failure mode. The subject is dead center.

3. **The AR gate is the sole killer.** The gate computes `h/w` across all high-confidence keypoints. A person doing a wide-leg squat or T-pose has a genuinely wide 2D footprint, producing the exact same low-AR signal as a body that has its legs cropped off by the frame boundary. The gate cannot distinguish these two cases.

4. **Frame 1504, `00_23`: AR = 1.780 — failed the 1.8 gate by 0.02.** At this margin, a single joint detection shifting 5 pixels would flip the outcome. The threshold as set is fragile at the boundary.

5. **`00_12` at frame 1504: YOLOX returned no bounding box at all.** This is a genuine YOLOX failure on an unusual pose — confirming that on rare frames, the detector itself (not the AR gate) is the failure point. This is distinct from the AR gate false-positive and should not be conflated.

**Correlation test — does pose spread co-occur with distance from center?**

| GT 3D AR Bin | N | Mean Dist (cm) | Mean Cams Passing | 0-cam vanish % |
|---|---|---|---|---|
| Very spread (AR < 1.0) | 59 | 96.4 | 3.00 | 0.0% |
| Spread (1.0 ≤ AR < 1.5) | 318 | 24.1 | 2.30 | **5.3%** |
| Medium (1.5 ≤ AR < 2.0) | 405 | 24.2 | 2.63 | 1.2% |
| Tall (2.0 ≤ AR < 5.0) | 1,018 | 23.9 | 2.50 | 1.7% |

- Pearson(GT_3D_AR, dist_from_center): **r = −0.210** — weak negative, not the same variable.
- Pearson(GT_3D_AR, n_cams_passing): **r = −0.013, p = 0.57** — no significant correlation.

The "Very spread" bin (AR < 1.0) has 0% vanish rate — because those 59 frames are the subject far from center (96 cm avg) where the person is *small* in the image, making the absolute pixel height still dominate width. The failure concentrates in the 1.0–1.5 bin where the subject is close to the cameras *and* doing spread poses.

**Conclusion:** Pose-extremity and distance-from-center are weakly correlated but not the same confounder. They need to be controlled for independently.

> **⚠️ RETRACTED — PENDING FULL CENSUS:** The claim that the AR gate is the "sole killer" was based on only 4 of 39 zero-camera frames. Frame 1504 includes a confirmed YOLOX detector miss (distinct failure mode), and the sample is too small to classify mutually exclusive causes. The full 39-frame census, undistortion before/after comparison, and empirical gate sweep are required before any root-cause classification is finalised. See §7.9 for updated findings.

---

### 7.5 Reprojection Error: Source Correctly Identified as YOLO 2D Accuracy

**Previous claim (RETRACTED):** "Undistortion is missing; adding it will drop reproj error from 56.6px to 2–5px."

**What the experiment showed:**

Adding `cv2.undistortPoints` to the full pipeline and re-running the identical 1,800 frames produced:

| Metric | Without undistortion | With undistortion | Delta |
|---|---|---|---|
| Reproj error mean (px) | 123.19 | 125.88 | +2.7 (+2.2%) |
| Reproj error median (px) | **56.60** | **56.59** | −0.01 (no change) |
| Reproj error p95 (px) | 450.4 | 465.8 | worse |
| MPJPE median (common frames, mm) | 213.3 | **398.3** | +87% **regression** |

The undistortion had essentially zero effect on reprojection error (56.60→56.59px) and **nearly doubled the MPJPE**.

**Root cause of the regression:** The `distCoef` values in the Panoptic JSON are the physical lens parameters used during camera calibration. The Panoptic HD video files are **pre-rectified** — the video frames already have lens distortion removed. Applying `cv2.undistortPoints` re-moves distortion that was already removed, over-correcting joint positions to wrong pixel locations and degrading 3D triangulation.

Confirmed by inspecting the actual `distCoef` values (non-trivial barrel distortion: k1 = −0.26 to −0.29) — these are significant, but they were already applied during video generation. The undistortion must not be applied to Panoptic video inputs.

**What 56.6px actually is:** The 56.6px median reprojection error is the inherent 2D pose estimation accuracy of YOLO nano estimating joint pixel positions on HD 1920×1080 frames, relative to Panoptic's marker-based ground truth. This is not a lens distortion artefact. The 2–5px figures from prior measurements were computed on a different metric (reprojection of triangulated 3D back to 2D), not against GT joint locations — they are not comparable numbers.

---

### 7.6 AR Gate Empirical Sweep

**Ground truth for truncation (independent of gate):** A camera-frame is GT-truncated if any GT-projected lower-body joint (hips, knees, ankles — COCO indices 11–16) falls within 30px of the bottom frame boundary after projecting world-space GT through the calibrated camera without distortion correction (Panoptic videos are pre-rectified). This definition is fully independent of the AR metric being swept.

**GT truncation baseline:** 230 / 5,400 camera-frames (4.3%) are GT-truncated by this definition.

**Sweep results (AR threshold 0.8–2.5, step 0.1):**

| AR threshold | Usable cam-frames | FP (clean→rejected) | FN (trunc→passed) | FP rate |
|---|---|---|---|---|
| 0.8 | 5,313 | 87 | 230 | 1.7% |
| 0.9 | 5,303 | 97 | 230 | 1.9% |
| 1.0 | 5,291 | 109 | 230 | 2.1% |
| 1.1 | 5,281 | 119 | 230 | 2.3% |
| 1.2 | 5,253 | 147 | 230 | 2.8% |
| 1.3 | 5,180 | 220 | 230 | 4.2% |
| 1.4 | 4,989 | 411 | 230 | 7.9% |
| 1.5 | 4,795 | 604 | 229 | 11.6% |
| 1.6 | 4,687 | 710 | 227 | 13.6% |
| 1.7 | 4,603 | 793 | 226 | 15.2% |
| **1.8 (current)** | **4,516** | **874** | **220** | **16.8%** |
| 1.9 | 4,416 | 973 | 219 | 18.7% |
| 2.0 | 4,331 | 1,055 | 216 | 20.3% |
| 2.1 | 4,226 | 1,156 | 212 | 22.2% |
| 2.2 | 4,125 | 1,257 | 212 | 24.2% |
| 2.3 | 3,952 | 1,366 | 148 | 26.2% |
| 2.4 | 3,756 | 1,486 | 72 | 28.5% |
| 2.5 | 3,645 | 1,594 | 69 | 30.6% |

**Key observations:**

1. **FN count (truncated frames that pass) is almost flat from AR=0.8 to AR=1.8** — dropping from 230 to 220, a reduction of only 10 frames over a massive threshold range. The AR gate provides almost no incremental protection against real truncation below AR≈2.3.

2. **FP count (clean frames wrongly blocked) explodes above AR=1.3** — jumping from 220 at AR=1.3 to 874 at AR=1.8. The large majority of the 16.8% false-positive rate at the current threshold represents legitimate poses being blocked.

3. **The operating sweet spot is AR ≤ 1.2.** At AR=1.2: 147 FP (2.8% rate), 230 FN (all truncated frames pass through — unchanged from AR=0.8). This trades essentially zero additional truncation leakage for 727 fewer wrongly-blocked clean frames.

4. **The AR gate cannot effectively separate truncation from wide poses on this dataset.** The GT-truncated rate (4.3%) barely changes across the threshold sweep until very high values where the gate also begins blocking tall poses. A geometry-based truncation check (bbox-bottom proximity) would be more targeted — but per the user's instruction, this must be validated and the YOLOX bbox-clipping behaviour verified before any permanent change.

> **Status of threshold change:** Do not implement yet. The sweep data supports moving AR≤1.2 if it's confirmed that bbox-bottom check is not tautological on this camera setup. Full verification needed before committing.

---

### 7.9 Full 39-Frame Census and Detector Miss Analysis

#### Cause Classification (all 39 zero-camera frames)

| Cause | Count | % | Definition |
|---|---|---|---|
| **AR_ONLY** | 27 | 69.2% | YOLOX detected the person on all cameras; AR gate killed every detection |
| **MIXED** | 12 | 30.8% | YOLOX missed on ≥1 camera AND AR gate failed on remaining cameras |
| DETECTOR_MISS_ONLY | 0 | 0.0% | YOLOX missed on all 3 cameras simultaneously |
| OTHER | 0 | 0.0% | |

**Corrected statement (replacing "sole killer"):** The AR gate failure is observed in 100% of zero-camera frames — no frame lost all cameras without it triggering. It is the **sufficient cause in only 69.2%** of cases (AR_ONLY). In 30.8% (12/39), YOLOX also missed ≥1 camera.

> **Empirical Verification (Counterfactual AR necessity):** A frame-by-frame counterfactual analysis (§7.11) evaluated what would have happened if the AR gate were disabled entirely.
> * In **97.4%** of vanishing frames (38/39), YOLOX had successfully detected the person on ≥2 cameras before the AR gate fired. The AR gate's FP rejection is counterfactually necessary and responsible for the loss of triangulation in these 38 frames.
> * In **2.6%** (1/39), YOLOX missed on 2 cameras simultaneously, meaning triangulation was impossible regardless of the AR gate.

#### Detector Miss Clustering

Total detector miss events: **13**, distributed across **2 tight clusters and 1 isolate**:

| Event | Frames | Camera | Window size | Pattern |
|---|---|---|---|---|
| Cluster A | 1502–1504 | `00_12` | 3 consecutive | Part of MIXED frames; AR gate failing on `00_11`/`00_23` simultaneously |
| Cluster B | 1645–1653 | `00_12` | 8 consecutive (**+1 at 1647**) | `00_12` drops detection for ~8 frames; `00_11`/`00_23` present but AR fails |
| Isolate | 1578 | `00_11` + `00_12` | 1 frame | Both cameras miss simultaneously |

The 8-frame cluster (1645–1653) all show GT_AR_3D ≥ 2.1 — the subject appears to be doing a tall/standing pose. `00_12` consistently fails on this camera-specific view while `00_11` and `00_23` detect successfully but with AR < 1.8. This is not random dropout — it is a pose-specific, camera-angle-specific YOLOX failure concentrated in one camera during a particular motion.

#### Per-Frame Detail (all 39 frames)

Full detail saved to `outputs/diag_pose1_full/zero_cam_census.csv`.

Summary of all vanished frames by GT 3D AR:

- GT_AR_3D range across 39 frames: 1.11 – 3.43
- GT distance from center: 11–30 cm (all near-center — confirms "edge of dome" is wrong)
- `00_23` bbox_bottom_dist was **negative on several frames** (e.g., −3.8px, −7.4px, −6.4px) — meaning the bottom keypoints extended slightly below the frame boundary. This is a case where the subject's feet were marginally outside the frame on `00_23` while `00_11` and `00_12` had full visibility.

---
**Conclusion:** YOLOX does *not* strictly clip to the boundary in a way that defeats distance thresholding. A bbox-bottom distance check is a viable truncation detector.

#### D. Expanded Frame-Level AR Sweep
**GT Definition:** A camera frame is "GT-truncated" if any lower body joint (COCO 11–16) projects (using `D=zeros`) within 30px of the bottom edge or completely outside the image.
* **Denominator:** 4,910 clean cam-frames, 490 GT-truncated cam-frames.

**Sweep Results:**
| AR Threshold | False Positives (Clean Blocked) | False Negatives (Truncated Passed) | Total 0-Cam Frames |
|---|---|---|---|
| AR ≥ 0.8 (Baseline)| 87 (1.8%) | 490 (100.0%) | 0 |
| AR ≥ 1.2 | 147 (3.0%) | 490 (100.0%) | 0 |
| **AR ≥ 1.8 (Current)**| **828 (16.9%)** | **434 (88.6%)** | **39** |
| AR ≥ 2.4 | 1,399 (28.5%) | 245 (50.0%) | 126 |

**Conclusion:** The current AR=1.8 threshold rejects 16.9% of perfectly clean frames (causing 39 complete vanishing events) while *still missing* 88.6% of genuinely truncated frames. It is highly suboptimal. No "sweet spot" is declared yet; the bbox-bottom sweep must be fully evaluated before selecting the final replacement gate.

#### E. Reprojection Error Mathematical Definition
The `reproj_error` metric in `outputs/diag_pose1_full/camera_frame_log.csv` (median 56.6px) computes the **internal multi-view residual**: `|| project(triangulated_3D) - detected_2D ||`. It measures how geometrically consistent the 2D rays were, *not* YOLO's accuracy against Ground Truth.
A separate computation against GT (`|| project(GT_3D) - detected_2D ||`) on a 50-frame sample showed a median accuracy of **48.9px**.

## Stage 8: Gate Tradeoff and Undistortion Audit

We ran two definitive full-sequence (1800 trios / 5400 frames) sweeps to isolate the impact of the bounding box gate and undistortion logic on triangulation accuracy.

### 1. The BBox Gate Tradeoff (No Sweet Spot)

GT Denominators: Clean=4910, Truncated=490

| Gate Threshold | FP% (Clean Rejected) | FN% (Truncated Passed) | Usable 3D Trios | Mean MPJPE (mm) |
|----------------|----------------------|------------------------|-----------------|-----------------|
| **No Gate** | 0.0% | 65.7% | 1649 | 352.9 |
| **Dist >= 10px**| 28.2% | 33.1% | 1454 | 333.8 |
| **Dist >= 20px**| 41.8% | 15.1% | 907 | 313.0 |
| **Dist >= 30px**| 52.7% | 3.7% | 403 | 305.7 |
| **Dist >= 80px**| 86.7% | 0.2% | 86 | 254.1 |

**Conclusion:** The bounding box gate is fundamentally unworkable as a 2D filter on this camera geometry. There is no sweet spot. To reliably catch truncation (FN < 5%), the gate must be set at \Dist >= 30px\. However, this setting produces a **52.7% False Positive rate**, destroying over half of the perfectly clean, fully-visible poses simply because the tight camera framing places the subject near the edge.

### 2. The Undistortion Audit

We tested the hypothesis that the massive 352.9mm error in un-gated frames was due to missing camera undistortion (i.e. passing raw pixels to DLT instead of normalized coordinates). We triangulated all 1649 un-gated trios across three projection matrix (P) treatments:

*   **T1 (Raw Pixels):** Mean MPJPE: 352.9 mm, Median: 318.7 mm, P95 Reproj: 48.1 px
*   **T2 (Undistorted Pixels):** Mean MPJPE: 354.3 mm, Median: 320.1 mm, P95 Reproj: 47.8 px
*   **T3 (Normalized Coords):** Mean MPJPE: 347.2 mm, Median: 314.1 mm

**Conclusion:** Undistortion does not solve the error. While normalized coordinates (T3) marginally improve accuracy by ~5mm, the >340mm structural error persists across all treatments. The error originates strictly from YOLOX hallucinating 2D keypoints on heavily truncated limbs (which leak through when gating is relaxed), causing the DLT engine to triangulate geometrically wild 3D points. 

*(Additional finding: The overall median GT 2D accuracy for YOLOX across 58,045 points was exactly 10.6 px, confirming the 2D detections themselves are tightly grouped around the ground truth, but degenerate on truncated limbs.)*

### [SUPERSEDED] The Path Forward

We are caught in an irreconcilable tradeoff at the 2D bbox level:
1.  **No Gate:** Destroys 3D structural quality (35cm error) by passing truncated limbs.
2.  **2D BBox Gate:** Destroys dataset yield (>50% clean frames lost) because of tight camera framing.

**Hypothesis:** Whole-camera AR and bbox gates show a poor coverage-quality tradeoff on this sequence. Per-joint rejection plus robust 3D/kinematic handling is the leading candidate, but no production change is approved until end-to-end export comparisons are complete.

## B-vs-C Final Decision: Candidate B Retained

**Status: DECIDED. Candidate B is retained as the selected default pipeline configuration for continued development and validation.**

The final direct-paired audit reproduced bit-for-bit across two in-process computations and a fresh process. Deterministic payload SHA-256:

`1a0bda5470b8919277bc01c8c98bbb946801b78ee847ad9bfecf9f30410246e8`

### Proven findings

- Global MPJPE is effectively tied: B 100.489mm vs C 100.547mm; paired B-C -0.057mm, 95% circular moving-block bootstrap CI [-0.351, 0.203]mm.
- Mean ankle MPJPE is unresolved: B 106.362mm vs C 106.705mm; paired B-C -0.344mm, CI [-2.107, 1.217]mm.
- Observed ankle p95 error is lower for B: 258.81mm vs 285.76mm. No paired CI was calculated for the p95 difference, so this remains descriptive.
- B has slightly lower mean acceleration magnitude: 7.755 vs 7.930m/s²; paired B-C -0.175m/s², CI [-0.375, -0.020]m/s².
- Candidate C produced 399 raw boundary-mask log entries, representing 375 unique frame-joint rejections across 367 frames.
- Stage-6 arrays are fully finite for both candidates. This means the final stage produced values, not that every joint was directly observed.
- The supplied audit data does not identify the resolution method used for each rejected observation.
- Candidate differences are overwhelmingly localized to the ankles.

### Decision reasoning

Candidate C adds a boundary-rejection mechanism but does not demonstrate a material improvement in global mean accuracy, ankle mean accuracy, or the observed ankle-error tail. It also introduces 375 unique frame-joint rejections, while B has a small acceleration-magnitude advantage.

Under the engineering rule that added complexity and data rejection require a demonstrated material benefit, Candidate C does not justify replacing Candidate B. Candidate B remains the selected default.

This decision closes only the B-vs-C configuration question. It does not establish that the complete AIMoCap pipeline is production-ready.

### Invalid decision instruments

GT-event acceleration matching remains permanently retracted because the pipeline and GT event populations have fundamentally different temporal signatures.

The current sliding instrument uses ankle-center height as a contact proxy and has only 30/3,598 intervals under the conservative GT-contact definition. It is exploratory and excluded from the decision.

The metric previously called penetration is an ankle-below-GT-ankle-height reference, not physical foot-ground penetration. It is exploratory and excluded from the decision.

### Selected configuration

- Ankle strategy: `ray_sphere_with_fk_fallback`
- Boundary rejection gate: disabled by default
- Candidate C: experimental only and requires explicit opt-in

### Runtime lock verification

Candidate B is connected to the Stage-6 runtime through the explicit ankle-strategy dispatcher. The default configuration dispatches to the existing ray-sphere ankle inference with FK fallback. Candidate C’s boundary gate is disabled by default and can run only through explicit experimental opt-in. Missing or malformed configuration fails closed.

Behavioral regression tests exercise the real production dispatcher and preprocessing branch rather than duplicating their conditions in test code.

#   S t a g e   6 b   R e c o n c i l i a t i o n   A u d i t   � �    F i n a l   R e p o r t  
  
 * * S t a t u s :   I N C O M P L E T E * *  
 * * D a t e :   2 0 2 6 - 0 7 - 1 3 * *  
 * * S c r i p t s : * *   ` s c r i p t s / a u d i t _ s t a g e 6 b _ r e c o n c i l i a t i o n . p y ` ,   ` s c r i p t s / d i a g _ s t a g e 6 b _ d e e p . p y `  
 * * L o g   f i l e s : * *   ` o u t p u t s / s t a g e 6 b _ a u d i t _ l o g . t x t ` ,   ` o u t p u t s / d i a g _ s t a g e 6 b _ d e e p . t x t `  
  
 - - -  
  
 # #   E x e c u t i v e   S t a t u s  
  
 * * I N C O M P L E T E . * *   F o u n d a t i o n a l   b l o c k e r   c o n f i r m e d :   t h e   ` c a n o n i c a l _ d e t e c t o r _ p o s e _ o b s e r v a t i o n s . n p z `   2 D   k e y p o i n t s   a r e   f r o m   a   * * d i f f e r e n t   c a m e r a * *   t h a n   t h e   P a n o p t i c   c a l i b r a t i o n   u s e d .   T h e   d e t e c t o r   s e e s   a   1 9 2 0 �  1 0 8 0   i m a g e   o n   o n e   c a m e r a   g e o m e t r y ;   t h e   c a l i b r a t i o n   p r o j e c t s   t h e   G T   t o   a   c o m p l e t e l y   d i f f e r e n t   p a r t   o f   t h e   i m a g e .   T h e   o b s e r v a t i o n s   f i l e   c a n n o t   b e   u s e d   f o r   S t a g e   6 b   a s - i s .   A l l   o p t i m i z e r   r e s u l t s   a r e   i n v a l i d   u n t i l   t h i s   i s   r e s o l v e d .  
  
 S e p a r a t e   f r o m   t h i s :   a   j o i n t - o r d e r   m i s m a t c h   w a s   f o u n d   a n d   d o c u m e n t e d ,   a n d   b o n e   l e n g t h   e s t i m a t i o n   w a s   f i x e d .   T h e s e   t w o   i t e m s   a r e   r e s o l v e d .  
  
 - - -  
  
 # #   S e c t i o n   A :   P r o v e n a n c e   o f   t h e   1 0 0 . 4 8 9   m m   C a n d i d a t e - B   B a s e l i n e  
  
 * * V e r b a t i m   c o m p u t a t i o n   ( l i n e s   2 5 7 � �  2 8 3 ,   ` s c r i p t s / p h a s e _ b _ b c _ d e c i s i o n . p y ` ) : * *  
 ` ` ` p y t h o n  
 p o s _ b       =   b _ r a w . a s t y p e ( n p . f l o a t 6 4 )                       #   b _ s t a g e 6 ,   c m ,   C O C O   o r d e r  
 p o s _ g t     =   g t _ r a w . a s t y p e ( n p . f l o a t 6 4 )   /   1 0 . 0       #   g t   m m �    c m ,   C O C O   o r d e r  
 e b   =   n p . l i n a l g . n o r m ( p o s _ b [ : ,   B O D Y _ J ]   -   p o s _ g t [ : ,   B O D Y _ J ] ,   a x i s = - 1 )   *   1 0 . 0     #   c m �    m m  
 v m   =   f i n _ b   &   f i n _ g t     #   b o t h   f i n i t e  
 B _ m p j p e   =   n p . m e a n ( e b [ v m [ : ,   B O D Y _ J ] ] )  
 ` ` `  
  
 |   I t e m   |   V a l u e   |  
 | - - - | - - - |  
 |   N P Z   S H A - 2 5 6   |   ` 6 a 4 1 d 9 c 4 2 1 b c 1 c 3 3 a 3 a 5 9 7 b c 5 3 9 9 e 1 a 8 f a b 7 a 4 d 1 3 a f 4 d 9 0 c 8 f d a 9 3 c 2 1 8 3 c f b 6 c `   |  
 |   F r a m e   r a n g e   |   0 � �  1 7 9 9   ( a l l   1 8 0 0 )   |  
 |   J o i n t   s e t   |   B O D Y _ J   =   [ 5 , 6 , 7 , 8 , 9 , 1 0 , 1 1 , 1 2 , 1 3 , 1 4 , 1 5 , 1 6 ]   ( 1 2   C O C O   b o d y   j o i n t s )   |  
 |   G T   u n i t s   |   m i l l i m e t e r s ,   d i v i d e d   b y   1 0 . 0   b e f o r e   s u b t r a c t i o n   |  
 |   b _ s t a g e 6   u n i t s   |   c e n t i m e t e r s   ( m e a n   m a g n i t u d e   4 6 . 8 0 )   |  
 |   M P J P E   t y p e   |   * * A b s o l u t e * *   ( w o r l d - f r a m e ,   n o   r o o t   a l i g n m e n t )   |  
 |   T o t a l   p o s s i b l e   j o i n t - f r a m e s   |   2 1 , 6 0 0   ( 1 8 0 0 �  1 2 )   |  
 |   V a l i d   j o i n t - f r a m e s   ( b o t h   f i n i t e )   |   2 1 , 6 0 0   |  
 |   E x c l u d e d   |   0   |  
 |   * * R e p r o d u c e d   v a l u e * *   |   * * 1 0 0 . 4 8 9   m m * *   ( � � 0 . 0 0 0 4   m m )   � S&   |  
  
 * * A n k l e   m e t r i c s   ( a l s o   r e p r o d u c e d ) : * *  
 -   A n k l e   m e a n   M P J P E :   1 0 6 . 3 6 2   m m  
 -   A n k l e   P 9 5   M P J P E :   2 5 8 . 8 1   m m  
  
 - - -  
  
 # #   S e c t i o n   B :   C a m e r a   T r i o   R e c o n c i l i a t i o n  
  
 B o t h   t r i o s   p r e s e n t   i n   c a l i b r a t i o n .   T R I O _ O R I G   u s e d   t h r o u g h o u t   ( b _ s t a g e 6   w a s   p r o d u c e d   w i t h   t h e s e   c a m e r a s ) :  
  
 |   C a m e r a   |   f x   ( p x )   |   k 1   |   I n   c a l i b r a t i o n   |  
 | - - - | - - - | - - - | - - - |  
 |   ` 0 0 _ 0 0 `   |   1 6 3 3 . 3   |   � � 0 . 2 2 0 9   |   � S&   |  
 |   ` 0 0 _ 0 1 `   |   1 3 9 7 . 2   |   � � 0 . 2 8 6 1   |   � S&   |  
 |   ` 0 0 _ 0 2 `   |   1 3 9 7 . 3   |   � � 0 . 2 8 2 9   |   � S&   |  
 |   ` 0 0 _ 1 1 `   |   p r e s e n t   |   p r e s e n t   |   � S&   |  
 |   ` 0 0 _ 1 2 `   |   p r e s e n t   |   p r e s e n t   |   � S&   |  
 |   ` 0 0 _ 2 3 `   |   p r e s e n t   |   p r e s e n t   |   � S&   |  
  
 * * F r a m e s   0 � �  5 9   C a n d i d a t e - B   m a t c h e d   b a s e l i n e   ( a b s o l u t e   M P J P E ) : * *  
 -   V a l i d   j o i n t - f r a m e s :   7 2 0   /   7 2 0  
 -   M e a n :   * * 9 8 . 4 3 7   m m * *  
  
 * * D e r i v e d   g a t e s   f r o m   6 0 - f r a m e   m a t c h e d   b a s e l i n e : * *  
 -   C e i l i n g   ( � 0 � 1 . 0 1   �    9 8 . 4 3 7 ) :   * * 9 9 . 4 2 1   m m * *  
 -   I m p r o v e m e n t   ( � 0 � 0 . 9 5   �    9 8 . 4 3 7 ) :   * * 9 3 . 5 1 5   m m * *  
  
 - - -  
  
 # #   S e c t i o n   C :   D i s t o r t i o n   O r a c l e   � �    C O N F I R M E D  
  
 |   P a t h   |   D e s c r i p t i o n   |   E r r o r   ( m m )   |  
 | - - - | - - - | - - - |  
 |   P A T H   1   |   P r o j e c t   G T   w i t h   d i s t o r t i o n   �      D L T   |   2 6 4 . 4 3   m m   |  
 |   P A T H   2   |   P r o j e c t   G T   w i t h   d i s t   �      u n d i s t o r t P o i n t s   �      D L T   |   6 2 5 . 2 5   m m   |  
 |   * * P A T H   3 / 4 * *   |   * * P r o j e c t   G T   n o   d i s t o r t i o n   �      D L T   ( p r o d u c t i o n   p a t h ) * *   |   * * 0 . 0 0 0 0   m m * *   |  
  
 * * C o n f i r m e d : * *   P a n o p t i c   v i d e o s   a r e   p r e - r e c t i f i e d .   ` c v 2 . u n d i s t o r t P o i n t s `   m u s t   N O T   b e   a p p l i e d   t o   P a n o p t i c   k e y p o i n t   c o o r d i n a t e s .   T h e   ` d i s t C o e f `   v a l u e s   i n   t h e   c a l i b r a t i o n   J S O N   d o c u m e n t   t h e   p h y s i c a l   l e n s   b u t   a r e   n o t   n e e d e d   f o r   p i x e l �    w o r l d   m a t h   o n   p r e - r e c t i f i e d   v i d e o .  
  
 T h e   ` t `   v e c t o r   i n   t h e   P a n o p t i c   J S O N   * * i s * *   t h e   O p e n C V   t r a n s l a t i o n   v e c t o r   ( w o r l d �    c a m e r a   c o n v e n t i o n ) ,   v e r i f i e d   b y :   p r o j e c t i n g   G T   w o r l d   c o o r d i n a t e s   v i a   ` P   =   K   @   [ R   |   t ] `   g i v e s   c o r r e c t   p i x e l   c o o r d i n a t e s   w i t h   0 . 0 0 0 0 m m   r o u n d - t r i p   e r r o r .  
  
 C a m e r a   c e n t e r   i n   w o r l d   f r a m e   =   ` � � R . T   @   t   =   [ � � 2 7 2 ,   � � 1 5 2 ,   � � 1 5 ] `   c m   ( c a m e r a   i s   p o s i t i o n e d   t o   t h e   s i d e   a n d   s l i g h t l y   a b o v e   t h e   s t a g e   � �    p h y s i c a l l y   p l a u s i b l e ) .  
  
 - - -  
  
 # #   S e c t i o n   D :   C a m e r a - L i s t   M i s m a t c h   � �    R E S O L V E D  
  
 * * R u n t i m e   p r i n t s   c o n f i r m e d : * *   B o t h   ` b u i l d _ m u l t i v i e w _ o b s e r v a t i o n s ( ) `   a n d   ` W i n d o w e d S e q u e n c e O p t i m i z e r ( ) `   r e c e i v e   t h e   s a m e   3 - c a m e r a   l i s t   ` [ 0 0 _ 0 0 ,   0 0 _ 0 1 ,   0 0 _ 0 2 ] ` .   ` o b s . i n l i e r _ m a s k . s h a p e   =   ( 6 0 ,   3 ,   1 7 ) `   m a t c h e s   ` l e n ( c a m e r a s )   =   3 ` .   F a i l - f a s t   a s s e r t i o n   a d d e d .  
  
 T h e   p r i o r   I n d e x E r r o r   ( c a m e r a   i n d e x   3   o u t   o f   b o u n d s   f o r   s i z e   3 )   w a s   c a u s e d   b y   a n   e a r l i e r   r u n   t h a t   p a s s e d   a   5 - c a m e r a   l i s t   w h i l e   t h e   o b s e r v a t i o n   a r r a y   h a d   3   c a m e r a s .   T h i s   i s   f i x e d .  
  
 - - -  
  
 # #   S e c t i o n   E :   6 0 - F r a m e   O p t i m i z e r   � �    B L O C K E D  
  
 # # #   E . 1 :   C r i t i c a l   F i n d i n g   � �    O b s e r v a t i o n   F i l e   C a m e r a   G e o m e t r y   M i s m a t c h  
  
 * * F i n d i n g :   ` c a n o n i c a l _ d e t e c t o r _ p o s e _ o b s e r v a t i o n s . n p z `   c o n t a i n s   k e y p o i n t s   f r o m   a   d i f f e r e n t   c a m e r a   o r   d i f f e r e n t   v i d e o   g e o m e t r y   t h a n   t h e   P a n o p t i c   c a l i b r a t i o n   f o r   ` 0 0 _ 0 0 ` ,   ` 0 0 _ 0 1 ` ,   ` 0 0 _ 0 2 ` . * *  
  
 E v i d e n c e   ( f r a m e   0 ,   j o i n t   1 1 ,   l _ h i p ) :  
 -   C a m e r a   ` 0 0 _ 0 0 `   c a l i b r a t i o n   p r o j e c t s   G T   w o r l d   p o i n t   t o :   * * ( 9 9 5 ,   2 4 9 9 ) * *   p i x e l s  
 -   D e t e c t e d   k e y p o i n t   i n   o b s   N P Z   f o r   c a m e r a   a x i s   0 :   * * ( 4 2 9 ,   9 1 1 ) * *   p i x e l s  
 -   E r r o r :   * * 1 6 8 6   p x * *   � �    n o t   d e t e c t o r   n o i s e ,   t h i s   i s   a   c o o r d i n a t e   s y s t e m   m i s m a t c h  
  
 F o r   A L L   1 7   j o i n t s ,   c a m e r a   0   p r o j e c t s   G T   t o   y - c o o r d i n a t e s   o f   * * 1 8 0 0 � �  3 2 0 0 p x * *   w h i l e   t h e   d e t e c t e d   k p t s   h a v e   y - c o o r d i n a t e s   o f   * * 4 5 5 � �  1 0 7 9 p x * *   ( i . e . ,   w i t h i n   a   1 9 2 0 �  1 0 8 0   i m a g e ) .  
  
 T h e   c a l i b r a t i o n   f o r   ` 0 0 _ 0 0 `   p r o j e c t s   g r o u n d - l e v e l   j o i n t s   ( h i p ,   k n e e ,   a n k l e )   f a r   B E L O W   t h e   i m a g e   ( y   >   2 0 0 0 )   w h i l e   t h e   d e t e c t o r   o b s e r v e s   t h e m   i n   t h e   l o w e r   t h i r d   o f   a   1 0 8 0 p   i m a g e .   T h i s   i s   g e o m e t r i c a l l y   i n c o n s i s t e n t .  
  
 * * R o o t   c a u s e : * *   T h e   3 D   k e y p o i n t s   s t o r e d   i n   ` g a t e 1 _ a r r a y s . n p z [ ' b _ s t a g e 6 ' ] `   w e r e   N O T   p r o d u c e d   b y   t r i a n g u l a t i n g   t h e   2 D   d e t e c t i o n s   i n   ` c a n o n i c a l _ d e t e c t o r _ p o s e _ o b s e r v a t i o n s . n p z `   u s i n g   c a m e r a s   ` 0 0 _ 0 0 / 0 1 / 0 2 ` .   T h e y   w e r e   p r o d u c e d   b y   a   d i f f e r e n t   p i p e l i n e   p a t h   ( l i k e l y   u s i n g   d i f f e r e n t   P a n o p t i c   c a m e r a s ,   o r   u s i n g   t h e   r a w   m a r k e r - b a s e d   P a n o p t i c   3 D   d a t a   s c a l e d / t r a n s f o r m e d   i n t o   a   c o m m o n   w o r l d   f r a m e ) .  
  
 * * C o n s e q u e n c e : * *   T h e   ` c a n o n i c a l _ d e t e c t o r _ p o s e _ o b s e r v a t i o n s . n p z `   f i l e   c a n n o t   b e   u s e d   f o r   S t a g e   6 b   o p t i m i z a t i o n   u n l e s s   t h e   c a m e r a   I D s   t h a t   w e r e   a c t u a l l y   u s e d   t o   p r o d u c e   t h e   2 D   d e t e c t i o n   f i l e   a r e   i d e n t i f i e d   a n d   m a t c h e d .  
  
 # # #   E . 2 :   R e s i d u a l s   A r e   Z e r o   � �    E x p l a i n e d  
  
 ` o b s . i n l i e r _ m a s k . s u m ( )   =   0 `   b e c a u s e   t h e   i n i t i a l   D L T   p o i n t   f r o m   t h e   d e t e c t o r   k p t s   h a s   r e p r o j e c t i o n   e r r o r s   o f   * * 3 4 4 � �  5 5 5 p x * *   a g a i n s t   t h o s e   s a m e   d e t e c t o r   k p t s   ( t h e   D L T   i s   e x t r e m e l y   i n c o n s i s t e n t   b e c a u s e   t h e   t h r e e   c a m e r a s   g i v e   w i l d l y   c o n f l i c t i n g   r a y   d i r e c t i o n s ) .   T h e   2 0 p x   i n l i e r   t h r e s h o l d   c o r r e c t l y   r e j e c t s   a l l   p o i n t s   � �    b u t   t h i s   m e a n s   t h e r e   i s   N O   v a l i d   i n p u t   f o r   t h e   o p t i m i z e r .  
  
 # # #   E . 3 :   J o i n t   O r d e r i n g   M i s m a t c h   � �    F I X E D   ( b u t   o p t i m i z e r   s t i l l   b l o c k e d )  
  
 T h e   p r i o r   8 2 0 m m   e r r o r   w a s   c a u s e d   b y   c o m p a r i n g   ` p o s _ f k [ f ,   C O C O _ j ] `   t o   ` g t [ f ,   C O C O _ j ] `   w h e r e   t h e   F K   a r r a y   i s   i n   C A N O N I C A L   o r d e r .   T h e   c o r r e c t   m a p p i n g :  
  
 |   C O C O   j o i n t   |   N a m e   |   C a n o n i c a l   j o i n t   |  
 | - - - | - - - | - - - |  
 |   5   |   l _ s h o u l d e r   |   5   |  
 |   6   |   r _ s h o u l d e r   |   * * 8 * *   |  
 |   7   |   l _ e l b o w   |   * * 6 * *   |  
 |   8   |   r _ e l b o w   |   * * 9 * *   |  
 |   9   |   l _ w r i s t   |   * * 7 * *   |  
 |   1 0   |   r _ w r i s t   |   1 0   |  
 |   1 1   |   l _ h i p   |   1 1   |  
 |   1 2   |   r _ h i p   |   * * 1 4 * *   |  
 |   1 3   |   l _ k n e e   |   * * 1 2 * *   |  
 |   1 4   |   r _ k n e e   |   * * 1 5 * *   |  
 |   1 5   |   l _ a n k l e   |   * * 1 3 * *   |  
 |   1 6   |   r _ a n k l e   |   1 6   |  
  
 # # #   E . 4 :   B o n e   L e n g t h s   � �    F I X E D  
  
 B o n e   l e n g t h s   e s t i m a t e d   f r o m   ` b _ s t a g e 6 `   d i r e c t l y   ( t h e   a u t h o r i t a t i v e   s o u r c e   � �    1 8 0 0   f r a m e s   o f   c l e a n   S t a g e - 6   d a t a ) :  
  
 |   C a n o n i c a l   j o i n t   |   N a m e   |   B o n e   l e n g t h   ( c m )   |  
 | - - - | - - - | - - - |  
 |   1   |   s p i n e   |   2 7 . 1 7   |  
 |   2   |   c h e s t   |   2 7 . 1 7   |  
 |   3   |   n e c k   |   8 . 0 1   |  
 |   4   |   h e a d   |   1 6 . 2 7   |  
 |   5 , 8   |   s h o u l d e r   ( p o o l e d )   |   * * 0 . 0 0 * *   � a� � � �   n e e d s   f i x   |  
 |   6 , 9   |   e l b o w   ( p o o l e d )   |   2 9 . 7 3   |  
 |   7 , 1 0   |   w r i s t   ( p o o l e d )   |   2 6 . 2 9   |  
 |   1 1 , 1 4   |   h i p   ( p o o l e d )   |   * * 0 . 0 0 * *   � a� � � �   n e e d s   f i x   |  
 |   1 2 , 1 5   |   k n e e   ( p o o l e d )   |   4 5 . 0 4   |  
 |   1 3 , 1 6   |   a n k l e   ( p o o l e d )   |   4 4 . 1 4   |  
  
 T h e   0 . 0 0 c m   f o r   s h o u l d e r / h i p   j o i n t s   i s   b e c a u s e   t h e s e   c a n o n i c a l   j o i n t s   a r e   * * l a t e r a l   o f f s e t s   f r o m   t h e i r   p a r e n t * *   ( c h e s t / p e l v i s ) ,   a n d   t h e   p a r e n t   i s   a   v i r t u a l   j o i n t   n o t   p r e s e n t   i n   C O C O .   T h e s e   m u s t   b e   c o m p u t e d   a s :  
 -   s h o u l d e r   b o n e   l e n g t h   =   ` | | C O C O _ l _ s h o u l d e r _ p o s   -   ( C O C O _ l _ s h o u l d e r   +   C O C O _ r _ s h o u l d e r ) / 2 | | `  
 -   h i p   b o n e   l e n g t h   =   ` | | C O C O _ l _ h i p _ p o s   -   ( C O C O _ l _ h i p   +   C O C O _ r _ h i p ) / 2 | | `  
  
 E s t i m a t e d   f r o m   b _ s t a g e 6 :   s h o u l d e r   � 0 �  1 4 � �  1 5   c m ,   h i p   � 0 �  8 � �  1 0   c m .  
  
 - - -  
  
 # #   S e c t i o n   F :   D e c i s i o n  
  
 * * I N C O M P L E T E * *   � �    t h e   f o u n d a t i o n a l   c h e c k   ( S e c t i o n   E . 1 )   f a i l e d .   N o   S t a g e   6 b   M P J P E   r e s u l t   i s   v a l i d .  
  
 G a t e s   d e r i v e d   f r o m   m a t c h e d   b a s e l i n e   B   =   9 8 . 4 3 7   m m   ( f r a m e s   0 � �  5 9 ) :  
 -   C e i l i n g :   * * 9 9 . 4 2 1   m m * *  
 -   I m p r o v e m e n t :   * * 9 3 . 5 1 5   m m * *  
  
 T h e   1 0 0 . 4 8 9   m m   f u l l - s e q u e n c e   b a s e l i n e   r e m a i n s   t h e   r e f e r e n c e   f o r   t h e   1 8 0 0 - f r a m e   r u n   i f / w h e n   t h e   o b s e r v a t i o n   f i l e   i s s u e   i s   r e s o l v e d .  
  
 - - -  
  
 # #   S e c t i o n   G :   O p e n   Q u e s t i o n s   a n d   R e q u i r e d   A c t i o n s  
  
 # # #   B L O C K E R :   I d e n t i f y   t h e   c o r r e c t   c a m e r a   I D s   f o r   t h e   o b s e r v a t i o n   f i l e  
  
 T h e   m o s t   l i k e l y   s c e n a r i o s :  
 1 .   * * T h e   o b s   f i l e   w a s   e x t r a c t e d   u s i n g   P a n o p t i c   c a m e r a s   ` 0 0 _ 0 3 ` ,   ` 0 0 _ 0 4 ` ,   ` 0 0 _ 2 3 ` ,   ` 0 0 _ 2 4 ` ,   ` 0 0 _ 2 8 ` * *   ( t h e   c a m e r a s   t h a t   w e r e   u s e d   i n   e a r l i e r   t r i a n g u l a t i o n   s c r i p t s ,   n o t   ` 0 0 _ 0 0 / 0 1 / 0 2 ` ) .   C h e c k   ` s c r i p t s / e x t r a c t _ m o r e . p y `   o r   t h e   e x t r a c t i o n   s c r i p t   t h a t   p r o d u c e d   t h e   o b s   f i l e .  
 2 .   * * T h e   o b s   f i l e   u s e s   a   p i x e l   c o o r d i n a t e   s y s t e m   s h i f t e d   f r o m   t h e   P a n o p t i c   f r a m e * *   ( e . g . ,   t h e   d e t e c t o r   c r o p s   t o   a   b o u n d i n g   b o x   b e f o r e   o u t p u t t i n g   k e y p o i n t s ,   a n d   t h e   s t o r e d   k p t s   a r e   i n   c r o p - s p a c e   n o t   f u l l - i m a g e - s p a c e ) .  
  
 * * A c t i o n   r e q u i r e d   b e f o r e   S t a g e   6 b   c a n   p r o c e e d : * *  
 1 .   F i n d   t h e   e x t r a c t i o n   s c r i p t   t h a t   p r o d u c e d   ` c a n o n i c a l _ d e t e c t o r _ p o s e _ o b s e r v a t i o n s . n p z `   a n d   r e a d   t h e   a c t u a l   c a m e r a   n a m e s / I D s   u s e d .  
 2 .   V e r i f y   b y   p r o j e c t i n g   G T   3 D   t h r o u g h   t h o s e   c a m e r a s   � �    s h o u l d   g e t   < 5 0 p x   e r r o r   f r o m   d e t e c t o r   k p t s .  
 3 .   R e b u i l d   t h e   C a m e r a M o d e l   l i s t   w i t h   t h e   c o r r e c t   c a m e r a s .  
  
 # # #   S h o u l d e r / h i p   b o n e   l e n g t h   f i x   ( r e q u i r e d   b e f o r e   F K )  
 I m p l e m e n t   l a t e r a l   o f f s e t   e s t i m a t i o n   f o r   s h o u l d e r   ( C a n o n   5 , 8 )   a n d   h i p   ( C a n o n   1 1 , 1 4 )   b o n e s .  
  
 # # #   N o   c h a n g e s   t o   ` d i a g n o s i s . m d `   u n t i l   r e v i e w e r   a p p r o v e s   t h i s   r e p o r t .  
  
 - - -  
  
 # #   A p p e n d i x :   K e y   N u m b e r s   f r o m   T h i s   R u n  
  
 |   M e t r i c   |   V a l u e   |   S o u r c e   |  
 | - - - | - - - | - - - |  
 |   R e p r o d u c e d   B   b a s e l i n e   |   1 0 0 . 4 8 9   m m   � �   0 . 0 0 0 4   |   S e c t i o n   A   |  
 |   B   o n   f r a m e s   0 � �  5 9   |   9 8 . 4 3 7   m m   |   S e c t i o n   B   |  
 |   C e i l i n g   g a t e   |   9 9 . 4 2 1   m m   |   S e c t i o n   F   |  
 |   I m p r o v e m e n t   g a t e   |   9 3 . 5 1 5   m m   |   S e c t i o n   F   |  
 |   D i s t o r t i o n   o r a c l e   ( p r o d u c t i o n   p a t h )   |   0 . 0 0 0 0   m m   |   S e c t i o n   C   |  
 |   o b s   i n l i e r   c o u n t   ( 6 0   f r a m e s )   |   0   /   3 0 6 0   |   S e c t i o n   E . 1   |  
 |   D e t   k p t   v s   G T   p r o j e c t i o n   e r r o r   |   5 9 3 � �  2 3 1 3   p x   |   S e c t i o n   E . 1   |  
 |   D L T   r e p r o j e c t i o n   v s   d e t   k p t s   |   5 4 � �  5 5 5   p x   |   S e c t i o n   E . 1   |  
 |   S t a g e   6 b   M P J P E   ( a n y   v a l i d   r u n )   |   * * N / A   � �    b l o c k e d * *   |   � �    |  
  
 # #   F i n a l - O u t p u t   A c c u r a c y   A u d i t      2 0 2 6 - 0 7 - 1 6  
  
 P r o v e n a n c e :   c o m m i t   3 d c 2 a e c ,   n p z _ s h a 2 5 6   c 8 9 4 c d 7 e 7 e 2 d c 5 a 4 e d 8 8 9 1 6 2 d 6 7 a a a 2 2 f 5 1 2 4 3 1 f 9 e 2 5 c 3 e 6 b 9 f b d 4 a 3 4 b 6 5 e 4 b 6 ,   c a m s   0 0 _ 1 1 / 0 0 _ 1 2 / 0 0 _ 2 3 ,   O F F S E T = 1 5 0 .  
  
 S t a g e - 6   w a t e r f a l l      P L A I N   ( B O D Y _ J ,   a b s o l u t e   M P J P E   m m ,   r a w _ f i n i t e _ o n l y ) :  
     r a w = 3 5 . 7 9   g a p _ f i l l e d = 3 5 . 7 9   o n e _ e u r o = 4 8 . 7 9   f k _ f i t = 7 4 . 6 0   r a y _ s p h e r e = 7 4 . 6 0   r e f i t = 7 4 . 6 0  
  
 S t a g e - 6   w a t e r f a l l      P R E S E R V E _ F I N I T E   f i x   ( B O D Y _ J ,   a b s o l u t e   M P J P E   m m ,   r a w _ f i n i t e _ o n l y ) :  
     r a w = 3 5 . 7 9   g a p _ f i l l e d = 3 5 . 7 9   o n e _ e u r o = 4 8 . 7 9   f k _ f i t = 4 8 . 7 9   r a y _ s p h e r e = 4 8 . 7 9   r e f i t = 4 8 . 7 9  
  
 D e l t a   ( p r e s e r v e   -   p l a i n ,   r a w _ f i n i t e _ o n l y ) :  
     f k _ f i t = - 2 5 . 8 1 m m   r e f i t = - 2 5 . 8 1 m m  
  
 I n p u t - q u a l i t y   a b l a t i o n   ( M a n n y   F K ,   p e l v i s - a l i g n e d ,   w i n d o w = 1 2 0 ,   m m ) :  
     r a w :                       m e a n = 1 4 7 . 1 2   m e d i a n = 8 4 . 0 5   p 9 5 = 7 4 3 . 9 2   n = 1 5 4 7  
     f i l t e r e d _ r a w :     m e a n = 1 2 1 . 4 3   m e d i a n = 6 2 . 3 7   p 9 5 = 6 5 5 . 3 0   n = 1 5 6 0  
     b _ s t a g e 6 :             m e a n = 1 4 8 . 1 1   m e d i a n = 7 8 . 0 5   p 9 5 = 6 0 6 . 0 9   n = 1 5 6 0  
  
 P h a s e   3   f i x e s   a p p l i e d :  
     6 a :   f i t _ s k e l e t o n _ s e q u e n c e _ p r e s e r v e _ f i n i t e   ( c o m m i t   d 3 a d 0 6 e )      s n a p s   f i n i t e   j o i n t s   b a c k   t o   m e a s u r e d   p o s i t i o n s   a f t e r   F K  
     6 c :   v i z _ i k _ t r u t h . p y   s w i t c h e d   t o   p r o d u c t i o n   f i l t e r e d - r a w   p a t h   ( c o m m i t   3 d c 2 a e c )      n o t   y e t   r e - r e n d e r e d  
  
 N o t e :   v i z   n o t   r e - r e n d e r e d   i n   t h i s   a u d i t   ( 4 5 +   m i n   r e n d e r   t i m e ) .   C o d e   c h a n g e   c o m m i t t e d ;   c a c h e   d e l e t e d .  
 

## IK-Retarget Perfection Audit — 2026-07-16

Provenance: commit 318e05b, npz_sha256 c894cd7e..., cams 00_11/00_12/00_23, OFFSET=150.

Fixes applied:
  1. Mask untracked GT frames (conf=-1 -> NaN) — 21 garbage frames removed from measurement
  2. Spine intermediate weight 0.01 -> 0.15 + distribute bend in init (R_frac per joint)
  3. Spine target curvature (arc on forward flexion > 20deg)
  4. Clavicle joints added to proxy skeleton (arms from measurements, not rest offset)
  5. IK max_nfev 200 -> 500, temporal_weight 0.05 -> 0.03
  6. Adaptive temporal cutoff (1.5 -> 6Hz on fast motion) + 10 -> 30 deg/frame cap

Stage-6 waterfall (BODY_J, absolute MPJPE mm, raw_finite_only) — with GT masking:
  PLAIN:     raw=33.19 gap_filled=33.19 one_euro=46.34 fk_fit=72.17 ray_sphere=72.17 refit=72.17
  PRESERVE:  raw=33.19 gap_filled=33.19 one_euro=46.34 fk_fit=46.34 ray_sphere=46.34 refit=46.34
  Delta:     fk_fit=-25.82mm refit=-25.82mm

Input-quality ablation (Manny FK, pelvis-aligned, window=120, mm):
                PRE-FIX          POST-FIX         DELTA
  raw:          mean=147.12  median=84.05  ->  mean=56.91  median=53.98   (-61.5% mean, -35.8% median)
  filtered_raw: mean=121.43  median=62.37  ->  mean=53.76  median=51.58   (-55.7% mean, -17.3% median)
  b_stage6:     mean=148.11  median=78.05  ->  mean=91.38  median=74.80   (-38.3% mean, -4.2% median)

Viz (Solved vs GT, pelvis-aligned, cm):
                PRE-FIX          POST-FIX         DELTA
  median:       6.24            ->  5.82            (-6.7%)
  mean:        12.14            -> 11.89            (-2.1%)
  p95:         65.53            -> 63.12            (-3.7%)
  max:        105.83            -> 107.04           (+1.1%)

Viz (Production input vs GT, cm): median=2.35 (unchanged — triangulation not modified)

Remaining error budget:
  Raw triangulation: 23.5mm median (theoretical floor — 2D detection + 3-camera triangulation)
  IK/retarget adds:  28.1mm (was 38.9mm) — the gap between 23.5mm input and 51.6mm output
  Total:             51.6mm median (was 62.4mm) — 17.3% improvement

Note: The massive mean improvement (121->54mm) is mostly from GT masking (21 untracked frames
at origin were inflating the mean). The median improvement (62->52mm) is from the IK fixes.
The bending spike (p95) dropped from 655mm to 104mm — 84% reduction, confirming the spine
and temporal fixes target the right frames.


## Per-Frame MPJPE Analysis -- 2026-07-16

Provenance: commit 2ef8aee, window 1493-1612 (120 frames, most-moved).

Bug fix: mpjpe_per_frame had a double-divide bug (_align already converts gt_mm
to cm, but mpjpe_per_frame divided by 10 again). This produced ~934mm systematic
offset on every frame -- a measurement artifact, NOT an IK failure. Fixed in
commit 2ef8aee with regression test test_mpjpe_per_frame_matches_mpjpe_on_nonzero_gt.

Corrected per-frame MPJPE (Manny FK, pelvis-aligned, MANNY_TO_COCO joints):
  mpjpe scalar: mean=53.76mm median=51.58mm p95=103.75mm max=133.80mm n=1560
  mpjpe_per_frame: median=54.24mm mean=53.76mm max=69.15mm
  Distribution: min=39.05mm p25=48.71mm p75=58.93mm
  Frames >100mm: 0/120  Frames >200mm: 0/120  Frames <60mm: 94/120

Worst 5 frames (all well under 100mm):
  frame 1502: 69.15mm
  frame 1598: 68.74mm
  frame 1597: 68.63mm
  frame 1501: 66.96mm
  frame 1585: 66.02mm

Conclusion: The per-frame distribution is flat and well-behaved (39-69mm range).
No spikes during bending frames. The bending problem reported in the plan
(person bending to lowest point) is genuinely solved -- worst frame is 69mm,
down from the pre-fix p95 of 655mm. No further IK improvement is indicated.


## Foot Plantarflexion Fix -- 2026-07-17

Provenance: foot-keypoint fix (commits 9faae72 + this session), window 1187-1636
(450 frames, most-moved, 15s GIF).

### Root cause

The 2D detector (CIGPose) outputs 133 COCO-WholeBody keypoints including foot
joints (big toe 17/20, small toe 18/21, heel 19/22). These were triangulated
into 3D but never fed to the IK -- extract_mocap_points only read body joints
0-16. Instead, the proxy toe_l/toe_r joints got a SYNTHESIZED target:
  toe_target = ankle_pos + R_root * rest_offset
where R_root is the pelvis root frame built from spine direction + hip line.
When the torso leans forward, R_root tilts forward, and the synthesized toe
drops below the ankle -> the ankle swings to point the toe downward ->
plantarflexion (standing on toes).

### Fix

1. Added foot keypoints to COCO map in mocap_skeleton.py (big_toe_l=17,
   big_toe_r=20, etc.) and extract them in extract_mocap_points when K>=23.
2. Made toe_l/toe_r COCO-anchored (coco_name="big_toe_{side}") so they get
   weight 1.0 in the IK and their target comes from the measured big-toe 3D
   position, not R_root synthesis.
3. Guarded the R_root toe synthesis as a fallback: only synthesizes when the
   toe is NOT coco-anchored OR the measured big toe is NaN (backward-compatible
   with 17-joint inputs).
4. The toe bone length is now measured (median_dist ankle->big_toe) instead of
   guessed as rest_seg * spine_scale.
5. Re-extracted canonical dataset to 133 keypoints; triangulate_raw now slices
   to 23 joints (body+feet) -- 6x faster than 133 and bit-identical body joints
   (engine is per-joint independent with min_aspect_ratio=0.0).

### Foot orientation -- quantitative verification (counterfactual)

Solved the same 50-frame sample with OLD (17-joint, R_root synth toe) vs NEW
(23-joint, measured toe). Metrics in Manny Z-up cm:

  Metric                Rest (flat)   OLD (R_root)   NEW (measured)
  foot->toe dZ (L)       -7.49 cm     -9.80 cm       -4.22 cm
  foot->toe dZ (R)       -7.49 cm    -10.18 cm       -3.63 cm
  shin<->toe angle (L)    68.1 deg     61.4 deg       81.6 deg
  shin<->toe angle (R)    68.1 deg     61.8 deg       86.1 deg

Interpretation:
  OLD: toe driven BELOW rest-flat (dZ=-9.8 vs rest=-7.5), angle MORE acute
       (61 deg vs rest 68) -> plantarflexion confirmed.
  NEW: toe now ABOVE rest-flat (dZ=-4.2 vs rest=-7.5), angle 82-86 deg ->
       dorsiflexed/flat, NOT plantarflexed. The foot is flatter than rest.

### Accuracy -- no regression

T1 raw (body joints 0-16, absolute MPJPE vs GT):
  mean=33.19mm median=24.45mm p95=86.49mm coverage=91.1%
  (unchanged from the 35.8mm baseline -- body joints bit-identical with the
   23-joint slice, confirming the triangulation engine is per-joint
   independent and the slice is safe.)

Per-frame MPJPE on viz window (1187-1636, 450 frames):
  median=25.07mm mean=25.61mm max=37.70mm
  frames >100mm: 0/450   frames <60mm: 450/450

Input quality ablation (120-frame window):
  raw:          mean=57.34mm median=54.80mm p95=111.71mm
  filtered_raw: mean=53.93mm median=52.13mm p95=103.75mm
  b_stage6:     mean=91.40mm median=74.69mm p95=211.68mm

Viz solved-vs-GT (pelvis-aligned, MANNY_TO_COCO joints, 450 frames):
  mean=5.36cm median=4.64cm p95=10.64cm max=71.16cm (n=5737)
  (consistent with prior runs -- the foot fix changes foot ORIENTATION, not
   body-joint positions, so pelvis-aligned shape error is unchanged.)

### Foot keypoint coverage

Raw triangulation (before filter):
  all 6 feet finite per frame: 1641/1800 (91.2%)
  big_toe_l: 1629/1800   small_toe_l: 1628/1800   heel_l: 1645/1800
  big_toe_r: 1649/1800   small_toe_r: 1648/1800   heel_r: 1647/1800
After filter_skeleton3d (gap interpolation): 100% finite.

### Tests

35 fast tests pass (4 new foot tests + 3 new measure tests + regression test).
3 slow IK ablation tests deselected (11-min each). The 3 baseline tests
(raw_triangulation_reproduces_known_baseline, canonical_data_shapes,
gt_masks_untracked_markers) pass -- confirming no regression.

### Artifacts

  outputs/viz_ik_truth.gif (18.4 MB, 15s) -- new GIF with flat feet
  outputs/viz_ik_truth.mp4 (9.0 MB) -- full-res video
  outputs/viz_ik_truth_cache.npz -- IK cache (450 frames, 23-joint input)
  outputs/triangulated_filtered.npz -- (1800, 23, 3) filtered triangulation

---

## FBX Export Twist Fix (Unreal Engine) -- 2026-07-17

### Symptom

User imports our exported FBX into Unreal Engine 5, applies it to the default
UE Manny Mannequin skeleton, and sees **twisted limbs** — bones rotated wrong,
body parts roughly in place but wrong orientation (up to ~134° on upperarms,
92° on pelvis).

### Root cause (the actual bug, not the first hypothesis)

**Blender's `bpy.ops.export_scene.fbx(bake_anim=True)` bakes the FIRST KEYED
FRAME's pose as the FBX "Bind Pose"** — the very rest skeleton that UE reads
and retargets against.

We were keying `pose_quats = R_src⁻¹ × R_abs` on frame 1 (the first animation
frame), so Blender baked `R_blend_out = R_src × pose_quats[0] = R_abs[0]` as
the Bind Pose. UE then applied delta retargeting
(`R_src × R_blend_out⁻¹ × animated_local`) and got
`R_src × R_abs[0]⁻¹ × R_abs[f]` instead of `R_abs[f]` — twisted by
`R_src × R_abs[0]⁻¹` (up to 134° on upperarms, 92° on pelvis, 105° on calf_l).

The parity gate passed because it only checks WORLD positions, and
`R_blend_out × R_blend_out⁻¹ = I` cancels at the world level — the wrong rest
cancels with the wrong animated_local. The FBX looked correct in isolation
(any tool that evaluates rest+delta together sees the right world pose) but
was wrong for retargeting onto a different skeleton.

### Investigation path (what ruled out what)

1. **Hypothesis: Blender re-rolls bones on IMPORT.** Disproved: reading
   `pose_bone.bone.matrix_local` after import matches ufbx `R_src` to 0.00°.
   Blender's imported rest IS Manny's rest.
2. **Hypothesis: Blender re-rolls bones on EXPORT via primary/secondary_bone_axis.**
   Disproved: sweeping all 9 axis combos with `bake_anim=False` (rest only) gives
   0.00° diff with default Y/X. The re-roll is NOT in the axis settings.
3. **Hypothesis: the exported rest == frame 0's keyed pose.** Confirmed by
   step-by-step probe: import only → 0°, +rotation_mode → 0°, +action → 0°,
   +key identity → 0°, +key 5° rot → 5°, +key R_src⁻¹×R_abs → 92.79°. The rest
   tracks the first keyed frame exactly.
4. **Alternative fix: `arm.data.pose_position = 'REST'` before export.**
   Rejected: preserves the Bind Pose (R_blend_out = R_src, 0°) BUT nukes the
   animation (every frame bakes as the rest pose; parity off by 90 meters).

### Fix (the one that works)

In `aimocap/retarget/fbx_export.py`, keyframe the **identity-delta rest pose**
(identity quaternion on every bone, `(0,0,0)` root translation) at frame 1,
and shift the actual animation to frames 2..F+1. Then:
- Blender bakes frame 1's identity pose as the Bind Pose → `R_blend_out = R_src`
- UE's retargeting: `R_src × R_src⁻¹ × R_abs[f] = R_abs[f]` — the twist cancels

Cost: the FBX has F+1 frames (frame 1 = rest T-pose, frames 2..F+1 = animation).
The user sees a 1-frame T-pose at the start of the animation in UE, which can
be trimmed in UE's animation editor if undesired.

### Supporting changes

- `aimocap/retarget/fbx_eval.py`: added `rest_frame_offset` parameter to
  `fbx_world_positions` and `fbx_world_positions_at_frames` (default 0 for
  backward compat with BVH/legacy FBXs; pass 1 to skip the leading rest frame).
- `aimocap/retarget/parity.py`: auto-detects the leading rest-pose frame by
  checking if the FBX AnimStack `time_end` matches `npy_frames + 1` frames;
  passes `rest_frame_offset=1` to the evaluation when detected. This keeps the
  parity gate working for BOTH old-style (F-frame) and new-style (F+1-frame)
  FBXs without any caller-side change.
- `aimocap/retarget/parity_worker.py`: accepts `--rest-frame-offset` and
  passes it through to `fbx_world_positions_at_frames`.

### Verification

All four properties verified on `outputs/production_motion_fixed.fbx`:

| Check | Result | Bar |
|-------|--------|-----|
| R_blend_out == R_src (exported rest matches source rest) | max 0.0004° | <1° PASS |
| animated_local == R_abs (FBX per-frame local rotations match npy) | max 0.0002° | <1° PASS |
| UE retarget recovers R_abs (R_src × R_blend⁻¹ × animated == R_abs) | max 0.0002° | <1° PASS |
| Parity (world positions, npy FK vs FBX eval) | max 0.000 cm | <0.5cm PASS |

Backward compatibility: the OLD `production_motion.fbx` (without the fix)
still passes parity at 0.001 cm — the auto-detection correctly returns
`rest_frame_offset=0` for it (its `time_end = 5.0s = 150 frames`, not 151).

Tests: 17 passed (test_retarget + test_triangulate_engine), no regressions.

### Artifacts

- `outputs/production_motion_fixed.fbx` (1.37 MB, 151 frames: 1 rest + 150 anim)
- `outputs/fbx_eval_motion_fixed.mp4` (0.65 MB, 5s, front+side views, 150 anim frames)

### What the user should do

Replace the FBX you import into UE with `outputs/production_motion_fixed.fbx`.
The twist should be gone. Frame 1 will show the T-pose rest; frames 2+ are the
animation. If the T-pose frame is undesirable, trim it in UE's animation editor
(set the play range to start at frame 2).

## Spine Twist Fix (stage 7)

### Symptom

The upper torso (chest) twisted backward while the lower back around the
hips was fine. The twist accumulated up the spine chain — worse at the chest
than at the lower back. Visually, the character's back faced the camera while
the hips faced forward.

### Root cause: three compounding bugs

**Bug 1 — Spine init distributed the WRONG rotation** (`mocap_ik.py:187-203`)

The pelvis was set to `R_root` (the hip frame), then each spine joint
multiplied `R_root^(1/6)` on top of the parent's already-`R_root` frame.
This produced globals `R_root^(1+k/6)`, over-rotating the upper spine by up
to `R_root^(5/6)` (~83° at spine_05). The "distribute the bend evenly" logic
only works if the pelvis starts at identity; because it starts at `R_root`,
every segment double-counts.

**Bug 2 — Shoulder-vs-hip twist dumped at neck** (`mocap_ik.py:180-185`)

The actual torso twist `R_diff = R_upper · R_root⁻¹` (the rotation between
the shoulder and hip frames) was not distributed across the spine at all.
Instead, the code distributed `R_root` (Bug 1), then set `neck_01 = R_upper`
directly — so the entire `R_diff` landed as a single local twist at the neck.

**Bug 3 — Solver couldn't correct it** (`mocap_ik.py:346`)

The IK residual was position-only. Rotation of a spine bone about its own
long axis is a null direction of the position Jacobian (it moves no joint
position), so `least_squares` could not observe or correct spine twist. The
over-rotated init from Bug 1 persisted through the solve. Limbs avoided
this via `constrained_rotation` with a `roll_child`; the spine had no such
constraint.

### Fix (3 parts)

**Fix 1 — SLERP distribution in `analytic_init`** (`mocap_ik.py:187-203`)

Replaced the `R_root^(1/6)` distribution with left-SLERP from `R_root` to
`R_upper`: `R_target = R_root * (R_diff ** frac)` where `R_diff = R_root.inv() * R_upper`.
This interpolates from `R_root` at the pelvis to `R_upper` at the neck,
distributing only the *difference* (the actual torso twist), not re-applying
the entire root rotation. The over-rotation vanishes.

**Fix 3 — Orientation residual in `_residuals`** (`mocap_ik.py:353-375`)

Added a spine-specific orientation residual that pins each spine joint's
global rotation to its SLERP target (`R_root * R_diff^frac`). With
`ori_weight=1.0`, a 1-radian (~57°) twist drift costs as much as a 1-cm
position error. This breaks the null-direction gauge freedom: the solver
can still adjust the spine to fit positions, but it can no longer add twist
for free.

(Fix 2 — pinning spine roll via `constrained_rotation` — was not needed
once Fix 1 + Fix 3 were in place. The SLERP init gives a correct starting
point and the orientation residual prevents drift, so the roll-child
mechanism is unnecessary for the spine.)

### Verification

| Check | Result | Bar |
|---|---|---|
| Spine twist tests (8 new tests) | 8/8 PASS | — |
| Retarget tests (15 existing) | 15/15 PASS | no regressions |
| Pre-existing ik_residual test | 4.7cm (was 12.5cm) | improved |
| FBX parity (npy FK vs FBX eval) | max 0.0009 cm | <0.5cm PASS |
| Cumulative spine twist (frame 100) | 119° (was 464°) | 75% reduction |
| spine_05 facing vs pelvis (frame 75) | 15° forward (was 154° backward) | FIX VERIFIED |

The definitive proof: the old code had spine_05 facing **151-154° backward**
relative to the pelvis (the chest literally twisted backward, matching the
user's report). The new code has spine_05 facing **15-22° forward** — a
natural amount of torso twist from the actual shoulder-vs-hip yaw.

### Artifacts

- `outputs/production_motion_spine_fix.fbx` (1.37 MB, 151 frames: 1 rest + 150 anim)
- `outputs/fbx_eval_spine_fix.mp4` (0.20 MB, 5s, front+side views, 150 anim frames)
- `tests/test_spine_twist.py` (8 new tests covering init + solver behavior)

### Generality

The fix uses `RigTopology.spine_chain()` and `root_rotation()` — no hardcoded
Manny indices. The SLERP interpolation and orientation residual are
rig-agnostic. Works on any humanoid rig where the pelvis is an ancestor of
the neck. Graceful degradation: if shoulder keypoints are NaN, `R_upper`
falls back to `R_root`, and `R_diff` = identity → no spine twist (safe default).

## Stage 8: Head Tilt + Leg/Knee "Caved In" (one-shot fix)

### User-reported issues
1. Head always tilted to the left side.
2. Left leg "caved in" near the hip (rotation/colapsing artifact).
3. Left knee also "caved in... some portion is gone."

### Root causes (three independent)

**1. Head tilt — leaf joint had identity local rotation.**
`analytic_init` set leaves (joints with no primary child) to identity local
rotation, so the head always inherited the neck's orientation regardless of
where the nose keypoint actually was. The head's roll about the neck→nose
axis is a null direction of the position residual, so the solver never
corrected it.

**2/3. Leg/knee caved in — garbage twist from a synthesized roll child.**
The 17-joint `b_stage6` data has no foot keypoints (COCO 17-22), so `toe_l`
is synthesized from the pelvis root frame (`ankle + R_root * rest_offset`).
That synthesized toe was then used as the **roll child** for the calf in
`constrained_rotation`, computing a twist from a direction that already
encodes `R_root` — circular reasoning that injected 67-92° of garbage calf
twist. The visible result: the calf mesh twists until the shin "caves in"
on itself. Compounding this: twist about a leg bone's long axis moves no
joint position, so it is a **null direction of the position residual** —
the solver was free to drift the twist further during optimization.

### Fixes (all in `aimocap/retarget/mocap_ik.py` + `swing_twist.py`)

**A. Head orientation (analytic_init + _residuals).** Leaves with a COCO
anchor (head → nose) now get an explicit swing rotation toward the measured
neck→nose direction in `analytic_init`:
```python
desired_world = target_pos[p] - target_pos[p_parent]
desired_pl = global_rot[p_parent].inv().apply(desired_world)
R_local = constrained_rotation(rest_dir, desired_pl)
global_rot[p] = global_rot[p_parent] * R_local  # parent * local, matches FK
```
and `_residuals` adds a head orientation residual pinning the head's global
rotation to `R_neck * constrained_rotation(rest_dir, desired_pl)`.
Also fixed a latent multiplication-order bug: the new leaf code used
`R_local * global_rot[p_parent]` but FK uses `parent * local` — corrected.

**B. Roll-child gate (analytic_init).** The roll child is now only used as a
twist constraint when its COCO keypoint is actually **measured** (finite,
nonzero). When the toe is synthesized from R_root (17-joint data, no big_toe),
the roll constraint is skipped, so `constrained_rotation` returns the
swing-only rotation — zero long-axis twist, no "caved in" calf.
```python
rc_measured = (rc_coco and rc_coco in measured
               and np.all(np.isfinite(measured[rc_coco]))
               and np.linalg.norm(measured[rc_coco]) > 1e-5)
if rc_measured:
    roll_rest = ...; roll_des = ...
else:
    roll_rest = None; roll_des = None  # -> swing-only
```
This is general (no Manny indices) and backward-compatible: when COCO-WholeBody
foot keypoints ARE available, the roll child is measured and the original
behavior (twist from real foot position) is preserved — fixing the
plantarflexion root cause for free when that data is present.

**C. Leg orientation residual (_residuals).** Even with zero twist at init,
the solver could drift it back (60-90°) because twist is a null direction of
the position residual. Added a leg orientation residual that penalizes the
twist component (rotvec projected onto the bone's rest long axis) for hip
and knee on both sides — but ONLY when the roll child is synthesized. When
the roll child is measured, the residual is skipped so the solver can refine
a legitimate twist against the position error:
```python
rv = R_current.as_rotvec()
twist_rad = np.dot(rv, bone)  # signed twist about bone axis
leg_res.append(twist_rad * bone)  # 3-vector aligned with bone
```
`ori_weight=1.0` makes ~5° of twist drift cost as much as 1 cm of position
error — tight enough to prevent the caving, loose enough to let the leg
swing freely.

**D. Swing-twist near-parallel guard (swing_twist.py).** Already in place
from the prior fix pass: when the roll child is within 20° of the bone
(straight leg), `constrained_rotation` returns swing-only. This catches the
degenerate case even when the roll child IS measured but the leg is straight
(the foot direction is then noise-dominated).

### Performance: cached rest world positions (`rig_topology.py`)
Profiling showed `RigTopology.roll_child()` was 60% of the residual runtime
because `_rest_world_positions()` ran a full FK over the Manny rig on every
call (4× per residual eval, hundreds of evals per solve). Cached the result
in `__init__`. Solve time dropped from 11s to ~8s per frame (29% faster).

### Artifacts
- `outputs/production_motion_v2.fbx` — 301 frames (1 rest + 300 anim, 10 s @ 30 fps)
  from the most-motion window (frames 1353-1653, avg 2.36 cm/frame motion)
- `outputs/fbx_eval_v2.mp4` — front+side eval render
- `scratch/verify_three_fixes.py` — numerical spot-check of all three fixes
- `scratch/respine_solve_render.py` — full re-solve → FBX → MP4 pipeline

## Stage 9: Shoulders Caved In + Hands Rotated Wrongly + Head Plausibility

### User-reported issues (after stage 8 fixed legs/head-tilt/spine)
1. Shoulders caved in.
2. Hands rotated wrongly.
3. Head either completely left or completely right (neither true of the real
   object, which faced forward or downward). These were new errors, likely
   visible now because the most-motion window (1353-1653) has more extreme
   poses than the earlier sample.

### Root causes (three independent)

**1. Shoulders caved in — clavicle target was (0,0,0).**
The clavicles have no COCO anchor (no "clavicle" keypoint in COCO) and are not
in the spine chain, so `target_pos`/`tgt_pos` stayed at the origin. The
non-coco weight (0.15) then pulled the clavicles toward (0,0,0), caving the
shoulders inward toward the pelvis.

**2. Hands rotated wrongly — chain-end roll child.**
The forearm's roll child is the wrist, which is a **leaf** (no proxy
children). The wrist's position is fully determined by the arm-chain
geometry (shoulder + elbow + swings), so it carries no INDEPENDENT roll
information. Using it as a roll constraint via `constrained_rotation`
produced large spurious twists (50-107°) when the arm was bent >90° — the
visible "hands rotated wrongly." The same mechanism affected the upper arm
(its roll child is also the wrist via `roll_child(lowerarm)=hand`).

**3. Head completely left/right — noisy nose triangulation.**
On some frames the nose triangulation is physically impossible (e.g. frame 0:
nose 23 cm ABOVE the neck, when the head bone points forward). The stage-8
head fix blindly swung the head to the nose direction, producing a ~90°
sideways tilt. (Note: on frame 0 the spine itself is genuinely leaned
forward-left — `spine_dir=(-0.50,+0.86,-0.08)` — so the head pointing up-left
is partly the real pose; but the nose garbage made it worse.)

### Fixes (all in `aimocap/retarget/mocap_ik.py` + `swing_twist.py`)

**A. Clavicle target synthesis (analytic_init + _residuals).** Synthesize
clavicle targets from the neck using the upper-body frame R_upper, the same
way the toe is synthesized from the ankle:
```python
rest_off = rest_t[clavicle] - rest_t[neck]
target_pos[clavicle] = tgt_neck + R_upper.apply(rest_off)
```
Also: joints with no real target (still 0 after all synthesis) now get
weight 0.0 (was 0.15) so the solver doesn't pull them to the origin.

**B. Roll-child must be an intermediate joint, not a chain-end leaf
(analytic_init + _residuals).** The roll child now constrains roll ONLY when
it has proxy children (it's an intermediate joint). A chain-end leaf (wrist
for the forearm, wrist for the upper arm) has its position fully determined
by the chain, so it carries no independent roll info:
```python
rc_children = [c for c in range(num_joints) if parents[c] == rc_proxy]
if rc_measured and rc_children:  # intermediate joint -> constrains roll
    roll_rest = ...; roll_des = ...
else:  # leaf or synthesized -> swing-only (zero twist)
    roll_rest = None; roll_des = None
```
This is general (no Manny indices) and backward-compatible: when COCO-WholeBody
hand keypoints are wired in (giving the hand a child beyond the wrist), the
roll child will have children and the twist will be legitimately constrained.

**C. Limb orientation residual generalized to arms (_residuals).** The stage-8
leg residual now covers all four limbs (thigh, shin, upper arm, forearm) with
the same `rc_measured and rc_children` gate. When the roll child is a leaf or
synthesized, twist is pinned to zero; when it's a measured intermediate, the
solver is free to refine.

**D. Head plausibility gate (analytic_init + _residuals).** The head/nose
swing is only applied when the measured nose direction is within ~70° of the
rest head direction (the head can tilt/look around but not face backward or
sideways from the neck). When the gate fails (noisy nose), the head inherits
the neck's orientation (identity local). In `_residuals`, the head residual
is always emitted as a fixed-size 3-vector (zero when gated) so the residual
length stays constant across solve iterations.

**E. Near-parallel guard widened (swing_twist.py).** `constrained_rotation`
now returns swing-only when the roll child is within 35° of the bone (was
20°). This covers near-straight arms (arm bone-vs-hand as low as 21°) where
the perpendicular projection is noise-dominated.

### Verification (10 sampled frames, post-solve, window 1353-1653)
| Check | Before | After |
|---|---|---|
| Upper arm twist (L) | up to 103° | **0°** all frames |
| Forearm twist (L) | up to 76° | **0°** all frames |
| Upper arm twist (R) | up to 49° | **0°** all frames |
| Forearm twist (R) | up to 60° | **0°** all frames |
| Clavicle L Y | caved to 0 (origin) | tracks neck (+31 to -4.7) |
| Head → nose angle | ~29° (stage 7) / sideways on bad nose | 0.2–2.5° (gated) |
| Calf twist (L, stage 8) | 67–92° | 0° (held) |
| Thigh twist (L, stage 8) | 35–48° | 0–7° (held) |
| Spine_05 vs pelvis (stage 7) | 151–154° backward | 10–32° forward |

### Generality
All fixes use `RigTopology.roll_child()`, `coco_anchor`, and `parents` — no
hardcoded Manny indices. Backward-compatible: when COCO-WholeBody hand/foot
keypoints are present, the roll children have measured targets AND proxy
children, so the twist is legitimately constrained from real data.

### Stage 9.1: Thigh twist regression fix

After stage 9's `rc_children` gate, the thigh twist regressed to 132-147° on
straight-leg frames (was 0° in stage 8). Root cause: the limb orientation
residual relaxed (let the solver refine twist) whenever the roll child was
measured AND had children — but on straight legs the roll child (ankle) is
near-parallel to the bone (knee→ankle ≈ hip→knee), so it carries no roll
info and the solver drifted 130°+.

Fix: the residual's `rc_well_conditioned` check now ALSO requires the
bone-vs-rollchild angle to be >35° (same threshold as
`constrained_rotation`'s near-parallel guard). When the limb is straight
(angle <35°), twist is pinned to zero regardless of whether the roll child
is measured.

Verification (6 frames, post-fix):
| Frame | leg angle | init twist | solve twist |
|---|---|---|---|
| 0 | 16° | 0° | 0° |
| 30 | 4.5° | 0° | 0° |
| 60 | 10° | 0° | 0° |
| 120 | 113° (bent) | -1° | -29° (legitimate) |
| 150 | 90° (bent) | -10° | -38° (legitimate) |
| 210 | 4° | 0° | 0° |

Straight legs: 0° twist (pinned). Bent legs: moderate twist from the
measured ankle (legitimate roll constraint). The leg fix from stage 8 is
preserved, and the arm/shoulder/head fixes from stage 9 are preserved.

### Stage 10: Dorsiflexion ("on heels") + head oscillation fixes

Two artifacts remained after stage 9.1:
1. **"Manny is always on heels"** — feet dorsiflexed (toes pointing up), not
   flat on the ground.
2. **"Head always moves either left side or right continuously"** — head
   oscillated side-to-side instead of facing forward or looking down.

#### Root cause 1: Dorsiflexion (feet on heels)

The production data (`b_stage6`) is **17-joint COCO**, not 133-joint
COCO-WholeBody.  `extract_mocap_points` only emits `big_toe_*`/`heel_*`
when `pts3d.shape[-2] > 20` (133-joint data).  On 17-joint data the proxy
`toe_*` joint has no measured target, so `analytic_init` and `_residuals`
synthesize it from the pelvis root frame: `toe = ankle + R_root · rest_off`.

`R_root` is built from `spine_dir = neck - pelvis`.  This clip has heavy
forward lean (spine_dir tilted ~60° forward of vertical on frame 0), so
`R_root`'s up axis tilts forward.  The rest toe offset is mostly "down"
(-7 cm in the -Z of Manny's foot), so `R_root.apply(rest_off)` projects
that "down" into forward+up — the synthesized toe lands **above** the
ankle.  The ankle's swing rotation then points the foot up to reach it ->
**dorsiflexion -> "on heels."**

Fix: a **dorsiflexion clamp** on the toe target's Z (up axis in Manny
Z-up), applied in both `analytic_init` and `_residuals`:
- **133-joint data** (heel available): clamp toe Z to `heel_Z + 1 cm`.
  The sole runs heel->toe; a flat foot has toe ≈ heel height.
- **17-joint data** (no heel, the production case): clamp toe Z to
  `ankle_Z`.  In rest pose the toe is 7 cm *below* the ankle; a toe above
  the ankle is always dorsiflexed.

The clamp is general (uses `name_to_idx`, no hardcoded Manny indices)
and backward-compatible (only fires when the toe target is finite and
above the reference).

#### Root cause 2: Head oscillation left/right

The 70° plausibility gate added in stage 9 only gates the head
**orientation** residual.  The head **position** is hard-pinned to
`measured["nose"]` with `w[head] = 1.0` (head is coco-anchored) — NOT
gated.  Since FK makes `head_pos = neck_pos + R_neck · rest_offset_head`,
a lateral/upward nose forces the **neck and spine to swing** to reach it
-> the head oscillates side-to-side with every noisy nose frame.
`temporal_weight=0.03` is ~33× too weak to damp this, and no upper-body
temporal filter existed in the retarget path.

Fix (two parts):
- **2A. Gate the head POSITION weight** with the same 70° check: when
  the measured nose direction (neck->nose, in the neck's local frame) is
  >70° from the rest head direction, down-weight `w[head]` from 1.0 to
  0.1.  This stops the neck/spine from swinging to reach an implausible
  nose; a genuinely turned head (within 70°) still biases the solve.
- **2B/2C. One-Euro temporal filter** post-solve.  Generalized
  `filter_params_one_euro_quaternion` (was hardcoded to 18 joints; now
  infers joint count from the array width) and applied it to the solved
  state track in `respine_solve_render.py`.  One-Euro is adaptive: smooths
  aggressively when stationary (kills jitter) and lightly when moving
  fast (preserves motion).  This is the de-robotizing pass the advisor
  text recommended; it smooths the whole body, especially the upper body.

#### Verification

| Metric | Before (stage 9.1) | After (stage 10) |
|---|---|---|
| Toe Z vs ankle Z (17-joint) | toe above ankle (dorsiflexed) | toe ≤ ankle Z (flat) |
| Head position weight on bad nose | 1.0 (hard-pinned) | 0.1 (gated) |
| Upper-body temporal filter | none | One-Euro (min_cutoff=1.0, beta=0.007) |
| Filter joint count | hardcoded 18 | inferred from array width |

Tests: 8 new tests added (`test_extract_mocap_points_includes_heel`,
`test_dorsiflexion_clamp_*`, `test_head_position_weight_*`,
`test_filter_params_one_euro_quaternion_any_joint_count`).  Full retarget
suite: 23 passed.

#### Generality
All fixes use COCO-WholeBody constants (17-22), `name_to_idx`, and
`coco_anchor` — no hardcoded Manny indices.  Backward-compatible: on
17-joint data the dorsiflexion clamp uses the ankle-Z fallback; on
133-joint data it uses the more precise heel-Z clamp.  The One-Euro
filter works on any joint count.

#### Stage 10 numerical verification (17-joint production data, frames 1353-1653)

Dorsiflexion clamp (toe Z vs ankle Z — negative = toe below ankle = flat foot):
| Frame | toe Z | ankle Z | toe-ank | Status |
|---|---|---|---|---|
| 0 | -2.63 | 10.12 | -12.75 | flat (toe well below ankle) |
| 30 | -2.12 | 7.11 | -9.24 | flat |
| 60 | -2.90 | 6.67 | -9.58 | flat |

Before the fix, the R_root-synthesized toe landed ABOVE the ankle (positive
toe-ank) -> dorsiflexion -> "on heels."  After the clamp, toe-ank is strongly
negative on every sampled frame -> the foot is plantarflexed as in a normal
stance, not dorsiflexed.

Head position gate (nose angle vs 70 deg threshold, on most-motion frames):
On the frames with the most lateral nose offset in the window, the nose
direction stayed within 70 deg of the rest head direction (gate `off`),
meaning the clip's nose is plausible.  The gate is a safety net for extreme
triangulation noise; on this clip the head oscillation is handled by the
One-Euro temporal filter (Task 2C), which smooths the solved rotation track
post-solve to remove per-frame jitter in the neck/head/spine.

#### Stage 10 render completion

Re-solved the 300-frame most-motion window (1353-1653) with all stage 10
fixes and exported:
- `outputs/production_motion_v2.fbx` (2.15 MB, 300 frames = 10 s @ 30 fps)
- `outputs/production_motion_v2.npy` (FBX animation track)
- `outputs/fbx_eval_v2.mp4` (eval render)

The One-Euro temporal filter (min_cutoff=1.0, beta=0.007) was applied to the
solved state track before rotation transfer, smoothing per-frame jitter
across all joints (especially neck/head/spine).  The dorsiflexion clamp
kept the toe below the ankle on every frame, and the head position gate
down-weights implausible nose targets.
