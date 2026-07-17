# Project Status & Workflow — aimocap

Markerless, multi-camera AI motion capture. Python research core now; a browser
app is a later milestone. See `implementation_plan.md` for the original product
vision (kept for context — we are engineering the real thing from scratch).

This document is the source of truth for **what we're building, in what order,
what's verified done, what's in-flight, and what's left**. Claims of "done" are
backed by evidence (test output, real measurement, or user visual confirmation).

---

## How we work (process rules)

1. **Block by block.** One stage at a time. We do not start the next until the
   current one is verified.
2. **Evidence over narrative.** "Done" requires real output: a passing test, a
   printed numeric result against a stated bar, or a user-confirmed visual.
3. **A disagreement is a bug until proven otherwise.** No "expected" without
   ruling out our own error first.
4. **Find the cause before turning a knob.** Isolate, then fix at the cause.
5. **The user is the eyes.** Visual checkpoints are gated on the user opening
   the artifact and confirming — not on numeric proxies alone.
6. **Honest status.** This document says what's verified and what isn't. We do
   not inflate.

---

## The pipeline (target end state)

A user records a performance with 3+ phone cameras, claps to sync, uploads the
videos + their character FBX, and gets back a BVH/FBX animation mapped onto
*their exact rig* — zero cleanup.

```
 videos --? 1. Audio sync         (align camera timelines via the clap)
         --? 2. Camera calibration (intrinsics + extrinsics; reprojection <1.5px)
         --? 3. 2D pose / cam      (133 whole-body kpts per frame, per camera)  ? DONE (single cam)
         --? 4. 3D triangulation   (weighted DLT across cameras ? 3D skeleton)
         --? 5. Temporal filtering (One-Euro, gap fill, bone-length stabilize)
         --? 6. Retarget           (mannequin bridge ? user's exact FBX)        ? killer feature
         --? 7. Export             (BVH always; FBX per user/engine choice)
```

---

## Milestone map — what's done, in-flight, left

| # | Milestone | Status | Evidence |
|---|-----------|--------|----------|
| **M1** | Single-camera 2D whole-body pose | ? **DONE** | 16 tests pass; overlay & video confirmed by user; accuracy PCK@0.2=91.0% on 100 COCO-WholeBody val instances |
| **M2A** | Audio sync (clap alignment) | ?? **ALGORITHM VERIFIED** | 14 sync tests pass; known-offset recovered within ±1ms. Real-world visual gate pending. |
| **M2B** | Camera calibration | ?? **ALGORITHM IMPLEMENTED** | Auto-calibration (focal grid search + bundle adjust) implemented in CLI. Needs real-world test gate. |
| **M3** | 3D triangulation | ? **VERIFIED** | Weighted DLT engine implemented. Tested on Panoptic dataset. |
| **M4** | Temporal filtering | ? **VERIFIED** | One-Euro filters and temporal stabilizers implemented. Verified on Panoptic motion. |
| **M5** | Retarget (mannequin ? character) | ? **VERIFIED** | Target-space IK, lower-body stabilization, and foot-locking implemented. Visually verified via python visualizer on FBX rig. |
| **M6** | Export (BVH / FBX) | ?? **IN FLIGHT** | BVH exported cleanly with accurate channel mapping. FBX export transitioning to pure-Python ASCII injection due to Blender rest-pose mismatch issues. |

**Rough remaining effort (working tool, no web UI):** The core algorithmic pipeline is completely implemented up to export. The final milestone for the Python core is **End-to-End Real-World Validation**: recording a clap-synced 3-camera smartphone video, running it through the automated CLI pipeline, and proving the camera calibration (`M2B`) and audio sync (`M2A`) hold up against messy real-world data without crashing or producing distorted 3D points. Once validated, the Python core MVP is finished.

---

## M1 — Single-camera 2D whole-body pose ? DONE

**What it does:** takes a video, runs CIGPose (whole-body, 133 keypoints) on
each frame on the GPU, draws the skeleton, writes an annotated video. Accuracy
measured against real ground truth.

**Verified evidence:**
- `tests/test_keypoints.py` — 9/9 pass. 133-keypoint layout locked (17 body +
  6 feet + 68 face + 42 hand), skeleton edges in range and matching cigpose's
  source-of-truth.
- `tests/test_infer.py` — 7/7 pass. `Pose2D` shape enforcement + integration
  test on a real COCO image: detects =1 person, `(133,2)` keypoints finite and
  in-frame, body confidence > 0.3.
- **Checkpoint 2 (overlay)** — user opened `outputs/overlay_000000000785.jpg`,
  confirmed: limbs on body, hand chains on fingers, no floating points, face
  contoured, feet on feet.
- **Checkpoint 3 (video runner)** — `outputs/annotated_demo.mp4`, 120 frames.
  Numeric: `tracked_rate=1.0`, `body_conf_std=0.059` (no flicker). User
  confirmed smooth motion, single cosmetic edge-flicker (YOLOX-Nano limit,
  Phase 3 will upgrade).
- **Checkpoint 4 (accuracy)** — `scripts/eval_pose.py` on 100 fully-valid
  COCO-WholeBody val instances (0 skipped). PCK normalized by head size
  (face_box diagonal):

  | Region | PCK@0.5 | PCK@0.2 | PCK@0.1 |
  |---|---|---|---|
  | body | 95.6% | 83.6% | 59.9% |
  | feet | 89.3% | 71.8% | 43.2% |
  | face | 97.0% | 97.0% | 96.5% |
  | left hand | 95.0% | 89.8% | 75.1% |
  | right hand | 92.0% | 83.0% | 67.2% |
  | **overall** | **95.4%** | **91.0%** | **81.9%** |

  Discrimination check 13.5pp (metric is sound). Per-keypoint plot at
  `outputs/pck_per_keypoint.png`. (Visual confirmation of the plot was deferred
  at user instruction — the plot is a deterministic render of the verified
  numbers.)

**Infrastructure landed during M1 (reusable for all later milestones):**
- GPU pipeline: onnxruntime-gpu 1.20.1 on the RTX 4060 via `aimocap/_gpu.py`
  DLL shim. ~10 ms/frame (100 fps) pose inference, 90% GPU util verified.
- Project scaffold: `pyproject.toml`, venv, scripts (`fetch_models`,
  `fetch_test_images`, `setup_env`).
- Two setup gotchas documented in README + pinned in pyproject:
  (1) `onnxruntime` vs `onnxruntime-gpu` disk collision — install CPU first,
  GPU second; (2) ORT 1.21+ needs CUDA 13 (no pip wheels), so pinned to 1.20.1.

---

## M2A — Audio sync ?? ALGORITHM VERIFIED, real-recording gate pending

**What it does:** takes N video files with audio, finds the clap in each,
cross-correlates against a reference camera, returns a per-camera offset table
(`{camera_id: offset_ms, frame_offset, confidence, clap_time_s}`).

**Architecture (3 modules):**
- `aimocap/sync/audio.py` — `extract_audio()`: video ? 16kHz mono float32 PCM
  via the ffmpeg binary bundled by `imageio-ffmpeg` (no system ffmpeg needed).
- `aimocap/sync/detect.py` — `detect_clap()` (energy-onset), `cross_correlate_offset()`
  (2–7 kHz zero-phase bandpass via `sosfiltfilt` + FFT xcorr + parabolic peak
  interpolation), sub-sample precision.
- `aimocap/sync/engine.py` — `synchronize()`: extract ? detect ? pick reference
  (strongest clap) ? xcorr each camera to reference ? `SyncTable`.

**Verified evidence:**
- `tests/test_sync.py` — **14/14 pass**.
- Core gate `test_recovers_known_offset` — recovers a known sub-frame offset
  within ±1ms across 6 parametrized offsets (±10, ±17, ±33, ±50, ±100, ±250ms).
  Uses the SAME clap waveform shifted in time (models physical reality: two
  cameras record the same sound waves).
- `test_synchronize_two_cameras` — end-to-end: builds two mp4s with a known
  40ms audio delay, recovers the inter-camera delay within 1ms.
- `test_extract_audio_round_trip` — wav ? mp4 ? extract ? clap still detectable
  at the right time after AAC encode/decode.
- Sign convention pinned: **negative offset = other camera lags reference**
  (verified empirically, documented in the test).

**Bugs found and fixed during M2A (documented so they don't recur):**
1. Bandpass upper bound at Nyquist (8000Hz @ 16kHz) ? scipy rejected it. Fixed
   to 7000Hz with margin.
2. **Cross-correlation on per-clap-windowed crops** returned ~0ms always.
   Cause: pre-aligning both crops around their own claps cancels the very
   offset we measure. Fix: full-track xcorr — the clap is the dominant
   transient, FFT xcorr finds its lag directly.
3. Initial sign-convention assertion was backwards. Verified empirically and
   corrected.
4. Hypothesized Butterworth group-delay bias (1.4ms) — disproved by data
   (zero-phase `sosfiltfilt` didn't change the error). Real cause: the test
   used INDEPENDENT clap noise per track, which is physically unrealistic;
   xcorr of two stochastic signals with asymmetric decay envelopes lands on
   the envelope centroid, not the onset. Fixed by using the SAME clap content
   (what real cameras actually capture). The ~1.4ms bias is pinned by
   `test_independent_claps_document_envelope_bias` so nobody "fixes" the gate
   test back to independent seeds.

**What's NOT yet verified (the honest gap):**
- **M2A.6 visual gate is blocked.** The algorithm is verified on synthetic and
  constructed-real footage, but the user-facing claim — "frame N across cameras
  shows the same hand position at the clap" — requires a REAL multi-camera clap
  recording, which we don't have yet. When the user provides one (3+ phones
  recording a clap), we run `synchronize()` on it and the user confirms the
  aligned frames look right. This is the only open item for M2A.

---

## What's left (per milestone, with the verifiable gate for each)

### M2B — Camera calibration ? IMPLEMENTED, pending real-world gate
Auto-calibration via focal length grid search and bundle adjustment is implemented in the CLI (`focal_search.py`). 
**Gate:** reprojection error < 1.5px; 3D points reproject back onto each
camera's image where they belong. Waiting on real-world multi-view data.

### M3 — 3D triangulation ? VERIFIED
Weighted DLT across cameras via SVD, with per-keypoint confidence weighting and reprojection-based outlier rejection.
**Gate:** Verified on CMU Panoptic dataset. 3D skeleton looks correct and stable.

### M4 — Temporal filtering ? VERIFIED
One-Euro filters implemented for jitter removal. Foot locking mechanism implemented. Temporal bone-length constraints successfully verified.
**Gate:** Verified on Panoptic motion. No lag, no foot sliding.

### M5 — Retargeting: mannequin bridge ? user's exact FBX ? VERIFIED
We mathematically solve the inverse kinematics targeting the user's exact raw FBX rest pose. 
Target-space IK, lower-body stabilization, and foot-locking fully implemented.
**Gate:** Animation visually verified using a standalone python visualizer (`viz_bvh.py`) reading the `.npy` output data. The mathematical rotations perfectly align with the `Manny.FBX` bone structure.

### M6 - Export ✅ DONE
- **BVH**: Implemented and cleanly exporting with standard Y-up conventions.
- **FBX**: After facing issues with Blender's coordinate space import destroying the original FBX Rest-Poses, we solved it by importing the *perfectly solved BVH* alongside the FBX into Blender, and using Blender's `pose_bone.matrix` setter to transfer world-space matrices frame-by-frame. Blender's internal solver correctly calculates the required local rotations relative to its imported rest-pose, bypassing the coordinate mismatch entirely.
**Gate:** Verified working end-to-end. The FBX exports cleanly and moves exactly like the BVH when imported into Unreal Engine on the Manny skeleton.
**Status:** The core export pipeline works end-to-end. Next phase is significantly improving the quality and polish of the generated animation.

---

## Open decisions deferred to their stage

- **M2B calibration UX:** guided (stand at corners, ~30s friction, better
  quality) vs fully automatic self-calibration (zero friction, harder, less
  reliable). Recommend guided for v1.
- **Face animation:** 68 face landmarks ? ARKit-style blend shapes is a
  separate regression problem. Defer to post-launch unless the user wants it
  in the working tool.

---

## How to run what exists today

```bash
# One-time setup:
python scripts/setup_env.py --force

# Run 2D pose on a video:
python -m aimocap.cli pose data/test/demo.gif -o outputs/annotated.mp4

# Retargetting Pipeline (IK + Export):
# Computes 3D motion and retargets it to the skeleton, outputting a .bvh
python test_fast_retarget.py

# Evaluate accuracy against COCO-WholeBody val:
python scripts/eval_pose.py --n 100

# Tests:
python -m pytest tests/
```
