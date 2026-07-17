"""Step 0: Stage 4.2 pass/fail check -- MPJPE on same joint set as 52.9mm baseline."""
import numpy as np, json
from pathlib import Path

ROOT  = Path(".")
pts   = np.load("outputs/stage4_2_knee_rescue/pts3d_clean.npy")
gt    = np.load("outputs/stage4_2_knee_rescue/gt_kpts.npy")
valid = np.load("outputs/stage4_2_knee_rescue/gt_valid.npy")

gap_raw = json.loads((ROOT / "outputs/stage4_2_knee_rescue/gap_log.json").read_text())
recon = np.zeros(len(valid), dtype=bool)
for rec in gap_raw["gap_log"]:
    if rec.get("reconstructed"):
        recon[rec["start_frame"]: rec["end_frame"] + 1] = True

# Exclude knees (13,14) AND ankles (15,16) -- IDENTICAL joint set as Stage 5.1 52.9mm baseline
EXCL = [13, 14, 15, 16]

frame_errs = []
frame_is_recon = []

for f in range(len(valid)):
    if not valid[f]:
        continue
    p = pts[f].copy()
    g = gt[f].copy()
    for ji in EXCL:
        p[ji] = np.nan
        g[ji] = np.nan
    vj = np.isfinite(p).all(1) & np.isfinite(g).all(1)
    if vj.sum() < 3:
        continue
    rp = (p[11] + p[12]) / 2.0
    rg = (g[11] + g[12]) / 2.0
    diff = np.where(vj[:, None], (p - rp) - (g - rg), np.nan)
    frame_errs.append(float(np.nanmean(np.linalg.norm(diff, axis=1)) * 10.0))
    frame_is_recon.append(bool(recon[f]))

frame_errs = np.array(frame_errs)
frame_is_recon = np.array(frame_is_recon)

overall   = float(np.nanmean(frame_errs))
meas_only = float(np.nanmean(frame_errs[~frame_is_recon])) if (~frame_is_recon).any() else float("nan")
THRESHOLD = 52.9 * 1.05   # 55.46mm

print("STEP 0 -- RC-MPJPE on same joint set as Stage 5.1 52.9mm baseline")
print("  (knees + ankles excluded -- identical exam)")
print(f"  Measured (non-reconstructed) frames only: {meas_only:.2f}mm")
print(f"  Overall incl. recon frames (for reference): {overall:.2f}mm")
print(f"  Threshold (52.9 x 1.05):                   {THRESHOLD:.2f}mm")
verdict = "PASS -- proceed to Stage 6a" if meas_only <= THRESHOLD else "FAIL -- stop"
print(f"  VERDICT: {verdict}")
