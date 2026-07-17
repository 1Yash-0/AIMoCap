"""Evaluate 2D whole-body pose accuracy against COCO-WholeBody val ground truth.

Runs the PoseEstimator on a sample of fully-validated COCO-WholeBody val
annotations (those with reliable face/hand/foot labels), matches each prediction
to its GT person by bbox IoU, and computes PCK (Percentage of Correct Keypoints)
per region. PCK here is normalized by HEAD SIZE (face_box diagonal) — the
PCKh-style convention — so the threshold is local to the person and small enough
to actually discriminate keypoint quality on hands/face. (An earlier version
normalized by the full body bbox diagonal at alpha=0.5, giving a bogus 100%:
the threshold was ~200px, i.e. "is the keypoint anywhere on the person".)

Reports PCK at multiple alpha thresholds (0.5, 0.2, 0.1 of head size) so the
discrimination curve is visible — a meaningful metric must produce different
numbers at different thresholds.

Outputs:
    - printed per-region + overall PCK at each alpha
    - outputs/pck_per_keypoint.png  (bar chart at alpha=0.2, color-coded by region)
    - outputs/pck_summary.json      (raw numbers for later comparison)

Usage:
    python scripts/eval_pose.py                     # default sample of 100
    python scripts/eval_pose.py --n 307             # all fully-valid anns
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import aimocap  # noqa: F401  (ensures CUDA DLLs load first)
from aimocap.pose.infer import PoseEstimator
from aimocap.pose.keypoints import (
    KEYPOINT_NAMES_133, REGIONS_133, BODY_17, FEET_6,
    FACE_68, LEFT_HAND_21, RIGHT_HAND_21,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANN_FILE = PROJECT_ROOT / "data" / "coco_wholebody" / "coco_wholebody_val_v1.0.json"
IMG_DIR = PROJECT_ROOT / "data" / "coco_wholebody"
OUT_DIR = PROJECT_ROOT / "outputs"

VAL_BASE_URL = "http://images.cocodataset.org/val2017/"

# PCK alpha thresholds, as fraction of head size. 0.5 is lenient, 0.1 is strict.
DEFAULT_ALPHAS = (0.5, 0.2, 0.1)


def gt_to_133(ann: dict) -> np.ndarray:
    """Concatenate the 5 sub-fields into a single (133,3) [x,y,v] array.

    COCO-WholeBody order (verified against the data_format spec):
        body(17) | foot(6) | face(68) | lefthand(21) | righthand(21)
    """
    flat = (
        ann["keypoints"]          # 51
        + ann["foot_kpts"]        # 18
        + ann["face_kpts"]        # 204
        + ann["lefthand_kpts"]    # 63
        + ann["righthand_kpts"]   # 63
    )
    arr = np.array(flat, dtype=np.float32).reshape(-1, 3)
    assert arr.shape == (133, 3), f"unexpected GT shape {arr.shape}"
    return arr


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    if isinstance(b, (list, tuple, np.ndarray)) and len(b) == 4 and b[2] <= b[0] + 1e-6:
        # b is xywh
        bx1, by1, bw, bh = b
        bx2, by2 = bx1 + bw, by1 + bh
    else:
        bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1e-6)


def select_fully_valid(anns: list[dict], min_body: int = 15) -> list[dict]:
    """Annotations with reliable labels across all whole-body regions."""
    return [
        a for a in anns
        if a.get("num_keypoints", 0) >= min_body
        and a.get("face_valid")
        and a.get("lefthand_valid")
        and a.get("righthand_valid")
        and a.get("foot_valid")
    ]


def ensure_image(im_meta: dict) -> Path:
    dst = IMG_DIR / im_meta["file_name"]
    if dst.exists() and dst.stat().st_size > 5000:
        return dst
    url = VAL_BASE_URL + im_meta["file_name"]
    req = urllib.request.Request(url, headers={"User-Agent": "aimocap"})
    data = urllib.request.urlopen(req, timeout=30).read()
    dst.write_bytes(data)
    return dst


def pck_per_keypoint(
    preds: np.ndarray,        # (N, 133, 2)
    gts: np.ndarray,          # (N, 133, 2)
    vis: np.ndarray,          # (N, 133) bool
    norm: np.ndarray,         # (N,) per-instance normalizer (e.g. bbox diag)
    alpha: float,
):
    """Return (per_keypoint_pck, overall_pck). Only counts visible GT kpts."""
    dists = np.linalg.norm(preds - gts, axis=2)         # (N, 133)
    thresh = norm[:, None] * alpha                      # (N, 1)
    correct = (dists <= thresh) & vis                   # (N, 133)
    # per-keypoint: average over instances where that keypoint is visible
    per_kpt = np.zeros(133)
    for k in range(133):
        mask = vis[:, k]
        if mask.sum() > 0:
            per_kpt[k] = correct[mask, k].mean()
    overall = correct[vis].mean() if vis.any() else 0.0
    return per_kpt, overall


def head_size(ann: dict) -> float:
    """Head-size normalizer = diagonal of the GT face_box (xywh).

    Returns 0.0 if face_box is missing/zero (will be filtered out upstream
    because we require face_valid, but guard anyway).
    """
    fb = ann.get("face_box")
    if not fb or all(v == 0 for v in fb):
        return 0.0
    _, _, w, h = fb
    return float(np.hypot(w, h))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100,
                    help="Number of fully-valid annotations to evaluate (default 100).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not ANN_FILE.exists():
        print(f"annotations missing: {ANN_FILE}", file=sys.stderr)
        print("run: python scripts/fetch_coco_wholebody.py", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading {ANN_FILE.name} ...")
    with open(ANN_FILE) as f:
        data = json.load(f)
    imgs = {im["id"]: im for im in data["images"]}
    valid = select_fully_valid(data["annotations"])
    print(f"fully-valid annotations: {len(valid)}")

    rng = np.random.default_rng(args.seed)
    if args.n < len(valid):
        idx = rng.choice(len(valid), size=args.n, replace=False)
        sample = [valid[i] for i in idx]
    else:
        sample = valid
    print(f"evaluating {len(sample)} annotations at PCK alphas={DEFAULT_ALPHAS} "
          f"(normalized by head size)\n")

    est = PoseEstimator()
    print(f"providers: {est.active_providers}")

    preds, gts, vis, norms = [], [], [], []
    skipped = 0
    t0 = time.perf_counter()
    for i, ann in enumerate(sample):
        im_meta = imgs[ann["image_id"]]
        try:
            img_path = ensure_image(im_meta)
        except Exception as e:
            print(f"  [{i}] image download failed: {e}", file=sys.stderr)
            skipped += 1
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            skipped += 1
            continue

        poses = est.estimate(frame, pick="all")
        if not poses:
            skipped += 1
            continue
        # GT bbox is xywh
        gx, gy, gw, gh = ann["bbox"]
        gt_bbox_xyxy = (gx, gy, gx + gw, gy + gh)
        best = max(poses, key=lambda p: bbox_iou(gt_bbox_xyxy, p.bbox))
        if bbox_iou(gt_bbox_xyxy, best.bbox) < 0.3:
            skipped += 1
            continue

        hsz = head_size(ann)
        if hsz < 1.0:
            # can't normalize; skip rather than produce a bogus number
            skipped += 1
            continue

        gt133 = gt_to_133(ann)
        preds.append(best.keypoints)
        gts.append(gt133[:, :2])
        vis.append(gt133[:, 2] > 0)
        norms.append(hsz)

        if (i + 1) % 10 == 0:
            dt = time.perf_counter() - t0
            print(f"  [{i+1}/{len(sample)}] {dt:.1f}s elapsed, {skipped} skipped", flush=True)

    if not preds:
        print("no valid predictions collected", file=sys.stderr)
        return 1

    preds = np.array(preds)
    gts = np.array(gts)
    vis = np.array(vis)
    norms = np.array(norms)
    print(f"\ncomputed on {len(preds)} instances ({skipped} skipped); "
          f"mean head size {norms.mean():.1f}px")

    # Compute PCK at each alpha.
    results = {}
    for alpha in DEFAULT_ALPHAS:
        per_kpt, overall = pck_per_keypoint(preds, gts, vis, norms, alpha)
        results[alpha] = {"per_kpt": per_kpt, "overall": overall}
        print(f"\n=== PCK@{alpha} (normalized by head size = face_box diagonal) ===")
        print(f"  OVERALL: {overall*100:.1f}%")
        for name, region in REGIONS_133.items():
            mask = vis[:, region.start:region.end]
            region_pck = per_kpt[region.start:region.end]
            n_vis = int(mask.sum())
            weights = mask.sum(axis=0)
            region_overall = ((region_pck * weights).sum() / weights.sum()
                              if weights.sum() > 0 else 0.0)
            print(f"  {name:11s}: {region_overall*100:5.1f}%  (n_visible={n_vis})")

    # Sanity gate: a meaningful metric must discriminate across alphas.
    # If the strictest (0.1) and lenient (0.5) PCK are within 3%, the normalizer
    # is too coarse — flag it rather than silently report a useless number.
    delta = results[DEFAULT_ALPHAS[0]]["overall"] - results[DEFAULT_ALPHAS[-1]]["overall"]
    print(f"\n=== discrimination check ===")
    print(f"  PCK@{DEFAULT_ALPHAS[0]} - PCK@{DEFAULT_ALPHAS[-1]} = {delta*100:.1f}pp "
          f"(want > 3pp; less means the metric is too coarse to discriminate)")

    # Per-keypoint bar chart at alpha=0.2 (the middle, most informative threshold).
    plot_alpha = 0.2
    per_kpt = results[plot_alpha]["per_kpt"]
    overall = results[plot_alpha]["overall"]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 5))
        region_colors = {"body": "tab:green", "feet": "tab:cyan",
                         "face": "tab:orange", "left_hand": "tab:blue",
                         "right_hand": "tab:pink"}
        colors = []
        for k in range(133):
            for name, region in REGIONS_133.items():
                if region.start <= k < region.end:
                    colors.append(region_colors[name])
                    break
        ax.bar(range(133), per_kpt * 100, color=colors)
        ax.axhline(overall * 100, color="k", linestyle="--", linewidth=1,
                   label=f"overall {overall*100:.1f}%")
        ax.set_xlabel("keypoint index")
        ax.set_ylabel(f"PCK@{plot_alpha} (%)")
        ax.set_title("Per-keypoint PCK vs COCO-WholeBody val GT "
                     f"(alpha={plot_alpha} head size; color = region)")
        ax.set_xlim(-1, 133)
        ax.set_ylim(0, 105)
        for region in REGIONS_133.values():
            ax.axvline(region.start - 0.5, color="gray", linewidth=0.5, alpha=0.5)
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=n) for n, c in region_colors.items()]
        handles += [plt.Line2D([], [], color="k", ls="--",
                               label=f"overall {overall*100:.1f}%")]
        ax.legend(handles=handles, loc="lower right", fontsize=8)
        plot_path = OUT_DIR / "pck_per_keypoint.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"\nsaved plot: {plot_path}")
    except Exception as e:
        print(f"plot failed: {e}", file=sys.stderr)

    # Raw numbers
    summary_path = OUT_DIR / "pck_summary.json"
    summary = {
        "n_instances": len(preds),
        "mean_head_size_px": float(norms.mean()),
        "normalizer": "face_box diagonal (head size)",
        "per_alpha": {
            str(a): {
                "overall": float(results[a]["overall"]),
                "per_region": {
                    name: float((results[a]["per_kpt"][r.start:r.end]
                                 * vis[:, r.start:r.end].sum(axis=0)).sum()
                                / max(vis[:, r.start:r.end].sum(), 1))
                    for name, r in REGIONS_133.items()
                },
                "per_keypoint_pck": results[a]["per_kpt"].tolist(),
            }
            for a in DEFAULT_ALPHAS
        },
        "keypoint_names": list(KEYPOINT_NAMES_133),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
