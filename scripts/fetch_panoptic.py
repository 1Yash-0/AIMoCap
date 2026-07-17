"""Fetch a CMU Panoptic Studio sequence for calibration / 3D-math verification.

Panoptic is the gold-standard multi-view dataset: HD video from 30+ views, full
camera calibration (intrinsics + extrinsics), and 3D body-pose ground truth from
a marker system. We use it to verify camera calibration (M2B) and 3D
triangulation (M3) against known-good data — there is no other way to measure
reprojection error meaningfully without ground-truth camera matrices.

Default fetch is ONE sequence (171204_pose1 — single person, full calibration)
with:
  - calibration_<seq>.json     (~0.2 MB)   per-camera K, R, t, distortion
  - hdPose3d_stage1_coco19.tar (~40 MB)    3D body-pose GT (coco19 = 19 keypoints)
  - N HD videos                (~2.8 GB each)  hd_00_<node>.mp4, nodes 0..30

Disk usage warning: each HD video is ~2.8 GB. Default is 3 HD views (~8.6 GB).
For fast iteration use --vga (VGA videos are ~30 MB each, lower resolution).

The 3D body GT uses coco19 format (the standard COCO-17 + 2 extra: neck + mid-
hip), NOT the COCO-WholeBody 133 layout our pose model outputs. The loader
(aimocap/calibration/panoptic.py) handles this mapping when comparing.

URL pattern verified against panoptic-toolbox/scripts/getData.sh and confirmed
via HTTP HEAD on all assets (June 2026).

Usage:
    python scripts/fetch_panoptic.py                     # default: 3 HD views
    python scripts/fetch_panoptic.py --hd-views 5
    python scripts/fetch_panoptic.py --vga --vga-views 10
    python scripts/fetch_panoptic.py --sequence 160224_haggling1
    python scripts/fetch_panoptic.py --force             # re-download even if present
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "panoptic"

ENDPOINT = "http://domedb.perception.cs.cmu.edu"
DEFAULT_SEQUENCE = "171204_pose1"


def _url(seq: str, *parts: str) -> str:
    return f"{ENDPOINT}/webdata/dataset/{seq}/" + "/".join(parts)


def _human_size(n: float) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024:
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} PB"


def _download(url: str, dst: Path, force: bool = False, min_bytes: int = 1024) -> bool:
    """Stream-download url -> dst with progress. Returns True on success.

    Downloads to a `.part` temp file and atomically renames on completion, so a
    crash mid-download leaves no half-written file (a retry starts fresh).
    """
    if dst.exists() and not force and dst.stat().st_size >= min_bytes:
        print(f"  exists: {dst.name} ({_human_size(dst.stat().st_size)})")
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "aimocap"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            tmp = dst.with_suffix(dst.suffix + ".part")
            downloaded = 0
            t0 = time.time()
            last_report = 0.0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB blocks (not 8KB — 100x faster)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_report > 2.0:
                        if total:
                            pct = 100 * downloaded / total
                            rate = downloaded / max(now - t0, 1e-3) / (1 << 20)
                            print(f"    {pct:5.1f}%  {_human_size(downloaded)}/"
                                  f"{_human_size(total)}  ({rate:.1f} MB/s)",
                                  flush=True)
                        else:
                            print(f"    {_human_size(downloaded)}", flush=True)
                        last_report = now
            tmp.replace(dst)  # atomic on same filesystem
        print(f"  saved: {dst.name} ({_human_size(dst.stat().st_size)})")
        return True
    except Exception as e:
        tmp = dst.with_suffix(dst.suffix + ".part")
        if tmp.exists():
            tmp.unlink()  # clean up partial so retry starts fresh
        print(f"  FAIL {url}: {e}", file=sys.stderr)
        return False


def _check_disk(needed_gb: float) -> None:
    """Abort early if free disk is less than 1.2x what we need (20% margin)."""
    free_bytes, _, _ = shutil.disk_usage(str(DATA_DIR.anchor))
    free_gb = free_bytes / 1024**3
    print(f"  disk free: {free_gb:.1f} GB, estimated need: {needed_gb:.1f} GB")
    if free_gb < needed_gb * 1.2:
        print(f"  ERROR: insufficient disk space (need 20% margin). Aborting.",
              file=sys.stderr)
        sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sequence", default=DEFAULT_SEQUENCE,
                    help=f"Panoptic sequence name (default: {DEFAULT_SEQUENCE})")
    ap.add_argument("--hd-views", type=int, default=3,
                    help="Number of HD videos to fetch (default 3; max 31). "
                         "Each is ~2.8 GB. Nodes 0..N-1.")
    ap.add_argument("--vga", action="store_true",
                    help="Fetch VGA videos instead of HD (much smaller, lower res).")
    ap.add_argument("--vga-views", type=int, default=10,
                    help="Number of VGA videos if --vga (default 10; max 480).")
    ap.add_argument("--no-pose-gt", action="store_true",
                    help="Skip the 3D body-pose ground-truth tar.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if files already exist.")
    args = ap.parse_args()

    seq = args.sequence
    seq_dir = DATA_DIR / seq
    seq_dir.mkdir(parents=True, exist_ok=True)
    print(f"Panoptic sequence: {seq}")
    print(f"Output dir: {seq_dir}\n")

    # Disk budget estimate for the warning.
    if args.vga:
        per_view_gb = 0.03
        n_views = args.vga_views
    else:
        per_view_gb = 2.8
        n_views = args.hd_views
    needed_gb = per_view_gb * n_views + 0.1  # calibration + pose gt
    _check_disk(needed_gb)

    ok_all = True

    # 1. Calibration JSON (always — required for everything downstream).
    print("\n[1/3] Calibration")
    calib_dst = seq_dir / f"calibration_{seq}.json"
    ok_all &= _download(_url(seq, f"calibration_{seq}.json"), calib_dst, args.force)

    # 2. Videos.
    print(f"\n[2/3] Videos ({'VGA' if args.vga else 'HD'}, {n_views} views)")
    vid_dir = seq_dir / ("vgaVideos" if args.vga else "hdVideos")
    vid_dir.mkdir(exist_ok=True)
    if args.vga:
        # VGA naming: vga_<panel>_<node>.mp4 across panels 1..20, nodes 1..24.
        # We fetch the first N by simple enumeration — sufficient for math
        # verification; caller wanting specific views can edit this loop.
        count = 0
        for panel in range(1, 21):
            for node in range(1, 25):
                if count >= n_views:
                    break
                fname = f"vga_{panel:02d}_{node:02d}.mp4"
                url = _url(seq, "videos", "vga_shared_crf10", fname)
                if _download(url, vid_dir / fname, args.force,
                             min_bytes=5_000_000):
                    count += 1
            if count >= n_views:
                break
        ok_all &= count > 0
    else:
        # HD naming: hd_00_<node>.mp4, node 0..30 (31 views max).
        if args.hd_views > 31:
            print(f"  WARNING: hd-views {args.hd_views} > 31, clamping",
                  file=sys.stderr)
            args.hd_views = 31
        for node in range(args.hd_views):
            fname = f"hd_00_{node:02d}.mp4"
            url = _url(seq, "videos", "hd_shared_crf20", fname)
            ok_all &= _download(url, vid_dir / fname, args.force,
                                min_bytes=10_000_000)

    # 3. 3D body-pose GT (coco19 format).
    if not args.no_pose_gt:
        print("\n[3/3] 3D body-pose ground truth (coco19)")
        pose_tar = seq_dir / "hdPose3d_stage1_coco19.tar"
        if _download(_url(seq, "hdPose3d_stage1_coco19.tar"), pose_tar, args.force):
            print(f"  extracting {pose_tar.name}...")
            try:
                with tarfile.open(pose_tar) as tf:
                    tf.extractall(seq_dir)
                print(f"  extracted into {seq_dir}")
            except Exception as e:
                print(f"  extract FAIL: {e}", file=sys.stderr)
                ok_all = False
        else:
            ok_all = False

    print("\n" + ("=" * 60))
    if ok_all:
        print(f"Done. Data at: {seq_dir}")
        for p in sorted(seq_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(seq_dir)
                print(f"  {rel}  ({_human_size(p.stat().st_size)})")
    else:
        print("Completed with errors. See messages above.", file=sys.stderr)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
