"""Download a small set of photorealistic test images for development.

These are real COCO val2017 images — clean full-body shots we use to verify the
2D pose pipeline end-to-end. Kept tiny on purpose; real benchmarking happens
against CMU Panoptic data (see fetch_panoptic.py).

Usage:
    python scripts/fetch_test_images.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST = PROJECT_ROOT / "data" / "test"

# COCO image IDs known to contain clear, standing, full-body people — good for
# whole-body (hands + face) keypoint checks.
IMAGES = {
    "000000000785.jpg": "http://images.cocodataset.org/val2017/000000000785.jpg",
    "000000000802.jpg": "http://images.cocodataset.org/val2017/000000000802.jpg",
    "000000332836.jpg": "http://images.cocodataset.org/val2017/000000332836.jpg",
    "000000355269.jpg": "http://images.cocodataset.org/val2017/000000355269.jpg",
}


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name, url in IMAGES.items():
        dst = DEST / name
        if dst.exists() and dst.stat().st_size > 5000:
            print(f"  exists: {dst}")
            ok += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aimocap"})
            data = urllib.request.urlopen(req, timeout=30).read()
            dst.write_bytes(data)
            print(f"  saved {len(data)} bytes -> {dst}")
            ok += 1
        except Exception as e:
            print(f"  FAIL {url}: {e}", file=sys.stderr)
    print(f"\n{ok}/{len(IMAGES)} images available in {DEST}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
