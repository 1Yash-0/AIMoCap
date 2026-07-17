"""Fetch COCO-WholeBody validation annotations for accuracy evaluation.

The annotations are hosted on Google Drive / OneDrive / BaiduPan (not a direct
URL). We use gdown for the Google Drive file. The val set annotations contain
133-keypoint GT (body+feet+face+hands) for COCO val2017 person instances.

Images are NOT downloaded here (several GB); eval_pose.py fetches individual
val2017 images on demand by ID.

Usage:
    python scripts/fetch_coco_wholebody.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST = PROJECT_ROOT / "data" / "coco_wholebody"
ANN_FILE = DEST / "coco_wholebody_val_v1.0.json"

# Google Drive file ID for the official COCO-WholeBody Validation annotations
# (from the jin-s13/COCO-WholeBody README).
GDRIVE_FILE_ID = "1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb"


def human_size(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.1f} {u}"
        x /= 1024
    return f"{x:.1f} TB"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    if ANN_FILE.exists() and ANN_FILE.stat().st_size > 1_000_000:
        print(f"exists: {ANN_FILE} ({human_size(ANN_FILE.stat().st_size)})")
        return 0

    try:
        import gdown
    except ImportError:
        print("gdown is required: pip install gdown", file=sys.stderr)
        return 1

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"downloading from Google Drive (id={GDRIVE_FILE_ID}) -> {ANN_FILE}")
    try:
        gdown.download(url, str(ANN_FILE), quiet=False)
    except Exception as e:
        print(f"gdown failed: {e}", file=sys.stderr)
        return 1

    if not ANN_FILE.exists() or ANN_FILE.stat().st_size < 1_000_000:
        print(f"downloaded file looks invalid: {ANN_FILE}", file=sys.stderr)
        return 1

    print(f"\nsaved: {ANN_FILE} ({human_size(ANN_FILE.stat().st_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
