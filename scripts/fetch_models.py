"""Download CIGPose ONNX models from the cigpose-onnx GitHub release.

The release ships one 1.6 GB zip containing all 14 model variants. We don't
need all of them — this script streams the zip from disk (downloaded once) and
extracts only the files we actually use, controlled by a small registry below.

Usage:
    python scripts/fetch_models.py                  # download default set
    python scripts/fetch_models.py --all            # extract everything
    python scripts/fetch_models.py --force          # re-download even if present
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# --- Configuration ----------------------------------------------------------

RELEASE_URL = (
    "https://github.com/namas191297/cigpose-onnx/releases/download/"
    "v1.0.0/cigpose_models.zip"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ZIP_CACHE = MODELS_DIR / "cigpose_models.zip"

# Files we want by default. CIGPose-L COCO-WholeBody is our workhorse; YOLOX-Nano
# is the person detector the top-down pipeline needs.
DEFAULT_FILES = {
    "cigpose-l_coco-wholebody_384x288.onnx",
    "cigpose-x_coco-wholebody_384x288.onnx",
    "cigpose-x_coco-ubody_384x288.onnx",
    "yolox_nano.onnx",
}

# --- Helpers ----------------------------------------------------------------


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_with_progress(url: str, dest: Path) -> None:
    """Stream a URL to disk with a simple progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "aimocap"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        print(f"Downloading {url}")
        print(f"  -> {dest}  ({human_size(total) if total else 'size unknown'})")

        downloaded = 0
        last_pct = -1
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        print(
                            f"  {pct:3d}%  ({human_size(downloaded)}/"
                            f"{human_size(total)})",
                            flush=True,
                        )
                        last_pct = pct
        print(f"  done ({human_size(downloaded)})")


def extract_files(zip_path: Path, names: set[str], dest_dir: Path) -> list[Path]:
    """Extract named entries from a zip, flattening any directory prefix."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    remaining = set(names)

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name in remaining:
                # Read just this entry and write it out flat.
                with zf.open(info) as src:
                    data = src.read()
                out = dest_dir / name
                out.write_bytes(data)
                extracted.append(out)
                remaining.discard(name)
                print(f"  extracted {name} ({human_size(len(data))})")

    if remaining:
        print(f"  WARNING: not found in zip: {sorted(remaining)}", file=sys.stderr)
    return extracted


# --- Main -------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all",
        action="store_true",
        help="Extract all models from the zip, not just the default set.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download the zip even if a cached copy exists.",
    )
    ap.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the 1.6 GB cached zip after extraction (default: delete it).",
    )
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ensure the zip is on disk.
    if args.force or not ZIP_CACHE.exists():
        download_with_progress(RELEASE_URL, ZIP_CACHE)
    else:
        print(f"Using cached zip: {ZIP_CACHE}")

    # 2. List the contents so we know what's available.
    print("\nZip contents:")
    with zipfile.ZipFile(ZIP_CACHE) as zf:
        names = [Path(i.filename).name for i in zf.infolist() if not i.is_dir()]
        for n in sorted(names):
            print(f"  {n}")
    print(f"  ({len(names)} files)")

    # 3. Extract.
    targets = None if args.all else DEFAULT_FILES
    print("\nExtracting " + ("all models" if args.all else f"default set: {sorted(DEFAULT_FILES)}"))
    if targets is None:
        with zipfile.ZipFile(ZIP_CACHE) as zf:
            zf.extractall(MODELS_DIR)
        print(f"  extracted everything into {MODELS_DIR}")
    else:
        extract_files(ZIP_CACHE, targets, MODELS_DIR)

    # 4. Optionally reclaim the 1.6 GB zip cache.
    if not args.keep_zip and ZIP_CACHE.exists():
        ZIP_CACHE.unlink()
        print(f"\nRemoved zip cache to reclaim space ({ZIP_CACHE.name}).")
        print("  Pass --keep-zip to retain it for extracting more models later.")

    print("\nDone.")
    print(f"Models directory: {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
