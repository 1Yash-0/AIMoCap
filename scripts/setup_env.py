"""Set up the aimocap dev environment with a working CUDA-enabled ONNX Runtime.

Why this script exists (the gotcha it works around):

    `cigpose-onnx` depends on plain `onnxruntime` (CPU-only). `onnxruntime-gpu`
    and `onnxruntime` both install into the same `onnxruntime/` directory and
    ship their own copies of `onnxruntime_pybind11_state.pyd` and
    `onnxruntime.dll`. Whichever installs LAST overwrites the other. The CPU
    `.pyd` is compiled without CUDA support, so if it lands last the CUDA
    Execution Provider silently disappears.

    Fix: install them in explicit order — `onnxruntime` first, then
    `onnxruntime-gpu` — so the GPU `.pyd`/`.dll` win, and the CUDA provider DLL
    is the one on disk. A single `pip install onnxruntime onnxruntime-gpu` does
    NOT guarantee order.

Usage:
    python scripts/setup_env.py            # fresh venv, full install, verify CUDA
    python scripts/setup_env.py --verify   # just check an existing env
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / "venv"


def _py() -> str:
    if sys.platform == "win32":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def _pip(*args: str) -> None:
    print(f"\n$ python -m pip {' '.join(args)}")
    subprocess.check_call([_py(), "-m", "pip", *args])


def create_venv(force: bool) -> None:
    if VENV_DIR.exists():
        if not force:
            print(f"venv exists at {VENV_DIR} (use --force to recreate)")
            return
        shutil.rmtree(VENV_DIR)
    print(f"Creating venv at {VENV_DIR}")
    venv.create(VENV_DIR, with_pip=True, clear=True)


def install_deps() -> None:
    _pip("install", "--upgrade", "pip")

    # 1. Everything except onnxruntime, from pyproject.toml editable install.
    #    cigpose-onnx will pull in plain onnxruntime here.
    _pip("install", "-e", ".[dev]")

    # 2. Re-install the two ORT wheels in the CORRECT ORDER so the GPU build's
    #    pyd/dll win. --no-deps because deps are already satisfied and we don't
    #    want pip re-resolving onnxruntime to the CPU wheel.
    _pip("install", "--no-deps", "--force-reinstall", "onnxruntime==1.20.1")
    _pip("install", "--no-deps", "--force-reinstall", "onnxruntime-gpu==1.20.1")


def verify_cuda() -> bool:
    """Return True if CUDAExecutionProvider is active on a model session."""
    model = PROJECT_ROOT / "models" / "cigpose-l_coco-wholebody_384x288.onnx"
    if not model.exists():
        print(f"\n[verify] model not found at {model}; skipping CUDA test.")
        print("         Run scripts/fetch_models.py first.")
        return False

    check = (
        "import aimocap\n"
        "import onnxruntime as ort\n"
        "print('ORT version:', ort.__version__)\n"
        "print('Available providers:', ort.get_available_providers())\n"
        f"sess = ort.InferenceSession(r'{model}',\n"
        "    providers=['CUDAExecutionProvider','CPUExecutionProvider'])\n"
        "active = sess.get_providers()\n"
        "print('Session active providers:', active)\n"
        "assert 'CUDAExecutionProvider' in active, 'CUDA EP NOT active'\n"
        "import numpy as np, time\n"
        "x = np.random.randn(1,3,384,288).astype(np.float32)\n"
        "for _ in range(3): sess.run(None, {'input': x})\n"
        "t0=time.perf_counter()\n"
        "for _ in range(20): sess.run(None, {'input': x})\n"
        "dt=(time.perf_counter()-t0)/20\n"
        "print(f'GPU inference: {dt*1000:.1f} ms/frame ({1/dt:.0f} fps)')\n"
        "print('CUDA_OK')\n"
    )
    print("\n=== Verifying CUDA ===")
    r = subprocess.run([_py(), "-c", check], capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace("\x00", "")
    print(out)
    return r.returncode == 0 and "CUDA_OK" in out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="Recreate the venv even if it exists.")
    ap.add_argument("--verify", action="store_true",
                    help="Only verify an existing env; skip install.")
    args = ap.parse_args()

    if not args.verify:
        create_venv(args.force)
        install_deps()

    ok = verify_cuda()
    if ok:
        print("\n✅ Environment ready: CUDA active on the GPU.")
        return 0
    print("\n❌ CUDA not active. See messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
