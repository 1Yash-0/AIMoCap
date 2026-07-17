import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)

def main():
    cams = ["00_26", "00_29", "00_30"]
    cams_str = " ".join(cams)
    
    print("=== Running Stage 2 (Detection) on Spread Cams ===")
    run_cmd([
        sys.executable, "scripts/stage2_detection_audit.py",
        "--cams"
    ] + cams + [
        "--outdir", "outputs/stage2_check_c"
    ])
    
    print("\n=== Running Stage 3 (Pose) on Spread Cams ===")
    run_cmd([
        sys.executable, "scripts/stage3_pose_audit.py",
        "--cams"
    ] + cams + [
        "--outdir", "outputs/stage3_check_c"
    ])
    
    print("\n=== Running Stage 4 (Triangulation) on Spread Cams ===")
    run_cmd([
        sys.executable, "scripts/stage4_triangulation_audit.py",
        "--cams"
    ] + cams + [
        "--npz", "outputs/stage3_check_c/kpts.npz",
        "--outdir", "outputs/stage4_check_c",
        "--min-conf", "0.5"  # Using 0.5 because Check D proved it's better
    ])
    
    print("\n=== Check C Complete ===")

if __name__ == "__main__":
    main()
