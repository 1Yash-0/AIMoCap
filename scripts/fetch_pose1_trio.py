import os
import subprocess
from pathlib import Path

def main():
    root = Path(r"E:\Chaos\Projects\aimocap_re")
    cams = ["00_11", "00_12", "00_23"]
    seq = "171204_pose1"
    
    out_dir = root / f"data/panoptic/{seq}/hdVideos"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1 minute at 30fps is 1800 frames
    max_frames = 1800
    
    for cam in cams:
        url = f"http://domedb.perception.cs.cmu.edu/webdata/dataset/{seq}/videos/hd_shared_crf20/hd_{cam}.mp4"
        out_path = out_dir / f"hd_{cam}.mp4"
        
        print(f"\n[{cam}] Fetching {max_frames} frames from {url}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", url,
            "-vframes", str(max_frames),
            "-c:v", "copy",
            "-c:a", "copy",
            str(out_path)
        ]
        subprocess.run(cmd, check=True)
        
    print("\n[Done] Fetched 60 seconds (1800 frames) for cameras:", cams)

if __name__ == '__main__':
    main()
