"""Fetch EXACTLY 450 frames of new candidate cameras for Stage 6a.8"""

import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fetch_panoptic import _url

def stream_and_save(cam_id: str, max_frames: int = 450):
    url = _url("171204_pose1", "videos", "hd_shared_crf20", f"hd_{cam_id}.mp4")
    print(f"Opening stream for {cam_id}: {url}")
    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Failed to open stream for {cam_id}")
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps == 0 or w == 0:
        fps, w, h = 29.97, 1920, 1080 # Fallbacks
        
    out_dir = ROOT / "data" / "panoptic" / "171204_pose1" / "hdVideos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hd_{cam_id}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    count = 0
    print(f"Saving exactly {max_frames} frames to {out_path}...")
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Stream ended early at frame {count}.")
            break
        writer.write(frame)
        count += 1
        if count % 100 == 0:
            print(f"  {count}/{max_frames} frames...")
            
    cap.release()
    writer.release()
    print(f"Finished {cam_id}.\\n")
    return True

if __name__ == "__main__":
    for cam in ["00_13", "00_21", "00_28"]:
        stream_and_save(cam)
