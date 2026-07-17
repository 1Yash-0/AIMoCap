"""Stage 4C: Deterministic Synchronization Fix and Decoder Regression"""

import sys
import hashlib
import json
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEQ = "171204_pose3"
CAMS = ["00_03", "00_04", "00_28", "00_24"]
TARGET_FRAMES = 450
SAVE_START = 150
SAVE_END = 449

def get_hash(frame):
    return hashlib.md5(frame.tobytes()).hexdigest()

def decode_pass(cn, pass_idx):
    url = ROOT / f"data/panoptic/{SEQ}/hdVideos/hd_{cn}.mp4"
    if not url.exists():
        print(f"Error: {url} does not exist.")
        sys.exit(1)
        
    cap = cv2.VideoCapture(str(url))
    frames_data = []
    
    count = 0
    while count < TARGET_FRAMES:
        pts = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        if not ret: break
        
        frames_data.append({
            "idx": count,
            "pts": pts,
            "hash": get_hash(frame),
            "frame": frame if pass_idx == 1 else None # Only save actual arrays on Pass 1
        })
        count += 1
    cap.release()
    return frames_data

def main():
    print(f"Stage 4C Decoder Regression Test for {SEQ}\\n")
    
    out_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for cn in CAMS:
        print(f"Testing camera {cn}...")
        
        print("  Running Pass 1 (Sequential decode)...")
        pass1 = decode_pass(cn, 1)
        
        print("  Running Pass 2 (Sequential decode)...")
        pass2 = decode_pass(cn, 2)
        
        # 1. Assert exactly 450 frames
        if len(pass1) != TARGET_FRAMES or len(pass2) != TARGET_FRAMES:
            print(f"  FAILED: Extracted {len(pass1)} frames, expected {TARGET_FRAMES}")
            sys.exit(1)
            
        # 2. Hash match between passes
        mismatches = sum(1 for p1, p2 in zip(pass1, pass2) if p1["hash"] != p2["hash"])
        if mismatches > 0:
            print(f"  FAILED: {mismatches} hash mismatches between sequential decodes.")
            sys.exit(1)
            
        # 3. Duplicate frame check
        duplicates = sum(1 for i in range(1, len(pass1)) if pass1[i]["hash"] == pass1[i-1]["hash"])
        if duplicates > 0:
            print(f"  FAILED: {duplicates} duplicate frames detected!")
            sys.exit(1)
            
        # 4. Strictly increasing PTS
        pts_violations = sum(1 for i in range(1, len(pass1)) if pass1[i]["pts"] <= pass1[i-1]["pts"] and pass1[i]["pts"] != 0.0)
        if pts_violations > 0:
            print(f"  FAILED: {pts_violations} frames have non-increasing PTS.")
            sys.exit(1)
            
        print("  -> Decoder Regression PASSED!")
        print(f"     First PTS: {pass1[0]['pts']:.1f} ms | Last PTS: {pass1[-1]['pts']:.1f} ms")
        
        # 5. Save Authoritative Frames & Metadata
        cam_dir = out_dir / f"hd_{cn}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        print("  Saving frames 150-449 and metadata...")
        for data in pass1:
            idx = data["idx"]
            if SAVE_START <= idx <= SAVE_END:
                # Save image
                img_path = cam_dir / f"{idx:08d}.jpg"
                cv2.imwrite(str(img_path), data["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                # Save metadata contract
                meta_path = cam_dir / f"{idx:08d}.json"
                meta = {
                    "sequence_id": SEQ,
                    "camera_id": cn,
                    "global_panoptic_frame_id": idx,
                    "video_frame_index": idx,
                    "pts_ms": data["pts"],
                    "timestamp_s": data["pts"] / 1000.0,
                    "gt_frame_id": idx,
                    "extraction_method": "sequential_deterministic"
                }
                with open(meta_path, "w") as fp:
                    json.dump(meta, fp, indent=2)
                    
        print(f"  Finished {cn}.\\n")
        
    print("ALL CAMERAS PASSED STAGE 4C DECODER REGRESSION.")

if __name__ == "__main__":
    main()
