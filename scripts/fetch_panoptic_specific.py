import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fetch_panoptic import _download, _url, _check_disk

def main():
    seq = "171204_pose1"
    seq_dir = ROOT / "data" / "panoptic" / seq
    vid_dir = seq_dir / "hdVideos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    
    # 3 HD videos = ~8.4 GB
    _check_disk(8.5)
    
    nodes = [26, 29, 30]
    for node in nodes:
        fname = f"hd_00_{node:02d}.mp4"
        url = _url(seq, "videos", "hd_shared_crf20", fname)
        print(f"Downloading {fname}...")
        _download(url, vid_dir / fname, False, min_bytes=10_000_000)

if __name__ == "__main__":
    main()
