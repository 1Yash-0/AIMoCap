import av
import sys

cams = ['00_00', '00_01', '00_02', '00_03', '00_04', '00_28', '00_24']
valid_cams = []

for cn in cams:
    path = f"data/panoptic/171204_pose3/hdVideos/hd_{cn}.mp4"
    try:
        container = av.open(path)
    except Exception:
        continue
        
    stream = container.streams.video[0]
    
    first_pts = None
    last_pts = None
    count = 0
    
    for frame in container.decode(video=0):
        if first_pts is None:
            first_pts = frame.pts
        last_pts = frame.pts
        count += 1
        
    duration = stream.duration * float(stream.time_base) if stream.duration else 0.0
    
    print(f"--- Camera {cn} ---")
    print(f"  Container start_time: {container.start_time}")
    print(f"  Stream start_time: {stream.start_time}")
    print(f"  First decoded PTS: {first_pts}")
    print(f"  Last decoded PTS: {last_pts}")
    print(f"  Time base: {stream.time_base}")
    print(f"  Nominal FPS (average_rate): {stream.average_rate}")
    print(f"  Average FPS (calculated): {count / duration if duration > 0 else 'N/A':.3f}")
    print(f"  Frame count: {count}")
    print(f"  Duration (s): {duration:.3f}")
    print(f"  Zero-based timestamps: {first_pts == 0}")
    print("")
