import sys, cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import imageio

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))
from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = "171204_pose1"
CAMS = ["00_11", "00_12", "00_23"]
FRAMES_NUM = 1800
OFFSET = 150
SUBSAMPLE = 3 # 30fps -> 10fps (600 frames total)
W, H = 480, 270 # Per camera resolution

BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def opencv_to_internal(pts3d: np.ndarray) -> np.ndarray:
    out = np.copy(pts3d)
    out[..., 1] = -out[..., 1]
    out[..., 2] = -out[..., 2]
    return out

def internal_to_opencv(pts3d: np.ndarray) -> np.ndarray:
    out = np.copy(pts3d)
    out[..., 1] = -out[..., 1]
    out[..., 2] = -out[..., 2]
    return out

def main():
    print("Loading 3D Triangulation CSV...")
    df_d = pd.read_csv(ROOT / "outputs" / "diag_pose1_full" / "triangulation_3d_log.csv")
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    
    vid_dir = ROOT / f"data/panoptic/{SEQ}/hdVideos"
    caps = [cv2.VideoCapture(str(vid_dir / f"hd_{cn}.mp4")) for cn in CAMS]
    
    extrinsics = []
    K_list = []
    for cn in CAMS:
        extrinsics.append((calib[cn].R.astype(np.float64), calib[cn].t.astype(np.float64).reshape(3, 1)))
        K_list.append(calib[cn].K.astype(np.float64))
        
    frames_gif = []
    print("Rendering frames...")
    
    for fi in tqdm(range(FRAMES_NUM)):
        frames_cams = []
        for c in range(len(CAMS)):
            ret, fr = caps[c].read()
            if not ret: fr = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frames_cams.append(fr)
            
        if fi % SUBSAMPLE != 0:
            continue
            
        global_frame = fi + OFFSET
        df_f = df_d[df_d['frame'] == global_frame].sort_values('joint')
        if len(df_f) == 17:
            pts3d = np.zeros((17, 3), dtype=np.float64)
            for j in range(17):
                row = df_f.iloc[j]
                pts3d[j] = [row['clean_x'], row['clean_y'], row['clean_z']]
            
            # Convert internal Y-up to OpenCV Y-down
            pts3d_cv = internal_to_opencv(pts3d)
            
            for c in range(len(CAMS)):
                fr = frames_cams[c]
                R, t = extrinsics[c]
                K = K_list[c]
                
                # Project
                proj, _ = cv2.projectPoints(pts3d_cv, R, t, K, None)
                proj = proj.reshape(-1, 2)
                
                for j in range(17):
                    if not np.isnan(pts3d_cv[j][0]):
                        x, y = int(proj[j][0]), int(proj[j][1])
                        cv2.circle(fr, (x, y), 8, (0, 0, 255), -1)
                
                for j1, j2 in BONES:
                    if not np.isnan(pts3d_cv[j1][0]) and not np.isnan(pts3d_cv[j2][0]):
                        p1 = (int(proj[j1][0]), int(proj[j1][1]))
                        p2 = (int(proj[j2][0]), int(proj[j2][1]))
                        cv2.line(fr, p1, p2, (0, 0, 255), 4)
                        
        # Resize and concat
        resized_cams = []
        for fr in frames_cams:
            rs = cv2.resize(fr, (W, H))
            # Convert BGR to RGB
            rs = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
            resized_cams.append(rs)
            
        canvas = np.hstack(resized_cams)
        frames_gif.append(canvas)
        
    for cap in caps: cap.release()
    
    out_path = ROOT / "outputs" / "pose1_full_multiview.gif"
    print(f"Saving GIF to {out_path} ...")
    imageio.mimsave(str(out_path), frames_gif, fps=10)
    print("Done!")

if __name__ == '__main__':
    main()
