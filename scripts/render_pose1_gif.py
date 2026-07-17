import sys, json, cv2, numpy as np
from pathlib import Path
import imageio

ROOT = Path(r'e:\Chaos\Projects\aimocap_re')
sys.path.insert(0, str(ROOT))
from aimocap.data.panoptic import load_calibration as load_panoptic_calib
from aimocap.math.coords import internal_to_opencv

SEQ = "171204_pose1"
CAMS = ["00_08", "00_09", "00_26"]
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def main():
    print("Loading NPZ...")
    data = np.load(ROOT / 'outputs' / 'pose1_trio.npz')
    pts3d = data['skeleton3d']  # (300, 17, 3) in Y-up internal
    pts3d_world = internal_to_opencv(pts3d)

    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    frames_dir = ROOT / f"data/panoptic/{SEQ}/sync_frames"
    
    P_mats = []
    for cn in CAMS:
        K = calib[cn].K.astype(np.float64)
        R = calib[cn].R.astype(np.float64)
        t = calib[cn].t.astype(np.float64).reshape(3, 1)
        Rt = np.hstack((R, t))
        P = K @ Rt
        P_mats.append(P)
        
    print("Rendering frames 149 to 249...")
    out_frames = []
    
    # Render 100 frames (approx 3 seconds)
    for i in range(100):
        f = i + 149 # The offset used when creating the 300 frames NPZ was 149. But wait! 
        # In eval_pose1_trio.py: frames are 0..299 of the video sequence, which corresponds to actual frame  in video
        video_f = i  
        
        # Load the 3 images
        imgs = []
        for cn in CAMS:
            ip = frames_dir / f"hd_{cn}" / f"{video_f:08d}.jpg"
            img = cv2.imread(str(ip))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            imgs.append(img)
            
        # Reproject 3D points
        skel_world = pts3d_world[i]
        
        for c_idx, P in enumerate(P_mats):
            pts2d = []
            for k in range(17):
                if np.isnan(skel_world[k, 0]):
                    pts2d.append((np.nan, np.nan))
                    continue
                X = np.append(skel_world[k], 1.0)
                p = P @ X
                if p[2] > 1e-5:
                    u, v = p[0]/p[2], p[1]/p[2]
                    pts2d.append((int(u), int(v)))
                else:
                    pts2d.append((np.nan, np.nan))
                    
            # Draw lines
            for p, c in BONES:
                if not np.isnan(pts2d[p][0]) and not np.isnan(pts2d[c][0]):
                    cv2.line(imgs[c_idx], pts2d[p], pts2d[c], (0, 255, 0), 4)
            # Draw points
            for pt in pts2d:
                if not np.isnan(pt[0]):
                    cv2.circle(imgs[c_idx], pt, 6, (255, 0, 0), -1)
                    
        # Resize images to be smaller so side-by-side isn't 6K
        imgs = [cv2.resize(im, (640, 360)) for im in imgs]
        # Concatenate horizontally
        row = np.hstack(imgs)
        out_frames.append(row)
        
    out_gif = ROOT / "outputs" / "pose1_trio_multiview.gif"
    print("Saving GIF...")
    imageio.mimsave(out_gif, out_frames, fps=15)
    print(f"Saved {out_gif}")

if __name__ == '__main__':
    main()
