import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aimocap.data.panoptic import load_calibration as load_panoptic_calib

SEQ = "171204_pose3"
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def main():
    gt_dir = ROOT / f"data/panoptic/{SEQ}/hdPose3d_stage1_coco19"
    kpts = []
    for fi in range(150, 450):
        fpath = gt_dir / f"body3DScene_{fi:08d}.json"
        if fpath.exists():
            with open(fpath) as fp: d = json.load(fp)
            if d.get("bodies"):
                k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)
                kpts.append(k[COCO17_TO_PAN19, :3])
    kpts = np.array(kpts)
    # median pelvis
    hips = (kpts[:, 11, :] + kpts[:, 12, :]) / 2.0
    median_pelvis = np.nanmedian(hips, axis=0)
    print(f"Median Pelvis [X, Y, Z]: {median_pelvis}")
    
    calib = load_panoptic_calib(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    hd_cams = {c.name: c for c in calib.values() if c.name.startswith("00_")}
    
    print("\\nSubject-Centered Camera Azimuths:")
    cam_angles = {}
    for cn, c in hd_cams.items():
        C = -c.R.astype(np.float64).T @ c.t.reshape(3, 1).astype(np.float64)
        v = C.flatten() - median_pelvis
        az = np.arctan2(v[1], v[0]) * 180 / np.pi
        el = np.arctan2(v[2], np.linalg.norm(v[0:2])) * 180 / np.pi
        dist = np.linalg.norm(v)
        cam_angles[cn] = (az, el, dist, C.flatten())
    
    for cn, (az, el, dist, C) in sorted(cam_angles.items(), key=lambda x: x[1][0]):
        print(f"  {cn}: Az={az:6.1f}°, El={el:4.1f}°, Dist={dist:6.1f}cm")
        
    azs = [a[0] for a in cam_angles.values()]
    print(f"\\nMin Azimuth: {min(azs):.1f}°")
    print(f"Max Azimuth: {max(azs):.1f}°")
    print(f"Arc spread: {max(azs) - min(azs):.1f}°")

if __name__ == "__main__":
    main()
