import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(r'e:\Chaos\Projects\aimocap_re')
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')

def get_ar(cam):
    ip = ROOT / f'data/panoptic/171204_pose3/sync_frames/hd_{cam}/00000333.jpg'
    if not ip.exists():
        print(f'{cam} missing image')
        return
    res = model(str(ip), verbose=False)
    kpts = res[0].keypoints.data[0].cpu().numpy()  # (17, 3)
    k_valid = kpts[kpts[:, 2] > 0.5]
    if len(k_valid) > 0:
        w = np.max(k_valid[:, 0]) - np.min(k_valid[:, 0])
        h = np.max(k_valid[:, 1]) - np.min(k_valid[:, 1])
        ar = h / w if w > 0 else 0
        print(f'AR {cam}: {ar:.3f}')
    else:
        print(f'AR {cam}: No valid kpts')

get_ar('00_03')
get_ar('00_04')
