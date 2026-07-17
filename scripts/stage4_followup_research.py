import json
import numpy as np
from pathlib import Path

def get_cam_pos(R, t):
    # R * pos + t = 0 => pos = -R.T * t
    return -R.T @ t

def get_angle(pos1, pos2, center):
    v1 = pos1 - center
    v2 = pos2 - center
    v1_n = v1 / np.linalg.norm(v1)
    v2_n = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_n.T, v2_n), -1.0, 1.0)))[0,0]

def main():
    calib_json = Path('data/panoptic/171204_pose1/calibration_171204_pose1.json')
    with open(calib_json) as f:
        data = json.load(f)
    
    cams = {c['name']: c for c in data['cameras'] if c['type'] == 'hd'}
    
    # Just approximate center as [0, 0, 0] for now or use GT
    gt_json = Path('data/panoptic/171204_pose1/hdPose3d_stage1_coco19/body3DScene_00000149.json')
    with open(gt_json) as f:
        gt_data = json.load(f)
    joints19 = np.array(gt_data['bodies'][0]['joints19']).reshape(19, 4)
    # hip center
    center = (joints19[6, :3] + joints19[12, :3]) / 2.0
    center = center.reshape(3, 1)
    
    # Compute positions for all HD cameras
    positions = {}
    for n, cam in cams.items():
        R = np.array(cam['R'])
        t = np.array(cam['t'])
        positions[n] = get_cam_pos(R, t)
    
    print(f"Center pos: {center.T}")
    
    print("Angles between original cameras:")
    print(f"00_00 vs 00_01: {get_angle(positions['00_00'], positions['00_01'], center):.1f} deg")
    print(f"00_01 vs 00_02: {get_angle(positions['00_01'], positions['00_02'], center):.1f} deg")
    print(f"00_02 vs 00_00: {get_angle(positions['00_02'], positions['00_00'], center):.1f} deg")
    
    # Find 3 well-spread cameras
    import itertools
    hd_names = sorted(list(positions.keys()))
    
    best_cams = None
    best_score = float('inf')
    
    for c1, c2, c3 in itertools.combinations(hd_names, 3):
        a1 = get_angle(positions[c1], positions[c2], center)
        a2 = get_angle(positions[c2], positions[c3], center)
        a3 = get_angle(positions[c3], positions[c1], center)
        
        # We want all angles to be as close to 120 as possible
        score = abs(a1 - 120) + abs(a2 - 120) + abs(a3 - 120)
        
        if score < best_score:
            best_score = score
            best_cams = (c1, c2, c3)
            
    print(f"Best 3 cameras for 120 deg spread: {best_cams} (score: {best_score:.1f})")
    a1 = get_angle(positions[best_cams[0]], positions[best_cams[1]], center)
    a2 = get_angle(positions[best_cams[1]], positions[best_cams[2]], center)
    a3 = get_angle(positions[best_cams[2]], positions[best_cams[0]], center)
    print(f"Angles: {a1:.1f}, {a2:.1f}, {a3:.1f}")

if __name__ == '__main__':
    main()
