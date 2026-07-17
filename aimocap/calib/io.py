"""Input/output for camera calibration data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_calibration(
    path: str | Path,
    K_list: list[np.ndarray],
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    image_sizes: list[tuple[int, int]] | None = None,
) -> None:
    """
    Save N cameras to a JSON file.
    
    Args:
        path: Output path.
        K_list: N (3,3) intrinsic matrices.
        extrinsics: N tuples of (3,3) Rotation, (3,1) translation.
        image_sizes: N tuples of (width, height).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    cameras = []
    for i in range(len(K_list)):
        cam_data = {
            "id": i,
            "K": K_list[i].tolist(),
            "R": extrinsics[i][0].tolist(),
            "t": extrinsics[i][1].flatten().tolist()
        }
        if image_sizes is not None and i < len(image_sizes):
            cam_data["image_size"] = list(image_sizes[i])
            
        cameras.append(cam_data)
        
    data = {
        "coordinate_system": "y_up_right_handed",
        "cameras": cameras
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(
    path: str | Path,
    camera_names: list[str] | None = None,
    scale_factor: float = 1.0
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Load calibration data from JSON.
    
    Args:
        path: Path to the calibration JSON file.
        camera_names: Optional list of camera names to load in that exact order.
                      If provided, matches the camera's JSON name against the elements.
        scale_factor: Multiplier to apply to the translation vector (t). 
                      If Panoptic is in cm, use scale_factor=0.01 to enforce meters.
    
    Returns:
        K_list: N (3,3) intrinsic matrices.
        extrinsics: N tuples of (3,3) Rotation, (3,1) translation.
    """
    with open(path, "r") as f:
        data = json.load(f)
        
    if data.get("coordinate_system") != "y_up_right_handed":
        import warnings
        warnings.warn("Calibration JSON does not declare 'y_up_right_handed'. Ensure math is consistent.")
        
    K_list = []
    extrinsics = []
    
    if camera_names is None:
        for cam in data["cameras"]:
            K = np.array(cam["K"], dtype=np.float64)
            R = np.array(cam["R"], dtype=np.float64)
            t = np.array(cam["t"], dtype=np.float64).reshape((3, 1)) * scale_factor
            
            K_list.append(K)
            extrinsics.append((R, t))
    else:
        cam_dict = {cam.get("name", cam.get("node", str(cam.get("id")))): cam for cam in data["cameras"]}
        for name in camera_names:
            best_match = None
            # e.g., name might be "hd_00_01.mp4" or "Cam_1"
            import os
            base_name = os.path.basename(name).split('.')[0] # "hd_00_01"
            
            # Exact match first
            if name in cam_dict:
                best_match = cam_dict[name]
            elif base_name in cam_dict:
                best_match = cam_dict[base_name]
            else:
                # Try to extract trailing digits from base_name, e.g. "hd_00_01" -> "1"
                import re
                m = re.search(r'(\d+)$', base_name)
                if m:
                    idx_str = str(int(m.group(1))) # "01" -> "1"
                    if idx_str in cam_dict:
                        best_match = cam_dict[idx_str]
            
            if not best_match:
                # Fallback fuzzy match
                for cam_name, cam in cam_dict.items():
                    if cam_name in name or name in cam_name:
                        best_match = cam
                        break
            
            if not best_match:
                raise ValueError(f"Could not find calibration for camera matching '{name}' in {path}")
                
            K = np.array(best_match["K"], dtype=np.float64)
            R = np.array(best_match["R"], dtype=np.float64)
            t = np.array(best_match["t"], dtype=np.float64).reshape((3, 1)) * scale_factor
            
            K_list.append(K)
            extrinsics.append((R, t))
            
    return K_list, extrinsics

def load_cameras(
    path: str | Path,
    camera_names: list[str] | None = None
) -> list["aimocap.motion.camera.CameraModel"]:
    """
    Load calibration data from JSON into CameraModel objects.
    Strict loader: requires exact name matching.
    """
    import json
    import numpy as np
    from aimocap.motion.camera import CameraModel
    
    with open(path, "r") as f:
        data = json.load(f)
        
    if data.get("coordinate_system") != "y_up_right_handed":
        import warnings
        warnings.warn("Calibration JSON does not declare 'y_up_right_handed'. Ensure math is consistent.")
        
    cameras = []
    
    if camera_names is None:
        for i, cam in enumerate(data.get("cameras", [])):
            K = np.array(cam["K"], dtype=np.float64)
            R = np.array(cam["R"], dtype=np.float64) if "R" in cam else np.eye(3)
            t = np.array(cam["t"], dtype=np.float64).reshape((3, 1)) if "t" in cam else np.zeros((3, 1))
            
            if "distCoef" in cam:
                dist = np.array(cam["distCoef"], dtype=np.float64)
            elif "dist" in cam:
                dist = np.array(cam["dist"], dtype=np.float64)
            else:
                dist = np.zeros(5, dtype=np.float64)  # Explicit zero-distortion provenance when absent
                
            name = str(cam.get("name", cam.get("node", str(cam.get("id", i)))))
            img_size = tuple(cam["resolution"]) if "resolution" in cam else (tuple(cam["image_size"]) if "image_size" in cam else None)
            
            cameras.append(CameraModel(name=name, K=K, R=R, t=t, dist=dist, image_size=img_size))
    else:
        cam_dict = {str(cam.get("name", cam.get("node", str(cam.get("id"))))): cam for cam in data.get("cameras", [])}
        for name in camera_names:
            if name not in cam_dict:
                raise ValueError(f"Could not find exact calibration name '{name}' in {path}")
                
            best_match = cam_dict[name]
            K = np.array(best_match["K"], dtype=np.float64)
            R = np.array(best_match["R"], dtype=np.float64) if "R" in best_match else np.eye(3)
            t = np.array(best_match["t"], dtype=np.float64).reshape((3, 1)) if "t" in best_match else np.zeros((3, 1))
            
            if "distCoef" in best_match:
                dist = np.array(best_match["distCoef"], dtype=np.float64)
            elif "dist" in best_match:
                dist = np.array(best_match["dist"], dtype=np.float64)
            else:
                dist = np.zeros(5, dtype=np.float64)  # Explicit zero-distortion provenance when absent
                
            img_size = tuple(best_match["resolution"]) if "resolution" in best_match else (tuple(best_match["image_size"]) if "image_size" in best_match else None)
            
            cameras.append(CameraModel(name=name, K=K, R=R, t=t, dist=dist, image_size=img_size))
            
    return cameras
