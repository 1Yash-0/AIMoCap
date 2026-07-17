import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class CameraModel:
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int] | None

    def __post_init__(self):
        # Validate K shape
        if self.K.shape != (3, 3):
            raise ValueError("K must be 3x3")
        # Validate R shape and properties
        if self.R.shape != (3, 3):
            raise ValueError("R must be 3x3")
        
        # Check orthonormal R.T @ R = I
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-4):
            raise ValueError("R must be orthonormal")
            
        # Check determinant near +1
        det = np.linalg.det(self.R)
        if not np.isclose(det, 1.0, atol=1e-4):
            raise ValueError(f"R determinant must be +1, got {det}")
            
        # Validate t shape
        if self.t.shape not in [(3,), (3, 1)]:
            raise ValueError("t must be 3x1 or 3")
            
        # Positive focal lengths
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("Focal lengths must be positive")
            
        # Positive image dimensions
        if self.image_size is not None:
            if self.image_size[0] <= 0 or self.image_size[1] <= 0:
                raise ValueError("Image dimensions must be positive")
                
        # Ensure finite values
        if not (np.isfinite(self.K).all() and np.isfinite(self.R).all() and 
                np.isfinite(self.t).all() and np.isfinite(self.dist).all()):
            raise ValueError("Camera parameters must be finite")

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Projects 3D points (in OpenCV world coordinates) into 2D image coordinates.
        Uses A: distorted pixel observations with cv2.projectPoints.
        """
        import cv2
        if len(points_3d) == 0:
            return np.zeros((0, 2), dtype=np.float32)
            
        points_3d = np.asarray(points_3d, dtype=np.float64)
        
        rvec, _ = cv2.Rodrigues(self.R)
        tvec = self.t.reshape(3, 1)
        
        projected, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist)
        return projected.reshape(-1, 2)
