import numpy as np

# OpenCV space:
# Handedness: Right-handed
# Up axis: -Y (Y points down)
# Forward axis: +Z

# Canonical space:
# Handedness: Right-handed
# Up axis: +Y
# Forward axis: -Z (so Z points back)

# Matrix transforming OpenCV -> Canonical
# R_cv2can @ [1, 0, 0].T = [1, 0, 0].T   (X remains X)
# R_cv2can @ [0, 1, 0].T = [0, -1, 0].T  (Y-down becomes Y-up)
# R_cv2can @ [0, 0, 1].T = [0, 0, -1].T  (Z-forward becomes Z-backward)

OPENCV_TO_CANONICAL_MATRIX = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
], dtype=np.float64)

CANONICAL_TO_OPENCV_MATRIX = OPENCV_TO_CANONICAL_MATRIX.T

def verify_coordinate_contract():
    R = OPENCV_TO_CANONICAL_MATRIX
    # 1. Exact 3x3 matrix (done)
    # 2. Determinant
    det = np.linalg.det(R)
    assert np.isclose(det, 1.0), f"Determinant is {det}, expected +1.0 for right-handed rotation"
    
    # 3. Orthonormal R.T @ R = I
    I = R.T @ R
    assert np.allclose(I, np.eye(3)), "Matrix is not orthonormal"
    
    # 4. Cross product preservation (orientation preservation)
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    # R * (u x v) == (Ru) x (Rv)
    cp_orig = np.cross(u, v)
    cp_trans = R @ cp_orig
    cp_new = np.cross(R @ u, R @ v)
    assert np.allclose(cp_trans, cp_new), "Cross product not preserved"

def opencv_to_canonical(pts3d: np.ndarray) -> np.ndarray:
    """Convert points from OpenCV space to Canonical space."""
    return pts3d @ OPENCV_TO_CANONICAL_MATRIX.T

def canonical_to_opencv(pts3d: np.ndarray) -> np.ndarray:
    """Convert points from Canonical space to OpenCV space."""
    return pts3d @ CANONICAL_TO_OPENCV_MATRIX.T
