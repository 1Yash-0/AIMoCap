"""Coordinate system definitions and conversions.

This module defines the internal canonical 3D coordinate system for the pipeline:
**Y-up, right-handed, meters**.

Conventions:
- X: Right
- Y: Up
- Z: Back (towards the viewer)

This matches glTF, Three.js, and Blender. It minimizes conversions at the export stage.
OpenCV's native 3D space (from triangulation) is Y-down, Z-forward (right-handed).
We convert from OpenCV space to our internal space immediately after triangulation.
"""

import numpy as np

def opencv_to_internal(points_3d: np.ndarray) -> np.ndarray:
    """Convert points from OpenCV space to the internal Y-up, right-handed space.
    
    OpenCV: X right, Y down, Z forward
    Internal: X right, Y up, Z back
    
    Args:
        points_3d: Array of shape (..., 3) containing 3D points in OpenCV space.
        
    Returns:
        Array of the same shape in internal coordinate space.
    """
    out = np.copy(points_3d)
    out[..., 1] = -out[..., 1]  # Y up
    out[..., 2] = -out[..., 2]  # Z back
    return out

def internal_to_opencv(points_3d: np.ndarray) -> np.ndarray:
    """Convert points from the internal Y-up space back to OpenCV space.
    
    This is useful for reprojection (rendering 3D points back to 2D cameras).
    """
    out = np.copy(points_3d)
    out[..., 1] = -out[..., 1]  # Y down
    out[..., 2] = -out[..., 2]  # Z forward
    return out
