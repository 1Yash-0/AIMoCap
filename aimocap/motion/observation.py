from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Observation:
    """A single frame of multi-view 2D observations.
    
    Attributes:
        frame_idx: Integer frame index.
        kpts2d: (C, K, 2) array of 2D coordinates.
        scores: (C, K) array of confidence scores.
        camera_ids: List of camera identifiers corresponding to the C axis.
    """
    frame_idx: int
    kpts2d: np.ndarray
    scores: np.ndarray
    camera_ids: List[int]
    
    @property
    def num_cameras(self) -> int:
        return len(self.camera_ids)
        
    @property
    def num_keypoints(self) -> int:
        return self.kpts2d.shape[1]
