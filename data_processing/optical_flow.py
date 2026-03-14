import cv2
import numpy as np
from typing import List, Tuple

class OpticalFlowExtractor:
    """
    Computes Farneback dense optical flow between consecutive frames.
    """
    def __init__(self):
        pass

    def compute_flow(self, prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow using Farneback algorithm.
        Returns flow map of shape (H, W, 2) where channels are (magnitude, angle)
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Convert Cartesian coordinates to magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize magnitude
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Stack as shape (H, W, 2)
        flow_features = np.stack([mag, ang], axis=-1)
        return flow_features

    def compute_sequence_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute flow between all consecutive frames in a sequence.
        Returns N-1 flow maps for N frames.
        """
        if len(frames) < 2:
            return []
            
        flows = []
        for i in range(len(frames) - 1):
            flow = self.compute_flow(frames[i], frames[i+1])
            flows.append(flow)
            
        return flows
        
    def sequence_flow_magnitudes(self, frames: List[np.ndarray]) -> List[float]:
        """
        Returns global motion magnitude per frame transition.
        Useful for motion-aware sampling.
        """
        flows = self.compute_sequence_flow(frames)
        magnitudes = [float(np.mean(f[..., 0])) for f in flows]
        return magnitudes
