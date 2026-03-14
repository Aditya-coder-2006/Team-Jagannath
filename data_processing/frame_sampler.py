import numpy as np
from typing import List, Optional

class FrameSampler:
    """
    Sample a fixed number of frames from a varying length video sequence.
    """
    def __init__(self, target_frames: int = 32, method: str = "uniform"):
        self.target_frames = target_frames
        self.method = method

    def sample(self, frames: List[np.ndarray], flow_magnitudes: Optional[List[float]] = None) -> List[int]:
        total_frames = len(frames)
        if total_frames == 0:
            return []

        if self.method == "uniform":
            return self._uniform_sampling(total_frames)
        elif self.method == "motion_aware":
            return self._motion_aware_sampling(total_frames, flow_magnitudes)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

    def _uniform_sampling(self, total_frames: int) -> List[int]:
        if total_frames <= self.target_frames:
            # Pad with last frame index if shorter than target
            indices = list(range(total_frames))
            while len(indices) < self.target_frames:
                indices.append(total_frames - 1)
            return indices
        else:
            # Uniformly sample indices
            indices = np.linspace(0, total_frames - 1, self.target_frames, dtype=int)
            return indices.tolist()

    def _motion_aware_sampling(self, total_frames: int, flow_magnitudes: Optional[List[float]]) -> List[int]:
        if flow_magnitudes is None or len(flow_magnitudes) != total_frames - 1:
            # Fallback to uniform if flow data is missing or mismatched
            return self._uniform_sampling(total_frames)
            
        if total_frames <= self.target_frames:
            return self._uniform_sampling(total_frames)

        # Distribute frames based on motion magnitude (more motion = denser sampling)
        # Add small epsilon to avoid zero probability
        probs = np.array(flow_magnitudes) + 1e-5
        probs = probs / probs.sum()
        
        # We sample target_frames - 1 from the intervals, plus the first frame
        # to ensure we capture motion peaks
        sampled_indices = [0]
        chosen = np.random.choice(range(1, total_frames), size=self.target_frames - 1, replace=False, p=probs)
        sampled_indices.extend(sorted(chosen.tolist()))
        
        return sorted(sampled_indices)
