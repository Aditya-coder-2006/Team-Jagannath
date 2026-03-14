import os
import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Ensure `project_root` is on sys.path so local imports work when run from repo root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.video_sorter import VideoChronologicalSorter
from data_processing.frame_extractor import FrameExtractor
from data_processing.optical_flow import OpticalFlowExtractor
from data_processing.frame_sampler import FrameSampler
from inference.ranking import aggregate_ranking_scores, build_pairwise_matrix
from utils.config import load_config

class VideoPredictor:
    def __init__(self, checkpoint_path: str, config_path: str):
        self.config = load_config(config_path)
        self.device = torch.device(self.config.training.device if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = VideoChronologicalSorter(self.config)
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Init Data Processors
        self.extractor = FrameExtractor()
        sampling_method = getattr(self.config.training, "sampling_method", "motion_aware")
        self.sampler = FrameSampler(target_frames=self.config.model.num_frames, method=sampling_method)
        self.flow_extractor = OpticalFlowExtractor()

    def predict(self, video_path: str) -> str:
        """
        Clean callable API: predict(video_path) -> frame_order string
        """
        # 1. Extract frames
        frames = self.extractor.get_frames(video_path)
        
        # 2. Extract flow magnitudes for sampling
        flow_mags = self.flow_extractor.sequence_flow_magnitudes(frames)
        
        # 3. Sample frames
        indices = self.sampler.sample(frames, flow_mags)
        sampled_frames = [frames[i] for i in indices]
        
        # 4. Extract dense flow for the sampled frames
        sampled_flows = self.flow_extractor.compute_sequence_flow(sampled_frames)
        # Pad last flow to match frame count
        if len(sampled_flows) > 0:
            sampled_flows.append(sampled_flows[-1])
        else:
            # Fallback for very short videos
            sampled_flows = [torch.zeros((224, 224, 2)).numpy() for _ in sampled_frames]
            
        # 5. Tensor conversions
        # frames: (T, H, W, C) -> (1, T, C, H, W)
        frames_tensor = torch.tensor(np.stack(sampled_frames)).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
        
        # Note: Depending on your exact flow shape, you might pool it
        # For simplicity, we globally average the flow: (T, H, W, 2) -> (T, 2)
        pooled_flows = [f.mean(axis=(0, 1)) for f in sampled_flows]
        flows_tensor = torch.tensor(np.stack(pooled_flows)).unsqueeze(0).float()
        
        frames_tensor = frames_tensor.to(self.device)
        flows_tensor = flows_tensor.to(self.device)
        
        with torch.no_grad():
            # 6. Model encoding
            embeddings = self.model.encode_frames(frames_tensor, flows_tensor)
            
            # 7. Pairwise comparator
            prob_matrix = build_pairwise_matrix(self.model, embeddings, self.device)
            
            # 8. Ranking
            pred_order = aggregate_ranking_scores(prob_matrix, self.config.model.num_frames)
            
        # Submission format expects one-indexed frame positions.
        order_str = " ".join([str(idx + 1) for idx in pred_order])
        return order_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file or frame directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    predictor = VideoPredictor(args.checkpoint, args.config)
    order_str = predictor.predict(args.video_path)
    
    video_id = Path(args.video_path).stem
    print(f'{video_id},"{order_str}"')

if __name__ == "__main__":
    main()
