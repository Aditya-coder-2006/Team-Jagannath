import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .temporal_transformer import TemporalTransformer
from .fusion import FeatureFusion
from .triplet_module import TripletTemporalModule
from .pairwise_model import PairwiseComparator

class VideoChronologicalSorter(nn.Module):
    """
    Unified model architecture that combines the CNN backbone,
    optical flow fusion, and the temporal transformer, before computing
    pairwise and triplet predictions.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config.model
        
        self.cnn_encoder = CNNEncoder(
            feature_dim=self.config.feature_dim,
            freeze_layers=True
        )
        
        self.fusion = FeatureFusion(
            cnn_dim=self.config.feature_dim,
            flow_dim=self.config.flow_dim,
            output_dim=self.config.fused_dim
        )
        
        self.temporal_transformer = TemporalTransformer(
            d_model=self.config.fused_dim,
            nhead=self.config.transformer_heads,
            num_layers=self.config.transformer_layers,
            dim_feedforward=self.config.fused_dim * 2,
            dropout=self.config.dropout
        )
        
        self.pairwise_comparator = PairwiseComparator(
            embed_dim=self.config.fused_dim,
            hidden_dim=self.config.fused_dim // 2
        )
        
        self.triplet_module = TripletTemporalModule(
            embed_dim=self.config.fused_dim,
            hidden_dim=self.config.fused_dim // 2
        )

    def encode_frames(self, frames: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, C, H, W)
        flows: (B, T, 2)
        Output: Contextual embeddings (B, T, fused_dim)
        """
        B, T, C, H, W = frames.shape
        
        # Merge batch and time dimensions for CNN backbone
        frames_flat = frames.view(-1, C, H, W)
        cnn_features = self.cnn_encoder(frames_flat)
        
        # Reshape to (B*T, CNN_DIM) and flows to (B*T, FLOW_DIM)
        flows_flat = flows.view(-1, self.config.flow_dim)
        
        # Fuse spatial and motion features
        fused_features = self.fusion(cnn_features, flows_flat)
        fused_features = fused_features.view(B, T, self.config.fused_dim)
        
        # Add broad temporal context using transformer
        contextual_embeddings = self.temporal_transformer(fused_features)
        
        return contextual_embeddings

    def forward(self, frames: torch.Tensor, flows: torch.Tensor):
        """
        Standard forward for compatibility with regular PyTorch training/inference calls.
        Returns pairwise and triplet heads along with contextual embeddings.
        """
        embeddings = self.encode_frames(frames, flows)
        return {
            "embeddings": embeddings,
            "pairwise_head": self.pairwise_comparator,
            "triplet_head": self.triplet_module,
        }
