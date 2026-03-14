import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """
    Fuses CNN frame features with optical flow motion features.
    Normalizes the fused embeddings.
    """
    def __init__(self, cnn_dim: int = 2048, flow_dim: int = 2, output_dim: int = 1024):
        super().__init__()
        
        self.cnn_dim = cnn_dim
        self.flow_dim = flow_dim
        self.output_dim = output_dim
        
        # To handle pooled flow features, wait, flow features might be flattened 
        # or globally pooled. Assuming flow features are globally averaged to (B, 2)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(cnn_dim + flow_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, cnn_feat: torch.Tensor, flow_feat: torch.Tensor) -> torch.Tensor:
        """
        cnn_feat: (B, 2048)
        flow_feat: (B, 2) global motion magnitude/direction
        """
        # Concatenate along feature dimension
        fused = torch.cat((cnn_feat, flow_feat), dim=-1)
        
        # Pass through fusion MLP
        out = self.fusion_layer(fused)
        
        # Normalize to unit length
        out = nn.functional.normalize(out, p=2, dim=-1)
        return out
