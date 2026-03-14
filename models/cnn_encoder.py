import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CNNEncoder(nn.Module):
    """
    Uses pretrained ResNet50 as a backbone to extract 2048-dimensional features
    from individual video frames.
    """
    def __init__(self, feature_dim: int = 2048, freeze_layers: bool = True):
        super().__init__()

        # Try pretrained weights first; fall back to random init if download/cache is unavailable.
        try:
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception as exc:
            print(f"[WARNING] Could not load pretrained ResNet50 weights ({exc}). Using random initialization.")
            resnet = resnet50(weights=None)
        
        # Remove the classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optional freezing of earlier layers to save memory and compute
        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze the last block
            for param in list(resnet.children())[-2].parameters():
                param.requires_grad = True
                
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Batch of frames (B, C, H, W)
        Output: Feature vectors (B, 2048)
        """
        features = self.backbone(x)
        # Squeeze spatial dimensions (B, 2048, 1, 1) -> (B, 2048)
        features = features.view(features.size(0), -1)
        return features
