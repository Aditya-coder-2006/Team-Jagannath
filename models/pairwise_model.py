import torch
import torch.nn as nn

class PairwiseComparator(nn.Module):
    """
    Takes two frame embeddings (frame_i, frame_j) and predicts the probability
    that frame_i comes before frame_j P(frame_i < frame_j).
    """
    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        
        # We use a Siamese-like approach or direct concatenation.
        # Concatenation followed by MLP is powerful for capturing relationship.
        self.comparator = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, frame_i: torch.Tensor, frame_j: torch.Tensor) -> torch.Tensor:
        """
        Inputs: frame_i, frame_j of shape (B, embed_dim)
        Output: logits of shape (B, 1) representing P(i < j)
        """
        x = torch.cat([frame_i, frame_j], dim=-1)
        logits = self.comparator(x)
        return logits
