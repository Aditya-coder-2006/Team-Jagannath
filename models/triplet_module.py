import torch
import torch.nn as nn

class TripletTemporalModule(nn.Module):
    """
    Takes three frame embeddings (A, B, C) and predicts their correct temporal arrangement.
    There are 3! = 6 possible permutations.
    """
    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6) # 6 permutations
        )

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor, frame_c: torch.Tensor) -> torch.Tensor:
        """
        Inputs: frame embeddings of shape (B, embed_dim)
        Output: logits for 6 permutations (B, 6)
        """
        # Concatenate the three frames
        x = torch.cat([frame_a, frame_b, frame_c], dim=-1)
        logits = self.classifier(x)
        return logits
