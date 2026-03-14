import torch
import numpy as np
from typing import List

def aggregate_ranking_scores(pairwise_probs: torch.Tensor, num_frames: int) -> List[int]:
    """
    Compute: score_i = sum_j P(i < j)
    Sort frames using these ranking scores.
    
    pairwise_probs: A (N, N) matrix where element (i, j) is P(i < j)
    """
    # Sum probabilities along the row (i.e. over all j)
    scores = pairwise_probs.sum(dim=1).cpu().numpy()
    
    # We want to sort such that frames with higher scores (more often predicted to be earlier)
    # come first. So we sort descending.
    # Alternatively, if score_i is P(i < j), high score means it's before many others.
    # Descending argsort:
    predicted_order = np.argsort(-scores).tolist()
    
    return predicted_order

def build_pairwise_matrix(model, contextual_embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Build an NxN matrix of pairwise probabilities from model predictions.
    """
    num_frames = contextual_embeddings.size(1)
    # (1, T, D)
    emb = contextual_embeddings.squeeze(0)
    
    prob_matrix = torch.zeros((num_frames, num_frames), device=device)
    
    # Can be optimized to batch inference
    with torch.no_grad():
        for i in range(num_frames):
            for j in range(num_frames):
                if i == j:
                    prob_matrix[i, j] = 0.5
                else:
                    # emb[i], emb[j]
                    logits = model.pairwise_comparator(emb[i].unsqueeze(0), emb[j].unsqueeze(0))
                    prob = torch.sigmoid(logits).squeeze()
                    prob_matrix[i, j] = prob
                    
    return prob_matrix
