import os
from collections import defaultdict
import numpy as np
from typing import List, Dict

metrics_registry = defaultdict(list)

def log_metric(name: str, value: float):
    metrics_registry[name].append(value)

def get_avg_metric(name: str) -> float:
    if not metrics_registry[name]:
        return 0.0
    return float(np.mean(metrics_registry[name]))

def reset_metrics():
    metrics_registry.clear()

def kendall_tau_score(pred_order: List[int], true_order: List[int]) -> float:
    """
    Calculate Kendall Tau score for pairwise ranking consistency.
    """
    from scipy.stats import kendalltau
    tau, _ = kendalltau(pred_order, true_order)
    # Handle NaN for sequences of length < 2 or identical sequences
    if np.isnan(tau):
        return 0.0
    return float(tau)

def pairwise_accuracy(pred_probs: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Calculate accuracy of pairwise predictions.
    """
    preds = (pred_probs > 0.5).astype(float).flatten()
    true_labels = true_labels.flatten()
    correct = (preds == true_labels).sum()
    return float(correct / len(true_labels)) if len(true_labels) > 0 else 0.0
