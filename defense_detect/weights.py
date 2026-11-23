# defense_detect/weights.py
import torch

def compute_weights(benign_set, G_n):
    """  
    Args:
        benign_set: list of 良性客户端索引
        G_n: list of 各客户端样本数
    
    Returns:
        weights: (N,) Tensor
    """
    N = len(G_n)
    
    counts = torch.as_tensor(G_n, dtype=torch.float32)
    device = counts.device
    weights = torch.zeros_like(counts)
    
    if benign_set is None or len(benign_set) == 0:
        return weights
    
    total_samples = counts[benign_set].sum()
    if total_samples <= 0:
        return weights
    
    for i in benign_set:
        weights[i] = counts[i] / total_samples
    
    return weights
