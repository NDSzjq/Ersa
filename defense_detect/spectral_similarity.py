# defense_detect/spectral_similarity.py
import torch

def compute_pairwise_similarity(K_list, method="cosine"):
    """
    计算谱坐标矩阵之间的相似度矩阵
    
    Args:
        K_list: list of (NR, NR) 张量
        method: "cosine" (默认) 或其他方法
    
    Returns:
        sim_matrix: (N, N) 相似度矩阵
    """
    N = len(K_list)
    if N == 0:
        return torch.tensor([])
    
    device = K_list[0].device
    sim_matrix = torch.zeros(N, N, device=device)
    
    if method == "cosine":
        for i in range(N):
            for j in range(i + 1, N):
                numerator = torch.sum(K_list[i] * K_list[j])
                denominator = torch.norm(K_list[i], p='fro') * torch.norm(K_list[j], p='fro')
                sim = numerator / (denominator + 1e-12)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
            sim_matrix[i, i] = 1.0
    else:
        raise ValueError(f"Unsupported similarity method: {method}")
    
    return sim_matrix