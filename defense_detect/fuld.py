# defense_detect/fuld.py
import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

from fed_server.aggregation.spectral import compute_all_spectral_coordinates
from .spectral_similarity import compute_pairwise_similarity
from .clustering import get_benign_set
from .weights import compute_weights


def fuld_detect(bfsvd_result, G_n, similarity_method="cosine", eps=None, min_samples=None):
    """ 
    Returns:
        dict: {
            "weights": (N,) Tensor,
            "benign_set": list,
            "labels": ndarray,
            "similarity": (N, N) Tensor,
            "K_list": list,
        }
    """
    # Step 1: 谱坐标
    K_list = compute_all_spectral_coordinates(bfsvd_result)

    print("\n===== FULD K_n =====")
    for i, K in enumerate(K_list):
        print(
            f"Client {i}: "
            f"shape={K.shape}, "
            f"norm={torch.norm(K).item():.6f}"
        )

    # Step 2: 相似度矩阵（SSCosine 明文版）
    sim_matrix_pytorch = compute_pairwise_similarity(K_list, method=similarity_method)
    print("\n===== Similarity =====")
    print(
        sim_matrix_pytorch.detach().cpu().numpy()
    )
    
    # Step 3: 聚类得到良性集合
    benign_set, labels = get_benign_set(sim_matrix_pytorch, eps, min_samples)
    print("\n===== Cluster =====")
    print("labels =", labels)
    print("benign_set =", benign_set)
    
    # Step 4: 计算聚合权重
    weights = compute_weights(benign_set, G_n)
    print("weights =", weights)
    
    return {
        "weights": weights,
        "benign_set": benign_set,
        "labels": labels,
        "similarity": sim_matrix_pytorch,
        "K_list": K_list,
    }