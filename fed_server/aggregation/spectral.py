# fed_server/aggregation/spectral.py
import torch

def compute_spectral_coordinate(Sigma_B, V_B_block, U_A_block, Sigma_A):
    """
    计算单个客户端的谱坐标矩阵 K_n
    
    K_n = Sigma_B @ V_B_block @ U_A_block @ Sigma_A
    """
    return Sigma_B @ V_B_block @ U_A_block @ Sigma_A


def compute_all_spectral_coordinates(bfsvd_result):
    """
    计算所有客户端的谱坐标矩阵
    
    参数:
        bfsvd_result: fedsvd_aggregate 的返回值
    
    返回:
        K_list: list of (NR, NR) 每个客户端的谱坐标矩阵
    """
    Sigma_B = bfsvd_result["Sigma_B"]
    Sigma_A = bfsvd_result["Sigma_A"]
    V_B_blocks = bfsvd_result["V_B_blocks"]
    U_A_blocks = bfsvd_result["U_A_blocks"]

    K_list = []
    for n in range(len(V_B_blocks)):
        K_n = compute_spectral_coordinate(
            Sigma_B, V_B_blocks[n], U_A_blocks[n], Sigma_A
        )
        K_list.append(K_n)

    return K_list