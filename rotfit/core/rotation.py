"""共享旋转数学原语。

从最干净的基准实现（原 1.py）抽取，供所有模型变体复用，避免各文件重复实现。
这里只放「与具体模型无关」的函数；模型相关逻辑（轴 softmax、Sinkhorn 温度调度等）
留在各 models 中。
"""

import numpy as np
import torch
import torch.nn.functional as F


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20, temp: float = 1.0) -> torch.Tensor:
    """用 Sinkhorn 迭代把对数分数矩阵变成近似双随机（软置换）矩阵。"""
    K = torch.exp(log_alpha / temp)
    for _ in range(n_iters):
        K = K / (K.sum(dim=1, keepdim=True) + 1e-12)
        K = K / (K.sum(dim=0, keepdim=True) + 1e-12)
    return K


def axis_angle_to_matrix_batch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """把一个（批）轴角表示转成旋转矩阵（Rodrigues 公式）。

    axis: (B,3) 已归一化方向；angle: (B,) 弧度。
    返回 (B,3,3)。
    """
    if axis.dim() == 1:
        axis = axis.unsqueeze(0)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-12)
    if angle.dim() == 0:
        angle = angle.unsqueeze(0)
    if angle.dim() == 1 and angle.shape[0] != axis.shape[0]:
        if axis.shape[0] == 1:
            angle = angle.unsqueeze(0)
        else:
            angle = angle.expand(axis.shape[0])
    ang = angle.view(-1)
    x = axis[:, 0]; y = axis[:, 1]; z = axis[:, 2]
    c = torch.cos(ang); s = torch.sin(ang)
    R = torch.stack([
        c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s,
        y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s,
        z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)
    ], dim=-1).reshape(-1, 3, 3)
    return R


def softperm_to_perm(P: np.ndarray) -> list:
    """把软置换矩阵转成硬置换（整数列表）。

    优先用 Hungarian 算法（scipy）；不可用时退化为贪心最大匹配。
    """
    try:
        from scipy.optimize import linear_sum_assignment
        cost = -P
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = [-1] * P.shape[0]
        for r, c in zip(row_ind, col_ind):
            perm[r] = int(c)
        return perm
    except Exception:
        n = P.shape[0]
        perm = [-1] * n
        used = set()
        flat = []
        for i in range(n):
            for j in range(n):
                flat.append((P[i, j], i, j))
        flat.sort(reverse=True)
        for _, i, j in flat:
            if perm[i] == -1 and j not in used:
                perm[i] = int(j)
                used.add(j)
            if all(x != -1 for x in perm):
                break
        remaining = [c for c in range(n) if c not in used]
        for i in range(n):
            if perm[i] == -1:
                perm[i] = remaining.pop(0)
        return perm


def permute_rows(arr: np.ndarray, perm: list) -> np.ndarray:
    """按 perm 重排行（perm[i] 表示第 i 行取原第 perm[i] 行）。"""
    out = np.zeros_like(arr)
    for i in range(len(perm)):
        out[i, :] = arr[perm[i], :]
    return out
