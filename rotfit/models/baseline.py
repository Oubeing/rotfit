"""基准实现（原 1.py）——逻辑最干净，依赖 rotfit.core。

这是推荐「从此处扩展」的版本：连续/离散两种模式、Sinkhorn 近似顺序、
低温度 + 训练软轴/推理离散轴。其余变体（compatible/optimized/ultimate/
fixed_discrete）是历史上不同优化方向的实现，保留为参考。

用法：
    from rotfit.models import baseline
    baseline.train_and_save("configs/baseline.json")   # 从项目根目录运行
或通过 scripts/train.py：
    python scripts/train.py --config configs/baseline.json
"""

import json
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from rotfit.core.rotation import (
    sinkhorn,
    axis_angle_to_matrix_batch,
    softperm_to_perm,
    permute_rows,
)
from rotfit.core.data import read_dataset

# -----------------------------
# 环境
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True


# -----------------------------
# 模型
# -----------------------------
class RotationFitter(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, mode: str = "continuous",
                 fixed_axes_in: Optional[Dict[int, List[float]]] = None,
                 fixed_axes_out: Optional[Dict[int, List[float]]] = None,
                 fixed_in_order: Optional[Dict[int, int]] = None,
                 fixed_out_order: Optional[Dict[int, int]] = None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.mode = mode
        if mode == "continuous":
            self.in_axes_param = torch.nn.Parameter(torch.randn(n_in, 3) * 0.5)
            self.out_axes_param = torch.nn.Parameter(torch.randn(n_out, 3) * 0.5)
        else:
            # 使用小的随机初始化而不是全零，避免对称性
            self.in_axes_logits = torch.nn.Parameter(torch.randn(n_in, 6) * 0.1)
            self.out_axes_logits = torch.nn.Parameter(torch.randn(n_out, 6) * 0.1)
        # 使用小的随机初始化打破对称性，避免Sinkhorn不收敛
        self.in_order_logits = torch.nn.Parameter(torch.randn(n_in, n_in) * 0.1)
        self.out_order_logits = torch.nn.Parameter(torch.randn(n_out, n_out) * 0.1)
        self.register_buffer("axis_map", torch.tensor(
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
             [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]))
        self.fixed_axes_in = fixed_axes_in or {}
        self.fixed_axes_out = fixed_axes_out or {}
        self.fixed_in_order = fixed_in_order or {}
        self.fixed_out_order = fixed_out_order or {}

    def forward(self, in_angles: torch.Tensor, out_angles: torch.Tensor,
                probe_v: torch.Tensor, sinkhorn_iters: int = 32, sinkhorn_temp: float = 0.5):
        B = in_angles.shape[0]
        device = in_angles.device  # 统一 device

        # Sinkhorn 近似排列矩阵
        P_in = sinkhorn(self.in_order_logits.to(device), sinkhorn_iters, sinkhorn_temp)
        P_out = sinkhorn(self.out_order_logits.to(device), sinkhorn_iters, sinkhorn_temp)

        # ---- 固定顺序处理 ----
        if self.fixed_in_order:
            with torch.no_grad():
                for pos_str, idx in self.fixed_in_order.items():
                    pos = int(pos_str)
                    self.in_order_logits[pos, :] = -50.0
                    self.in_order_logits[pos, int(idx)] = 50.0
            P_in = sinkhorn(self.in_order_logits.to(device), sinkhorn_iters, sinkhorn_temp)

        if self.fixed_out_order:
            with torch.no_grad():
                for pos_str, idx in self.fixed_out_order.items():
                    pos = int(pos_str)
                    self.out_order_logits[pos, :] = -50.0
                    self.out_order_logits[pos, int(idx)] = 50.0
            P_out = sinkhorn(self.out_order_logits.to(device), sinkhorn_iters, sinkhorn_temp)

        # ---- 构建旋转轴 ----
        if self.mode == "continuous":
            axes_in = P_in @ self.in_axes_param.to(device)
            axes_out = P_out @ self.out_axes_param.to(device)
        else:
            # 使用更低温度使选择更确定，然后强制离散化
            temp = 0.1
            axes_in_soft = P_in @ (F.softmax(self.in_axes_logits.to(device) / temp, dim=-1) @ self.axis_map.to(device))
            axes_out_soft = P_out @ (F.softmax(self.out_axes_logits.to(device) / temp, dim=-1) @ self.axis_map.to(device))

            # 训练时使用软轴保持梯度，推理时离散化
            if self.training:
                axes_in = axes_in_soft
                axes_out = axes_out_soft
            else:
                # 推理时强制离散化
                def discretize_axes(soft_axes, axis_map):
                    discretized = []
                    for axis in soft_axes:
                        similarities = F.cosine_similarity(axis.unsqueeze(0), axis_map, dim=1)
                        best_idx = torch.argmax(similarities)
                        discretized.append(axis_map[best_idx])
                    return torch.stack(discretized)

                axes_in = discretize_axes(axes_in_soft, self.axis_map.to(device))
                axes_out = discretize_axes(axes_out_soft, self.axis_map.to(device))

        # ---- 固定轴处理 ----
        def apply_fixed_axes(tensor, fixed_axes):
            for idx_str, vec in fixed_axes.items():
                idx = int(idx_str)
                tensor[idx, :] = torch.tensor(vec, device=device)
            return tensor

        axes_in = apply_fixed_axes(axes_in, self.fixed_axes_in)
        axes_out = apply_fixed_axes(axes_out, self.fixed_axes_out)

        # 增强数值稳定性的轴归一化
        axes_in = axes_in / (axes_in.norm(dim=-1, keepdim=True) + 1e-8)
        axes_out = axes_out / (axes_out.norm(dim=-1, keepdim=True) + 1e-8)

        # ---- 构建旋转矩阵 ----
        Rin = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        for pos in range(self.n_in):
            axis_batch = axes_in[pos].unsqueeze(0).repeat(B, 1).to(device)
            angle_batch = in_angles[:, pos].to(device)
            Rstep = axis_angle_to_matrix_batch(axis_batch, angle_batch)
            Rin = Rstep @ Rin

        Rout = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        for pos in range(self.n_out):
            axis_batch = axes_out[pos].unsqueeze(0).repeat(B, 1).to(device)
            angle_batch = out_angles[:, pos].to(device)
            Rstep = axis_angle_to_matrix_batch(axis_batch, angle_batch)
            Rout = Rstep @ Rout

        # ---- probe vector ----
        if probe_v.dim() == 2 and probe_v.shape[1] == 1:  # 3x1
            v_batch = probe_v.to(device).unsqueeze(0).repeat(B, 1, 1)
        else:  # 3-dim
            v_batch = probe_v.to(device).view(1, 3, 1).repeat(B, 1, 1)

        vin = Rin @ v_batch
        vout = Rout @ v_batch

        return vin, vout, Rin, Rout, P_in, P_out, axes_in, axes_out


# -----------------------------
# 训练 & 保存结果
# -----------------------------
def train_and_save(config_file: str):
    global DEVICE
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    data_file = cfg["data"]["file"]
    mode = cfg["rotation"].get("mode", "continuous")
    probe_vectors = cfg["rotation"].get("vectors", [[1, 0, 0]])
    probe_v = torch.tensor(probe_vectors, dtype=torch.float64, device=DEVICE)
    fixed_axes_in = cfg["rotation"].get("fixed_axes", {}).get("in", {})
    fixed_axes_out = cfg["rotation"].get("fixed_axes", {}).get("out", {})
    fixed_in_order = cfg["rotation"].get("fixed_order", {}).get("in", {})
    fixed_out_order = cfg["rotation"].get("fixed_order", {}).get("out", {})
    epochs = cfg["training"].get("epochs", 2000)
    lr = cfg["training"].get("learning_rate", 1e-2)
    sinkhorn_iters = cfg["training"].get("sinkhorn_iters", 32)
    sinkhorn_temp = cfg["training"].get("sinkhorn_temp", 0.5)
    device = cfg["training"].get("device", None)
    if device:
        DEVICE = torch.device(device)
    result_file = cfg["output"].get("result_file", "rotation_fit_result.json")

    # read data
    in_angles_rad, out_angles_rad, in_cols, out_cols = read_dataset(data_file)
    N, n_in = in_angles_rad.shape
    _, n_out = out_angles_rad.shape
    in_angles = torch.tensor(in_angles_rad, dtype=torch.float64, device=DEVICE)
    out_angles = torch.tensor(out_angles_rad, dtype=torch.float64, device=DEVICE)

    # model
    model = RotationFitter(n_in, n_out, mode, fixed_axes_in, fixed_axes_out, fixed_in_order, fixed_out_order).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        # 对每个probe vector计算损失并平均
        for i in range(probe_v.shape[0]):
            current_probe = probe_v[i:i + 1]  # 保持维度
            vin, vout, _, _, _, _, _, _ = model(in_angles, out_angles, current_probe, sinkhorn_iters, sinkhorn_temp)
            probe_loss = (vin - vout).pow(2).sum() / N
            total_loss += probe_loss

        loss = total_loss / probe_v.shape[0]  # 平均损失
        loss.backward()
        optimizer.step()
        if ep % 500 == 0 or ep == epochs - 1:
            print(f"[{ep}/{epochs}] Loss={loss.item():.6e}")

    # 获取最终结果 - 使用第一个probe vector作为代表
    first_probe = probe_v[0:1]  # 保持维度
    vin, vout, Rin, Rout, P_in, P_out, axes_in, axes_out = model(in_angles, out_angles, first_probe, sinkhorn_iters, sinkhorn_temp)
    # Sinkhorn 输出 NaN/inf 处理
    P_in_np = P_in.detach().cpu().numpy()
    P_in_np = np.nan_to_num(P_in_np, nan=1e-6, posinf=1e6, neginf=-1e6)
    P_out_np = P_out.detach().cpu().numpy()
    P_out_np = np.nan_to_num(P_out_np, nan=1e-6, posinf=1e6, neginf=-1e6)
    in_order = softperm_to_perm(P_in_np)
    out_order = softperm_to_perm(P_out_np)
    result = {
        "input_file": data_file,
        "output_file": os.path.basename(result_file),
        "fix_axes": bool(fixed_axes_in or fixed_axes_out),
        "fix_order": bool(fixed_in_order or fixed_out_order),
        "vector_v": probe_vectors,
        "epochs": epochs,
        "learning_rate": lr,
        "input_axes": axes_in.detach().cpu().numpy().tolist(),
        "output_axes": axes_out.detach().cpu().numpy().tolist(),
        "input_order": in_order,
        "output_order": out_order,
        "final_loss": loss.item()
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"结果保存至 {result_file}")


if __name__ == "__main__":
    # 从项目根目录运行：python -m rotfit.models.baseline
    train_and_save("configs/baseline.json")
