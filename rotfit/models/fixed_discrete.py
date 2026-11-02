#!/usr/bin/env python3
# 修复discrete模式输出标准轴的版本

import json
import os
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# 使用兼容版本的基础设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)

print(f"修复版本启动，设备: {DEVICE}")

# 复用兼容版本的函数
def read_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """数据读取函数"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    
    print(f"读取数据文件: {path}")
    
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"无法读取文件 {path}: {e}")
    
    cols = list(df.columns)
    in_cols = [c for c in cols if c.lower().startswith("in")]
    out_cols = [c for c in cols if c.lower().startswith("out")]
    
    if not in_cols or not out_cols:
        raise ValueError(f"无法识别输入输出列。找到的列: {cols}")
    
    in_arr = df[in_cols].to_numpy(dtype=np.float32)
    out_arr = df[out_cols].to_numpy(dtype=np.float32)
    
    if np.any(np.isnan(in_arr)) or np.any(np.isnan(out_arr)):
        warnings.warn("数据中包含NaN值，将被处理")
        in_arr = np.nan_to_num(in_arr, nan=0.0)
        out_arr = np.nan_to_num(out_arr, nan=0.0)
    
    in_rad = np.deg2rad(in_arr)
    out_rad = np.deg2rad(out_arr)
    in_rad = ((in_rad + np.pi) % (2 * np.pi)) - np.pi
    out_rad = ((out_rad + np.pi) % (2 * np.pi)) - np.pi
    
    print(f"数据加载完成: {in_rad.shape[0]} 样本")
    return in_rad, out_rad, in_cols, out_cols

def sinkhorn_compatible(log_alpha: torch.Tensor, n_iters: int = 50, temp: float = 1.0) -> torch.Tensor:
    """Sinkhorn算法"""
    eps = 1e-8
    log_alpha_scaled = log_alpha / temp
    max_val = torch.max(log_alpha_scaled)
    K = torch.exp(log_alpha_scaled - max_val)
    
    for i in range(n_iters):
        K = K / (K.sum(dim=1, keepdim=True) + eps)
        K = K / (K.sum(dim=0, keepdim=True) + eps)
        if i > 20 and i % 10 == 0:
            row_diff = torch.abs(K.sum(dim=1) - 1.0).max()
            col_diff = torch.abs(K.sum(dim=0) - 1.0).max()
            if row_diff < 1e-4 and col_diff < 1e-4:
                break
    return K

def rodrigues_compatible(axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rodrigues公式"""
    B = axes.shape[0]
    device = axes.device
    axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)
    
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)
    one_minus_cos = 1.0 - cos_theta
    
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    
    R = torch.zeros(B, 3, 3, device=device, dtype=axes.dtype)
    R[:, 0, 0] = cos_theta + x * x * one_minus_cos
    R[:, 1, 1] = cos_theta + y * y * one_minus_cos
    R[:, 2, 2] = cos_theta + z * z * one_minus_cos
    
    xy_omc = x * y * one_minus_cos
    xz_omc = x * z * one_minus_cos
    yz_omc = y * z * one_minus_cos
    
    R[:, 0, 1] = xy_omc - z * sin_theta
    R[:, 0, 2] = xz_omc + y * sin_theta
    R[:, 1, 0] = xy_omc + z * sin_theta
    R[:, 1, 2] = yz_omc - x * sin_theta
    R[:, 2, 0] = xz_omc - y * sin_theta
    R[:, 2, 1] = yz_omc + x * sin_theta
    
    return R

# 修复的模型类
class RotationFitterFixedDiscrete(torch.nn.Module):
    """修复discrete模式的旋转拟合器"""
    
    def __init__(self, n_in: int, n_out: int, mode: str = "discrete",
                 fixed_axes_in: Optional[Dict[str, List[float]]] = None,
                 fixed_axes_out: Optional[Dict[str, List[float]]] = None,
                 fixed_in_order: Optional[Dict[str, int]] = None,
                 fixed_out_order: Optional[Dict[str, int]] = None,
                 force_discrete: bool = True):
        super().__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.mode = mode
        self.force_discrete = force_discrete  # 新参数：是否强制离散化
        
        print(f"创建模型: mode={mode}, force_discrete={force_discrete}")
        
        # 参数初始化
        if mode == "continuous":
            self.in_axes_param = torch.nn.Parameter(torch.randn(n_in, 3) * 0.5)
            self.out_axes_param = torch.nn.Parameter(torch.randn(n_out, 3) * 0.5)
        else:
            self.in_axes_logits = torch.nn.Parameter(torch.randn(n_in, 6) * 0.1)
            self.out_axes_logits = torch.nn.Parameter(torch.randn(n_out, 6) * 0.1)
        
        self.in_order_logits = torch.nn.Parameter(torch.randn(n_in, n_in) * 0.1)
        self.out_order_logits = torch.nn.Parameter(torch.randn(n_out, n_out) * 0.1)
        
        # 基础轴映射
        self.register_buffer("axis_map", torch.tensor([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]
        ], dtype=DEFAULT_DTYPE))
        
        # 约束
        self.fixed_axes_in = fixed_axes_in or {}
        self.fixed_axes_out = fixed_axes_out or {}
        self.fixed_in_order = fixed_in_order or {}
        self.fixed_out_order = fixed_out_order or {}
        
        print(f"轴映射:")
        for i, axis in enumerate(self.axis_map):
            direction = ['+X', '+Y', '+Z', '-X', '-Y', '-Z'][i]
            print(f"  {i}: [{axis[0]:.0f}, {axis[1]:.0f}, {axis[2]:.0f}] ({direction})")
    
    def _discretize_axes(self, soft_axes: torch.Tensor) -> torch.Tensor:
        """强制将软轴离散化为标准轴"""
        if not self.force_discrete:
            return soft_axes
        
        discretized = []
        for axis in soft_axes:
            # 计算与所有标准轴的余弦相似度
            similarities = F.cosine_similarity(axis.unsqueeze(0), self.axis_map, dim=1)
            best_idx = torch.argmax(similarities)
            discretized.append(self.axis_map[best_idx])
        
        return torch.stack(discretized)
    
    def forward(self, in_angles: torch.Tensor, out_angles: torch.Tensor,
                probe_vectors: torch.Tensor, sinkhorn_iters: int = 50, 
                sinkhorn_temp: float = 0.5) -> Dict[str, torch.Tensor]:
        """修复的前向传播"""
        B = in_angles.shape[0]
        device = in_angles.device
        
        print(f"\n=== Forward Pass Debug (Batch={B}) ===")
        
        # 应用固定约束
        with torch.no_grad():
            for pos_str, idx in self.fixed_in_order.items():
                pos = int(pos_str)
                self.in_order_logits[pos, :] = -50.0
                self.in_order_logits[pos, int(idx)] = 50.0
            
            for pos_str, idx in self.fixed_out_order.items():
                pos = int(pos_str)
                self.out_order_logits[pos, :] = -50.0
                self.out_order_logits[pos, int(idx)] = 50.0
        
        # 计算置换矩阵
        P_in = sinkhorn_compatible(self.in_order_logits, sinkhorn_iters, sinkhorn_temp)
        P_out = sinkhorn_compatible(self.out_order_logits, sinkhorn_iters, sinkhorn_temp)
        
        print("Sinkhorn置换矩阵 P_in:")
        print(P_in.detach().numpy())
        
        # 获取旋转轴
        if self.mode == "continuous":
            axes_in = P_in @ self.in_axes_param
            axes_out = P_out @ self.out_axes_param
        else:
            print("\n=== Discrete轴生成过程 ===")
            
            # 显示logits
            print("轴logits (in):")
            print(self.in_axes_logits.detach().numpy())
            
            # 软选择
            softmax_weights_in = F.softmax(self.in_axes_logits, dim=-1)
            softmax_weights_out = F.softmax(self.out_axes_logits, dim=-1)
            
            print("Softmax权重 (in):")
            for i, weights in enumerate(softmax_weights_in):
                max_idx = torch.argmax(weights)
                max_prob = weights[max_idx].item()
                direction = ['+X', '+Y', '+Z', '-X', '-Y', '-Z'][max_idx]
                print(f"  轴{i}: 最可能={direction} (概率:{max_prob:.3f})")
                print(f"       权重: {weights.detach().numpy()}")
            
            # 原始方法：软选择
            axes_in_soft = P_in @ (softmax_weights_in @ self.axis_map)
            axes_out_soft = P_out @ (softmax_weights_out @ self.axis_map)
            
            print("软选择后的轴 (in):")
            for i, axis in enumerate(axes_in_soft):
                print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
            
            # 修复：强制离散化
            if self.force_discrete:
                axes_in = self._discretize_axes(axes_in_soft)
                axes_out = self._discretize_axes(axes_out_soft)
                
                print("强制离散化后的轴 (in):")
                for i, axis in enumerate(axes_in):
                    print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
            else:
                axes_in = axes_in_soft
                axes_out = axes_out_soft
        
        # 应用固定轴约束
        for idx_str, vec in self.fixed_axes_in.items():
            idx = int(idx_str)
            axes_in[idx, :] = torch.tensor(vec, device=device, dtype=axes_in.dtype)
            print(f"应用固定轴约束: 轴{idx} = {vec}")
        
        for idx_str, vec in self.fixed_axes_out.items():
            idx = int(idx_str)
            axes_out[idx, :] = torch.tensor(vec, device=device, dtype=axes_out.dtype)
        
        # 归一化
        axes_in = axes_in / (axes_in.norm(dim=-1, keepdim=True) + 1e-8)
        axes_out = axes_out / (axes_out.norm(dim=-1, keepdim=True) + 1e-8)
        
        print("最终输出轴 (in):")
        for i, axis in enumerate(axes_in):
            print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
        
        # 计算旋转矩阵
        R_in = torch.eye(3, device=device, dtype=in_angles.dtype).unsqueeze(0).repeat(B, 1, 1)
        for i in range(self.n_in):
            axis_batch = axes_in[i].unsqueeze(0).expand(B, -1)
            angle_batch = in_angles[:, i]
            R_step = rodrigues_compatible(axis_batch, angle_batch)
            R_in = torch.bmm(R_step, R_in)
        
        R_out = torch.eye(3, device=device, dtype=out_angles.dtype).unsqueeze(0).repeat(B, 1, 1)
        for i in range(self.n_out):
            axis_batch = axes_out[i].unsqueeze(0).expand(B, -1)
            angle_batch = out_angles[:, i]
            R_step = rodrigues_compatible(axis_batch, angle_batch)
            R_out = torch.bmm(R_step, R_out)
        
        # 处理探测向量
        if probe_vectors.dim() == 2:
            probe_batch = probe_vectors.t().unsqueeze(0).expand(B, -1, -1)
        else:
            probe_batch = probe_vectors
        
        v_in = torch.bmm(R_in, probe_batch)
        v_out = torch.bmm(R_out, probe_batch)
        
        return {
            'v_in': v_in,
            'v_out': v_out,
            'R_in': R_in,
            'R_out': R_out,
            'P_in': P_in,
            'P_out': P_out,
            'axes_in': axes_in,
            'axes_out': axes_out
        }

def train_fixed_discrete(config_file: str = "config.json", force_discrete: bool = True):
    """训练修复版本"""
    print("=== 启动修复版旋转拟合器 ===")
    print(f"强制离散化: {force_discrete}")
    
    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 读取数据
    data_file = config["data"]["file"]
    in_angles_rad, out_angles_rad, in_cols, out_cols = read_dataset(data_file)
    
    N, n_in = in_angles_rad.shape
    _, n_out = out_angles_rad.shape
    
    # 只取前10个样本用于调试
    debug_samples = min(10, N)
    in_angles = torch.tensor(in_angles_rad[:debug_samples], dtype=DEFAULT_DTYPE, device=DEVICE)
    out_angles = torch.tensor(out_angles_rad[:debug_samples], dtype=DEFAULT_DTYPE, device=DEVICE)
    
    # 探测向量
    probe_vectors_list = config["rotation"].get("vectors", [[1, 0, 0]])
    probe_vectors = torch.tensor(probe_vectors_list, dtype=DEFAULT_DTYPE, device=DEVICE)
    
    print(f"调试模式: 使用 {debug_samples} 个样本")
    print(f"探测向量: {len(probe_vectors_list)} 个")
    
    # 创建模型
    mode = config["rotation"].get("mode", "discrete")
    fixed_axes_in = config["rotation"].get("fixed_axes", {}).get("in", {})
    fixed_axes_out = config["rotation"].get("fixed_axes", {}).get("out", {})
    fixed_in_order = config["rotation"].get("fixed_order", {}).get("in", {})
    fixed_out_order = config["rotation"].get("fixed_order", {}).get("out", {})
    
    model = RotationFitterFixedDiscrete(
        n_in, n_out, mode, fixed_axes_in, fixed_axes_out,
        fixed_in_order, fixed_out_order, force_discrete
    ).to(DEVICE)
    
    # 简单训练几步用于调试
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"\n=== 训练前的输出 ===")
    model.eval()
    with torch.no_grad():
        initial_outputs = model(in_angles, out_angles, probe_vectors)
    
    print(f"\n=== 开始训练 ===")
    for epoch in range(5):  # 只训练5步用于调试
        model.train()
        optimizer.zero_grad()
        
        outputs = model(in_angles, out_angles, probe_vectors)
        loss = F.mse_loss(outputs['v_in'], outputs['v_out'])
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: Loss = {loss.item():.6e}")
    
    # 最终输出
    print(f"\n=== 最终结果 ===")
    model.eval()
    with torch.no_grad():
        final_outputs = model(in_angles, out_angles, probe_vectors)
    
    # 显示最终轴
    final_axes_in = final_outputs['axes_in'].cpu().numpy()
    final_axes_out = final_outputs['axes_out'].cpu().numpy()
    
    print("\n最终输入轴:")
    for i, axis in enumerate(final_axes_in):
        print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
        # 检查是否接近标准轴
        standard_axes = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]])
        similarities = np.array([np.dot(axis, std_axis) for std_axis in standard_axes])
        best_match_idx = np.argmax(np.abs(similarities))
        best_similarity = similarities[best_match_idx]
        direction = ['+X', '+Y', '+Z', '-X', '-Y', '-Z'][best_match_idx]
        print(f"       最接近: {direction} (相似度: {best_similarity:.6f})")
    
    print("\n最终输出轴:")
    for i, axis in enumerate(final_axes_out):
        print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
        similarities = np.array([np.dot(axis, std_axis) for std_axis in standard_axes])
        best_match_idx = np.argmax(np.abs(similarities))
        best_similarity = similarities[best_match_idx]
        direction = ['+X', '+Y', '+Z', '-X', '-Y', '-Z'][best_match_idx]
        print(f"       最接近: {direction} (相似度: {best_similarity:.6f})")
    
    return final_outputs

if __name__ == "__main__":
    print("=== 对比测试 ===")
    print("1. 测试强制离散化版本")
    result_discrete = train_fixed_discrete("configs/baseline.json", force_discrete=True)
    
    print("\n" + "="*50)
    print("2. 测试原始软选择版本")
    result_soft = train_fixed_discrete("configs/baseline.json", force_discrete=False)
