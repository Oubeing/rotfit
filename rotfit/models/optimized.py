#!/usr/bin/env python3
# rotfit/models/optimized.py
# 性能优先变体 —— float32 + 批量探针 + 稳定 Sinkhorn + AdamW/早停/梯度裁剪

import json
import os
from typing import List, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Optional: 使用匈牙利算法确定置换
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# -----------------------------
# 配置和常量
# -----------------------------
class Config:
    DEFAULT_DTYPE = torch.float32  # 改用float32提升性能
    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPSILON = 1e-8  # 数值稳定性常量
    SINKHORN_EPSILON = 1e-10
    
# 设置默认环境
torch.set_default_dtype(Config.DEFAULT_DTYPE)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# -----------------------------
# 优化的数据读取
# -----------------------------
def read_dataset(path: str, normalize_angles: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    优化的数据读取函数
    Args:
        path: 数据文件路径
        normalize_angles: 是否将角度归一化到[-π, π]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    
    # 优化文件读取
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"无法读取文件 {path}: {e}")
    
    # 提取列名
    cols = list(df.columns)
    in_cols = [c for c in cols if c.lower().startswith("in")]
    out_cols = [c for c in cols if c.lower().startswith("out")]
    
    if len(in_cols) == 0 or len(out_cols) == 0:
        raise ValueError("数据文件必须包含以 'in' 和 'out' 开头的列")
    
    # 数据转换和验证
    try:
        in_arr = df[in_cols].to_numpy(dtype=np.float32)
        out_arr = df[out_cols].to_numpy(dtype=np.float32)
    except Exception as e:
        raise ValueError(f"数据转换失败，请检查数据格式: {e}")
    
    # 检查NaN值
    if np.any(np.isnan(in_arr)) or np.any(np.isnan(out_arr)):
        warnings.warn("数据中包含NaN值，将被自动处理")
        in_arr = np.nan_to_num(in_arr, nan=0.0)
        out_arr = np.nan_to_num(out_arr, nan=0.0)
    
    # 角度转换和归一化
    in_rad = np.deg2rad(in_arr)
    out_rad = np.deg2rad(out_arr)
    
    if normalize_angles:
        in_rad = ((in_rad + np.pi) % (2 * np.pi)) - np.pi
        out_rad = ((out_rad + np.pi) % (2 * np.pi)) - np.pi
    
    return in_rad, out_rad, in_cols, out_cols

# -----------------------------
# 优化的Sinkhorn算法
# -----------------------------
def sinkhorn_stabilized(log_alpha: torch.Tensor, n_iters: int = 50, temp: float = 1.0, 
                       eps: float = Config.SINKHORN_EPSILON) -> torch.Tensor:
    """
    数值稳定的Sinkhorn算法实现
    """
    # 数值稳定的exp操作
    log_alpha_scaled = log_alpha / temp
    max_val = torch.max(log_alpha_scaled)
    K = torch.exp(log_alpha_scaled - max_val)  # 避免overflow
    
    # 早停机制
    prev_K = None
    for i in range(n_iters):
        # 行归一化
        row_sum = K.sum(dim=1, keepdim=True)
        K = K / (row_sum + eps)
        
        # 列归一化
        col_sum = K.sum(dim=0, keepdim=True)
        K = K / (col_sum + eps)
        
        # 收敛检查
        if prev_K is not None and i % 10 == 0:
            diff = torch.max(torch.abs(K - prev_K))
            if diff < 1e-6:
                break
        prev_K = K.clone() if i % 10 == 0 else prev_K
    
    return K

# -----------------------------
# 优化的Rodrigues公式
# -----------------------------
def rodrigues_batch_optimized(axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    优化的批量Rodrigues公式实现
    Args:
        axes: [B, 3] 归一化的旋转轴
        angles: [B] 旋转角度
    Returns:
        [B, 3, 3] 旋转矩阵
    """
    B = axes.shape[0]
    device = axes.device
    
    # 确保轴已归一化
    axes = axes / (axes.norm(dim=-1, keepdim=True) + Config.EPSILON)
    
    # 预计算三角函数
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)
    one_minus_cos = 1.0 - cos_theta
    
    # 轴分量
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    
    # 构建旋转矩阵 - 向量化计算
    R = torch.zeros(B, 3, 3, device=device, dtype=axes.dtype)
    
    # 对角线元素
    R[:, 0, 0] = cos_theta + x * x * one_minus_cos
    R[:, 1, 1] = cos_theta + y * y * one_minus_cos
    R[:, 2, 2] = cos_theta + z * z * one_minus_cos
    
    # 非对角线元素
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

# -----------------------------
# 置换处理优化
# -----------------------------
def softperm_to_perm_optimized(P: np.ndarray) -> List[int]:
    """优化的软置换到硬置换转换"""
    n = P.shape[0]
    
    if SCIPY_AVAILABLE:
        # 使用匈牙利算法
        try:
            row_ind, col_ind = linear_sum_assignment(-P)  # 最大化分配
            return col_ind.tolist()
        except Exception:
            pass  # 回退到贪心算法
    
    # 改进的贪心算法
    perm = [-1] * n
    used_cols = set()
    
    # 按概率值排序
    candidates = []
    for i in range(n):
        for j in range(n):
            candidates.append((P[i, j], i, j))
    candidates.sort(reverse=True)
    
    # 贪心分配
    for _, i, j in candidates:
        if perm[i] == -1 and j not in used_cols:
            perm[i] = j
            used_cols.add(j)
        if len(used_cols) == n:
            break
    
    # 处理未分配的情况
    remaining = [c for c in range(n) if c not in used_cols]
    for i in range(n):
        if perm[i] == -1:
            perm[i] = remaining.pop(0) if remaining else 0
    
    return perm

# -----------------------------
# 优化的旋转拟合模型
# -----------------------------
class RotationFitterOptimized(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, mode: str = "continuous",
                 fixed_axes_in: Optional[Dict[str, List[float]]] = None,
                 fixed_axes_out: Optional[Dict[str, List[float]]] = None,
                 fixed_in_order: Optional[Dict[str, int]] = None,
                 fixed_out_order: Optional[Dict[str, int]] = None,
                 device: Optional[torch.device] = None):
        super().__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.mode = mode
        self.device = device or Config.DEFAULT_DEVICE
        
        # 改进的参数初始化
        if mode == "continuous":
            # Xavier初始化
            std = np.sqrt(2.0 / 3)  # 3是输出维度
            self.in_axes_param = torch.nn.Parameter(torch.randn(n_in, 3) * std)
            self.out_axes_param = torch.nn.Parameter(torch.randn(n_out, 3) * std)
        else:
            # 离散模式：使用更好的初始化
            self.in_axes_logits = torch.nn.Parameter(torch.randn(n_in, 6) * 0.1)
            self.out_axes_logits = torch.nn.Parameter(torch.randn(n_out, 6) * 0.1)
        
        # 顺序参数：使用更小的初始化避免过早收敛
        self.in_order_logits = torch.nn.Parameter(torch.randn(n_in, n_in) * 0.05)
        self.out_order_logits = torch.nn.Parameter(torch.randn(n_out, n_out) * 0.05)
        
        # 基础轴映射
        self.register_buffer("axis_map", torch.tensor([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]
        ], dtype=Config.DEFAULT_DTYPE))
        
        # 固定约束
        self.fixed_axes_in = fixed_axes_in or {}
        self.fixed_axes_out = fixed_axes_out or {}
        self.fixed_in_order = fixed_in_order or {}
        self.fixed_out_order = fixed_out_order or {}
        
        # 缓存变量
        self._cached_P_in = None
        self._cached_P_out = None
        self._cache_valid = False

    def _apply_fixed_constraints(self):
        """应用固定约束"""
        with torch.no_grad():
            # 固定顺序约束
            for pos_str, idx in self.fixed_in_order.items():
                pos = int(pos_str)
                self.in_order_logits[pos, :] = -50.0
                self.in_order_logits[pos, int(idx)] = 50.0
            
            for pos_str, idx in self.fixed_out_order.items():
                pos = int(pos_str)
                self.out_order_logits[pos, :] = -50.0
                self.out_order_logits[pos, int(idx)] = 50.0

    def _get_rotation_axes(self, P_in: torch.Tensor, P_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取旋转轴"""
        if self.mode == "continuous":
            axes_in = P_in @ self.in_axes_param
            axes_out = P_out @ self.out_axes_param
        else:
            # 使用更低温度使选择更确定，然后强制离散化
            temp = 0.1
            axes_in_soft = P_in @ (F.softmax(self.in_axes_logits / temp, dim=-1) @ self.axis_map)
            axes_out_soft = P_out @ (F.softmax(self.out_axes_logits / temp, dim=-1) @ self.axis_map)
            
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
                
                axes_in = discretize_axes(axes_in_soft, self.axis_map)
                axes_out = discretize_axes(axes_out_soft, self.axis_map)
        
        # 应用固定轴约束
        for idx_str, vec in self.fixed_axes_in.items():
            idx = int(idx_str)
            axes_in[idx, :] = torch.tensor(vec, device=self.device, dtype=axes_in.dtype)
        
        for idx_str, vec in self.fixed_axes_out.items():
            idx = int(idx_str)
            axes_out[idx, :] = torch.tensor(vec, device=self.device, dtype=axes_out.dtype)
        
        # 归一化
        axes_in = axes_in / (axes_in.norm(dim=-1, keepdim=True) + Config.EPSILON)
        axes_out = axes_out / (axes_out.norm(dim=-1, keepdim=True) + Config.EPSILON)
        
        return axes_in, axes_out

    def forward(self, in_angles: torch.Tensor, out_angles: torch.Tensor,
                probe_vectors: torch.Tensor, sinkhorn_iters: int = 50, 
                sinkhorn_temp: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        优化的前向传播
        Args:
            in_angles: [B, n_in] 输入角度
            out_angles: [B, n_out] 输出角度  
            probe_vectors: [n_probes, 3] 探测向量
            sinkhorn_iters: Sinkhorn迭代次数
            sinkhorn_temp: Sinkhorn温度
        Returns:
            包含所有输出的字典
        """
        B = in_angles.shape[0]
        n_probes = probe_vectors.shape[0]
        
        # 应用固定约束
        self._apply_fixed_constraints()
        
        # 计算置换矩阵
        P_in = sinkhorn_stabilized(self.in_order_logits, sinkhorn_iters, sinkhorn_temp)
        P_out = sinkhorn_stabilized(self.out_order_logits, sinkhorn_iters, sinkhorn_temp)
        
        # 获取旋转轴
        axes_in, axes_out = self._get_rotation_axes(P_in, P_out)
        
        # 批量计算所有旋转矩阵
        R_in = torch.eye(3, device=self.device, dtype=in_angles.dtype).unsqueeze(0).repeat(B, 1, 1)
        for i in range(self.n_in):
            axis_batch = axes_in[i].unsqueeze(0).expand(B, -1)
            angle_batch = in_angles[:, i]
            R_step = rodrigues_batch_optimized(axis_batch, angle_batch)
            R_in = R_step @ R_in
        
        R_out = torch.eye(3, device=self.device, dtype=out_angles.dtype).unsqueeze(0).repeat(B, 1, 1)
        for i in range(self.n_out):
            axis_batch = axes_out[i].unsqueeze(0).expand(B, -1)
            angle_batch = out_angles[:, i]
            R_step = rodrigues_batch_optimized(axis_batch, angle_batch)
            R_out = R_step @ R_out
        
        # 批量处理所有探测向量
        probe_batch = probe_vectors.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)  # [B, 3, n_probes]
        
        v_in = R_in @ probe_batch  # [B, 3, n_probes]
        v_out = R_out @ probe_batch  # [B, 3, n_probes]
        
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

# -----------------------------
# 优化的训练器
# -----------------------------
class RotationTrainer:
    def __init__(self, model: RotationFitterOptimized, config: Dict):
        self.model = model
        self.config = config
        
        # 优化器设置
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-4)
        )
        
        # 学习率调度器
        try:
            # 尝试使用verbose参数（新版本PyTorch）
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=500,
                verbose=True
            )
        except TypeError:
            # 兼容旧版本PyTorch，不使用verbose参数
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=500
            )
        
        # 训练参数
        self.epochs = config['training']['epochs']
        self.sinkhorn_iters = config['training'].get('sinkhorn_iters', 50)
        self.sinkhorn_temp = config['training'].get('sinkhorn_temp', 0.5)
        
        # 早停
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = config['training'].get('early_stop_patience', 1000)

    def compute_loss(self, v_in: torch.Tensor, v_out: torch.Tensor, 
                    loss_type: str = 'mse') -> torch.Tensor:
        """计算损失函数"""
        if loss_type == 'mse':
            return F.mse_loss(v_in, v_out)
        elif loss_type == 'huber':
            return F.huber_loss(v_in, v_out)
        else:
            return F.mse_loss(v_in, v_out)

    def train_epoch(self, in_angles: torch.Tensor, out_angles: torch.Tensor, 
                   probe_vectors: torch.Tensor) -> float:
        """训练一个epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.model(in_angles, out_angles, probe_vectors, 
                           self.sinkhorn_iters, self.sinkhorn_temp)
        
        # 计算损失
        loss = self.compute_loss(outputs['v_in'], outputs['v_out'])
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()

    def train(self, in_angles: torch.Tensor, out_angles: torch.Tensor, 
             probe_vectors: torch.Tensor) -> Dict:
        """完整训练过程"""
        history = {'loss': []}
        
        print(f"开始训练，设备: {self.model.device}")
        print(f"数据形状: in_angles={in_angles.shape}, out_angles={out_angles.shape}")
        print(f"探测向量数量: {probe_vectors.shape[0]}")
        
        for epoch in range(self.epochs):
            loss = self.train_epoch(in_angles, out_angles, probe_vectors)
            history['loss'].append(loss)
            
            # 学习率调度
            self.scheduler.step(loss)
            
            # 早停检查
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stop_patience:
                print(f"早停于epoch {epoch}, 最佳损失: {self.best_loss:.6e}")
                break
            
            # 进度输出
            if epoch % 500 == 0 or epoch == self.epochs - 1:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"[{epoch}/{self.epochs}] Loss={loss:.6e}, LR={lr:.2e}")
        
        return history

# -----------------------------
# 主训练函数
# -----------------------------
def train_and_save_optimized(config_file: str):
    """优化的训练和保存函数"""
    # 加载配置
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 设置设备
    device_name = config['training'].get('device', 'auto')
    if device_name == 'auto':
        device = Config.DEFAULT_DEVICE
    else:
        device = torch.device(device_name)
    
    print(f"使用设备: {device}")
    
    # 读取数据
    data_file = config["data"]["file"]
    in_angles_rad, out_angles_rad, in_cols, out_cols = read_dataset(data_file)
    
    N, n_in = in_angles_rad.shape
    _, n_out = out_angles_rad.shape
    
    # 转换为tensor
    in_angles = torch.tensor(in_angles_rad, dtype=Config.DEFAULT_DTYPE, device=device)
    out_angles = torch.tensor(out_angles_rad, dtype=Config.DEFAULT_DTYPE, device=device)
    
    # 探测向量
    probe_vectors_list = config["rotation"].get("vectors", [[1, 0, 0]])
    probe_vectors = torch.tensor(probe_vectors_list, dtype=Config.DEFAULT_DTYPE, device=device)
    
    # 创建模型
    mode = config["rotation"].get("mode", "continuous")
    fixed_axes_in = config["rotation"].get("fixed_axes", {}).get("in", {})
    fixed_axes_out = config["rotation"].get("fixed_axes", {}).get("out", {})
    fixed_in_order = config["rotation"].get("fixed_order", {}).get("in", {})
    fixed_out_order = config["rotation"].get("fixed_order", {}).get("out", {})
    
    model = RotationFitterOptimized(
        n_in, n_out, mode, fixed_axes_in, fixed_axes_out,
        fixed_in_order, fixed_out_order, device
    ).to(device)
    
    # 训练
    trainer = RotationTrainer(model, config)
    history = trainer.train(in_angles, out_angles, probe_vectors)
    
    # 获取最终结果
    model.eval()
    with torch.no_grad():
        outputs = model(in_angles, out_angles, probe_vectors, 
                       trainer.sinkhorn_iters, trainer.sinkhorn_temp)
    
    # 处理置换矩阵
    P_in_np = outputs['P_in'].cpu().numpy()
    P_out_np = outputs['P_out'].cpu().numpy()
    P_in_np = np.nan_to_num(P_in_np, nan=1e-6, posinf=1e6, neginf=-1e6)
    P_out_np = np.nan_to_num(P_out_np, nan=1e-6, posinf=1e6, neginf=-1e6)
    
    in_order = softperm_to_perm_optimized(P_in_np)
    out_order = softperm_to_perm_optimized(P_out_np)
    
    # 保存结果
    result = {
        "input_file": data_file,
        "mode": mode,
        "fix_axes": bool(fixed_axes_in or fixed_axes_out),
        "fix_order": bool(fixed_in_order or fixed_out_order),
        "probe_vectors": probe_vectors_list,
        "training_config": config['training'],
        "input_axes": outputs['axes_in'].cpu().numpy().tolist(),
        "output_axes": outputs['axes_out'].cpu().numpy().tolist(),
        "input_order": in_order,
        "output_order": out_order,
        "final_loss": history['loss'][-1],
        "best_loss": trainer.best_loss,
        "training_history": {
            "loss": history['loss'][-100:]  # 只保存最后100个点
        }
    }
    
    result_file = config["output"].get("result_file", "result_optimized.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"优化训练完成！结果保存至 {result_file}")
    print(f"最终损失: {result['final_loss']:.6e}")
    print(f"最佳损失: {result['best_loss']:.6e}")

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    train_and_save_optimized("configs/optimized.json")
