#!/usr/bin/env python3
# rotfit/models/compatible.py
# 兼容性优先变体 —— 在不同 Python / PyTorch / 可选依赖环境下都能运行

import json
import os
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# 检查PyTorch版本
TORCH_VERSION = torch.__version__
TORCH_MAJOR, TORCH_MINOR = map(int, TORCH_VERSION.split('.')[:2])
print(f"检测到PyTorch版本: {TORCH_VERSION}")

# 可选依赖检查
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
    print("✓ SciPy可用")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠ SciPy不可用，将使用贪心算法")

# 兼容性导入
try:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    SCHEDULER_AVAILABLE = True
    print("✓ 学习率调度器可用")
except ImportError:
    SCHEDULER_AVAILABLE = False
    print("⚠ 学习率调度器不可用")

# -----------------------------
# 环境配置
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 根据PyTorch版本选择数据类型
if TORCH_MAJOR >= 1 and TORCH_MINOR >= 7:
    DEFAULT_DTYPE = torch.float32
    print("使用float32精度")
else:
    DEFAULT_DTYPE = torch.float64
    print("使用float64精度（兼容旧版本）")

torch.set_default_dtype(DEFAULT_DTYPE)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# -----------------------------
# 数据处理
# -----------------------------
def read_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """兼容的数据读取函数"""
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
    
    # 列识别
    cols = list(df.columns)
    in_cols = [c for c in cols if c.lower().startswith("in")]
    out_cols = [c for c in cols if c.lower().startswith("out")]
    
    if not in_cols or not out_cols:
        raise ValueError(f"无法识别输入输出列。找到的列: {cols}")
    
    print(f"输入列: {in_cols}")
    print(f"输出列: {out_cols}")
    
    # 数据转换
    try:
        in_arr = df[in_cols].to_numpy(dtype=np.float32)
        out_arr = df[out_cols].to_numpy(dtype=np.float32)
    except Exception as e:
        raise ValueError(f"数据转换失败: {e}")
    
    # 数据验证
    if np.any(np.isnan(in_arr)) or np.any(np.isnan(out_arr)):
        warnings.warn("数据中包含NaN值，将被处理")
        in_arr = np.nan_to_num(in_arr, nan=0.0)
        out_arr = np.nan_to_num(out_arr, nan=0.0)
    
    # 角度转换
    in_rad = np.deg2rad(in_arr)
    out_rad = np.deg2rad(out_arr)
    
    # 角度归一化
    in_rad = ((in_rad + np.pi) % (2 * np.pi)) - np.pi
    out_rad = ((out_rad + np.pi) % (2 * np.pi)) - np.pi
    
    print(f"数据加载完成: {in_rad.shape[0]} 样本")
    
    return in_rad, out_rad, in_cols, out_cols

# -----------------------------
# 兼容的Sinkhorn算法
# -----------------------------
def sinkhorn_compatible(log_alpha: torch.Tensor, n_iters: int = 50, temp: float = 1.0) -> torch.Tensor:
    """兼容版本的Sinkhorn算法"""
    eps = 1e-8
    
    # 数值稳定的exp
    log_alpha_scaled = log_alpha / temp
    max_val = torch.max(log_alpha_scaled)
    K = torch.exp(log_alpha_scaled - max_val)
    
    # 迭代归一化
    for i in range(n_iters):
        # 行归一化
        row_sum = K.sum(dim=1, keepdim=True)
        K = K / (row_sum + eps)
        
        # 列归一化
        col_sum = K.sum(dim=0, keepdim=True)
        K = K / (col_sum + eps)
        
        # 简单收敛检查
        if i > 20 and i % 10 == 0:
            # 检查是否接近双随机矩阵
            row_diff = torch.abs(K.sum(dim=1) - 1.0).max()
            col_diff = torch.abs(K.sum(dim=0) - 1.0).max()
            if row_diff < 1e-4 and col_diff < 1e-4:
                break
    
    return K

# -----------------------------
# 兼容的Rodrigues公式
# -----------------------------
def rodrigues_compatible(axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """兼容版本的Rodrigues公式"""
    B = axes.shape[0]
    device = axes.device
    
    # 轴归一化
    axes = axes / (axes.norm(dim=-1, keepdim=True) + 1e-8)
    
    # 三角函数
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)
    one_minus_cos = 1.0 - cos_theta
    
    # 轴分量
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    
    # 构建旋转矩阵
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
# 置换处理
# -----------------------------
def softperm_to_perm_compatible(P: np.ndarray) -> List[int]:
    """兼容版本的置换转换"""
    n = P.shape[0]
    
    if SCIPY_AVAILABLE:
        try:
            row_ind, col_ind = linear_sum_assignment(-P)
            return col_ind.tolist()
        except Exception:
            print("匈牙利算法失败，使用贪心算法")
    
    # 贪心算法
    perm = [-1] * n
    used_cols = set()
    
    # 按概率排序
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
    
    # 处理未分配
    remaining = [c for c in range(n) if c not in used_cols]
    for i in range(n):
        if perm[i] == -1:
            perm[i] = remaining.pop(0) if remaining else 0
    
    return perm

# -----------------------------
# 兼容的模型
# -----------------------------
class RotationFitterCompatible(torch.nn.Module):
    """兼容版本的旋转拟合器"""
    
    def __init__(self, n_in: int, n_out: int, mode: str = "continuous",
                 fixed_axes_in: Optional[Dict[str, List[float]]] = None,
                 fixed_axes_out: Optional[Dict[str, List[float]]] = None,
                 fixed_in_order: Optional[Dict[str, int]] = None,
                 fixed_out_order: Optional[Dict[str, int]] = None):
        super().__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.mode = mode
        
        # 参数初始化
        if mode == "continuous":
            self.in_axes_param = torch.nn.Parameter(torch.randn(n_in, 3) * 0.5)
            self.out_axes_param = torch.nn.Parameter(torch.randn(n_out, 3) * 0.5)
        else:
            self.in_axes_logits = torch.nn.Parameter(torch.randn(n_in, 6) * 0.1)
            self.out_axes_logits = torch.nn.Parameter(torch.randn(n_out, 6) * 0.1)
        
        self.in_order_logits = torch.nn.Parameter(torch.randn(n_in, n_in) * 0.1)
        self.out_order_logits = torch.nn.Parameter(torch.randn(n_out, n_out) * 0.1)
        
        # 轴映射
        self.register_buffer("axis_map", torch.tensor([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]
        ], dtype=DEFAULT_DTYPE))
        
        # 约束
        self.fixed_axes_in = fixed_axes_in or {}
        self.fixed_axes_out = fixed_axes_out or {}
        self.fixed_in_order = fixed_in_order or {}
        self.fixed_out_order = fixed_out_order or {}
    
    def forward(self, in_angles: torch.Tensor, out_angles: torch.Tensor,
                probe_vectors: torch.Tensor, sinkhorn_iters: int = 50, 
                sinkhorn_temp: float = 0.5) -> Dict[str, torch.Tensor]:
        """兼容的前向传播"""
        B = in_angles.shape[0]
        device = in_angles.device
        
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
        
        # 获取旋转轴
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
            axes_in[idx, :] = torch.tensor(vec, device=device, dtype=axes_in.dtype)
        
        for idx_str, vec in self.fixed_axes_out.items():
            idx = int(idx_str)
            axes_out[idx, :] = torch.tensor(vec, device=device, dtype=axes_out.dtype)
        
        # 归一化
        axes_in = axes_in / (axes_in.norm(dim=-1, keepdim=True) + 1e-8)
        axes_out = axes_out / (axes_out.norm(dim=-1, keepdim=True) + 1e-8)
        
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

# -----------------------------
# 兼容的训练器
# -----------------------------
class CompatibleTrainer:
    """兼容版本的训练器"""
    
    def __init__(self, model: RotationFitterCompatible, config: Dict):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # 优化器
        lr = config['training']['learning_rate']
        weight_decay = config['training'].get('weight_decay', 1e-4)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器（兼容版本）
        if SCHEDULER_AVAILABLE:
            try:
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=500,
                    verbose=True
                )
                print("✓ 使用ReduceLROnPlateau调度器")
            except TypeError:
                # 旧版本不支持verbose参数
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=500
                )
                print("✓ 使用ReduceLROnPlateau调度器（兼容模式）")
        else:
            self.scheduler = None
            print("⚠ 不使用学习率调度器")
        
        # 训练参数
        self.epochs = config['training']['epochs']
        self.sinkhorn_iters = config['training'].get('sinkhorn_iters', 50)
        self.sinkhorn_temp = config['training'].get('sinkhorn_temp', 0.5)
        
        # 早停
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = config['training'].get('early_stop_patience', 1000)
        
        # 历史记录
        self.history = {'loss': [], 'lr': []}
    
    def train_epoch(self, in_angles: torch.Tensor, out_angles: torch.Tensor, 
                   probe_vectors: torch.Tensor) -> float:
        """训练一个epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.model(in_angles, out_angles, probe_vectors, 
                           self.sinkhorn_iters, self.sinkhorn_temp)
        
        # 计算损失
        loss = F.mse_loss(outputs['v_in'], outputs['v_out'])
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, in_angles: torch.Tensor, out_angles: torch.Tensor, 
             probe_vectors: torch.Tensor) -> Dict:
        """完整训练过程"""
        print("开始训练...")
        print(f"数据形状: in_angles={in_angles.shape}, out_angles={out_angles.shape}")
        print(f"探测向量数量: {probe_vectors.shape[0]}")
        
        for epoch in range(self.epochs):
            loss = self.train_epoch(in_angles, out_angles, probe_vectors)
            self.history['loss'].append(loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 学习率调度
            if self.scheduler is not None:
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
        
        return {'history': self.history, 'best_loss': self.best_loss}

# -----------------------------
# 主函数
# -----------------------------
def train_compatible(config_file: str = "config.json"):
    """兼容版本的训练函数"""
    print("=== 启动兼容版旋转拟合器 ===")
    
    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 读取数据
    data_file = config["data"]["file"]
    in_angles_rad, out_angles_rad, in_cols, out_cols = read_dataset(data_file)
    
    N, n_in = in_angles_rad.shape
    _, n_out = out_angles_rad.shape
    
    # 转换为tensor
    in_angles = torch.tensor(in_angles_rad, dtype=DEFAULT_DTYPE, device=DEVICE)
    out_angles = torch.tensor(out_angles_rad, dtype=DEFAULT_DTYPE, device=DEVICE)
    
    # 探测向量
    probe_vectors_list = config["rotation"].get("vectors", [[1, 0, 0]])
    probe_vectors = torch.tensor(probe_vectors_list, dtype=DEFAULT_DTYPE, device=DEVICE)
    
    print(f"探测向量: {len(probe_vectors_list)} 个")
    
    # 创建模型
    mode = config["rotation"].get("mode", "continuous")
    fixed_axes_in = config["rotation"].get("fixed_axes", {}).get("in", {})
    fixed_axes_out = config["rotation"].get("fixed_axes", {}).get("out", {})
    fixed_in_order = config["rotation"].get("fixed_order", {}).get("in", {})
    fixed_out_order = config["rotation"].get("fixed_order", {}).get("out", {})
    
    model = RotationFitterCompatible(
        n_in, n_out, mode, fixed_axes_in, fixed_axes_out,
        fixed_in_order, fixed_out_order
    ).to(DEVICE)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练
    trainer = CompatibleTrainer(model, config)
    results = trainer.train(in_angles, out_angles, probe_vectors)
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        final_outputs = model(in_angles, out_angles, probe_vectors)
    
    # 处理结果
    P_in_np = final_outputs['P_in'].cpu().numpy()
    P_out_np = final_outputs['P_out'].cpu().numpy()
    P_in_np = np.nan_to_num(P_in_np, nan=1e-6, posinf=1e6, neginf=-1e6)
    P_out_np = np.nan_to_num(P_out_np, nan=1e-6, posinf=1e6, neginf=-1e6)
    
    in_order = softperm_to_perm_compatible(P_in_np)
    out_order = softperm_to_perm_compatible(P_out_np)
    
    # 保存结果
    result = {
        'config_info': {
            'input_file': data_file,
            'mode': mode,
            'pytorch_version': TORCH_VERSION,
            'device': str(DEVICE),
            'dtype': str(DEFAULT_DTYPE)
        },
        'data_info': {
            'input_columns': in_cols,
            'output_columns': out_cols,
            'total_samples': N,
            'n_in': n_in,
            'n_out': n_out
        },
        'training_results': {
            'epochs_completed': len(results['history']['loss']),
            'best_loss': results['best_loss'],
            'final_loss': results['history']['loss'][-1],
            'loss_history': results['history']['loss'][-100:]  # 最后100个点
        },
        'model_outputs': {
            'input_axes': final_outputs['axes_in'].cpu().numpy().tolist(),
            'output_axes': final_outputs['axes_out'].cpu().numpy().tolist(),
            'input_order': in_order,
            'output_order': out_order
        },
        'compatibility_info': {
            'scipy_available': SCIPY_AVAILABLE,
            'scheduler_available': SCHEDULER_AVAILABLE,
            'used_scheduler': trainer.scheduler is not None
        }
    }
    
    # 保存文件
    result_file = config["output"].get("result_file", "result_compatible.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 训练完成 ===")
    print(f"结果文件: {result_file}")
    print(f"最佳损失: {results['best_loss']:.6e}")
    print(f"完成轮数: {len(results['history']['loss'])}")
    print(f"兼容性: PyTorch {TORCH_VERSION}, SciPy {'✓' if SCIPY_AVAILABLE else '✗'}")
    
    return result

if __name__ == "__main__":
    train_compatible("configs/compatible.json")
