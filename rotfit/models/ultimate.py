#!/usr/bin/env python3
# rotfit/models/ultimate.py
# 工程化变体 —— torch.compile / JIT / 混合精度 / 多优化器 / 训练可视化

import json
import os
import logging
import time
import warnings
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torch.jit as jit

# 可选的TensorBoard支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("警告: TensorBoard不可用，将使用基础日志。安装方法: pip install tensorboard")

# Optional dependencies
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# -----------------------------
# 配置系统
# -----------------------------
@dataclass
class OptimizationConfig:
    """优化配置类"""
    # 数值精度
    dtype: torch.dtype = torch.float32
    epsilon: float = 1e-8
    sinkhorn_epsilon: float = 1e-10
    
    # 性能配置
    compile_model: bool = True  # PyTorch 2.0 编译
    use_mixed_precision: bool = True  # 混合精度训练
    gradient_checkpointing: bool = False  # 梯度检查点
    
    # 并行配置
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # 内存优化
    memory_efficient: bool = True
    max_batch_size: int = 10000

@dataclass 
class TrainingConfig:
    """训练配置类"""
    epochs: int = 5000
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    
    # 优化器配置
    optimizer_type: str = "adamw"  # adamw, adam, sgd, lion
    beta1: float = 0.9
    beta2: float = 0.999
    
    # 调度器配置
    scheduler_type: str = "cosine_warm"  # cosine_warm, reduce_on_plateau, none
    warmup_epochs: int = 100
    min_lr: float = 1e-6
    
    # 正则化
    gradient_clip: float = 1.0
    dropout: float = 0.0
    spectral_norm: bool = False
    
    # 早停和验证
    early_stop_patience: int = 1000
    validation_split: float = 0.1
    
    # Sinkhorn参数
    sinkhorn_iters: int = 50
    sinkhorn_temp: float = 0.5
    sinkhorn_temp_schedule: bool = True  # 温度调度
    
    # 损失函数
    loss_type: str = "mse"  # mse, huber, smooth_l1, cosine
    loss_weights: List[float] = None  # 多损失权重
    
    # 数据增强
    angle_noise_std: float = 0.0
    rotation_augment: bool = False

# 全局配置
OPT_CONFIG = OptimizationConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置环境
torch.set_default_dtype(OPT_CONFIG.dtype)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 日志配置（写入 logs/，若不存在则创建）
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rotfit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# 高级数据处理
# -----------------------------
class DataProcessor:
    """高级数据处理器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def read_dataset(self, path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """高性能数据读取"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        
        logger.info(f"读取数据文件: {path}")
        
        # 并行读取大文件
        try:
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(path, dtype=np.float32, **kwargs)
            else:
                df = pd.read_excel(path, engine="openpyxl", dtype=np.float32, **kwargs)
        except Exception as e:
            raise ValueError(f"无法读取文件 {path}: {e}")
        
        # 智能列识别
        cols = list(df.columns)
        in_cols = [c for c in cols if c.lower().startswith("in")]
        out_cols = [c for c in cols if c.lower().startswith("out")]
        
        if not in_cols or not out_cols:
            # 尝试其他模式
            in_cols = [c for c in cols if "input" in c.lower() or "in_" in c.lower()]
            out_cols = [c for c in cols if "output" in c.lower() or "out_" in c.lower()]
        
        if not in_cols or not out_cols:
            raise ValueError(f"无法识别输入输出列。找到的列: {cols}")
        
        logger.info(f"识别到输入列: {in_cols}")
        logger.info(f"识别到输出列: {out_cols}")
        
        # 高性能数据转换
        in_arr = df[in_cols].to_numpy(dtype=np.float32)
        out_arr = df[out_cols].to_numpy(dtype=np.float32)
        
        # 数据质量检查和清理
        in_arr, out_arr = self._clean_data(in_arr, out_arr)
        
        # 角度预处理
        in_rad = self._preprocess_angles(in_arr)
        out_rad = self._preprocess_angles(out_arr)
        
        logger.info(f"数据加载完成: {in_rad.shape[0]} 样本, {len(in_cols)} 输入轴, {len(out_cols)} 输出轴")
        
        return in_rad, out_rad, in_cols, out_cols
    
    def _clean_data(self, in_arr: np.ndarray, out_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据清理"""
        # NaN处理
        in_nan_mask = np.isnan(in_arr).any(axis=1)
        out_nan_mask = np.isnan(out_arr).any(axis=1)
        nan_mask = in_nan_mask | out_nan_mask
        
        if nan_mask.any():
            logger.warning(f"发现 {nan_mask.sum()} 行包含NaN，将被移除")
            in_arr = in_arr[~nan_mask]
            out_arr = out_arr[~nan_mask]
        
        # 异常值检测 (使用IQR方法)
        def remove_outliers(arr, factor=3.0):
            Q1 = np.percentile(arr, 25, axis=0)
            Q3 = np.percentile(arr, 75, axis=0)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            mask = np.all((arr >= lower) & (arr <= upper), axis=1)
            return mask
        
        in_mask = remove_outliers(in_arr)
        out_mask = remove_outliers(out_arr)
        valid_mask = in_mask & out_mask
        
        if not valid_mask.all():
            removed = (~valid_mask).sum()
            logger.warning(f"移除 {removed} 个异常样本")
            in_arr = in_arr[valid_mask]
            out_arr = out_arr[valid_mask]
        
        return in_arr, out_arr
    
    def _preprocess_angles(self, angles: np.ndarray) -> np.ndarray:
        """角度预处理"""
        # 度转弧度
        angles_rad = np.deg2rad(angles)
        
        # 角度归一化到 [-π, π]
        angles_rad = ((angles_rad + np.pi) % (2 * np.pi)) - np.pi
        
        return angles_rad

# -----------------------------
# 超高性能数值算法
# -----------------------------
@jit.script
def sinkhorn_ultimate(log_alpha: torch.Tensor, n_iters: int = 50, temp: float = 1.0, 
                     eps: float = 1e-10, early_stop_tol: float = 1e-6) -> torch.Tensor:
    """终极Sinkhorn算法 - JIT编译版本"""
    # 数值稳定的初始化
    log_alpha_scaled = log_alpha / temp
    max_val = torch.max(log_alpha_scaled, dim=-1, keepdim=True)[0]
    K = torch.exp(log_alpha_scaled - max_val)
    
    # 自适应迭代
    prev_K = K.clone()
    for i in range(n_iters):
        # 行归一化
        row_sum = K.sum(dim=-1, keepdim=True)
        K = K / (row_sum + eps)
        
        # 列归一化
        col_sum = K.sum(dim=-2, keepdim=True)
        K = K / (col_sum + eps)
        
        # 早停检查
        if i > 10 and i % 5 == 0:
            diff = torch.max(torch.abs(K - prev_K))
            if diff < early_stop_tol:
                break
            prev_K = K.clone()
    
    return K

@jit.script
def rodrigues_ultimate(axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """终极Rodrigues公式 - 完全向量化JIT版本"""
    # 输入验证和归一化
    axes = axes / (torch.norm(axes, dim=-1, keepdim=True) + 1e-8)
    
    # 预计算三角函数
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)
    one_minus_cos = 1.0 - cos_theta
    
    # 轴分量
    x, y, z = axes.unbind(-1)
    
    # 完全向量化的旋转矩阵构建
    xx, xy, xz = x * x, x * y, x * z
    yy, yz = y * y, y * z
    zz = z * z
    
    # 构建旋转矩阵
    R = torch.zeros(axes.shape[0], 3, 3, dtype=axes.dtype, device=axes.device)
    
    # 对角线
    R[:, 0, 0] = cos_theta + xx * one_minus_cos
    R[:, 1, 1] = cos_theta + yy * one_minus_cos
    R[:, 2, 2] = cos_theta + zz * one_minus_cos
    
    # 反对称部分
    R[:, 0, 1] = xy * one_minus_cos - z * sin_theta
    R[:, 0, 2] = xz * one_minus_cos + y * sin_theta
    R[:, 1, 0] = xy * one_minus_cos + z * sin_theta
    R[:, 1, 2] = yz * one_minus_cos - x * sin_theta
    R[:, 2, 0] = xz * one_minus_cos - y * sin_theta
    R[:, 2, 1] = yz * one_minus_cos + x * sin_theta
    
    return R

# -----------------------------
# 高级损失函数
# -----------------------------
class AdvancedLosses(nn.Module):
    """高级损失函数集合"""
    
    def __init__(self, loss_type: str = "mse", reduction: str = "mean"):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算损失"""
        if self.loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == "huber":
            loss = F.huber_loss(pred, target, reduction='none')
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(pred, target, reduction='none')
        elif self.loss_type == "cosine":
            # 余弦相似度损失
            pred_norm = F.normalize(pred, dim=-1)
            target_norm = F.normalize(target, dim=-1)
            loss = 1 - F.cosine_similarity(pred_norm, target_norm, dim=-1)
        elif self.loss_type == "angular":
            # 角度损失
            pred_norm = F.normalize(pred, dim=-1)
            target_norm = F.normalize(target, dim=-1)
            cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
            loss = torch.acos(torch.clamp(cos_sim, -1+1e-7, 1-1e-7))
        else:
            loss = F.mse_loss(pred, target, reduction='none')
        
        # 应用权重
        if weights is not None:
            loss = loss * weights
        
        # 归约
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# -----------------------------
# 终极旋转拟合模型
# -----------------------------
class RotationFitterUltimate(nn.Module):
    """终极旋转拟合器 - 集成所有先进技术"""
    
    def __init__(self, n_in: int, n_out: int, config: TrainingConfig,
                 fixed_axes_in: Optional[Dict[str, List[float]]] = None,
                 fixed_axes_out: Optional[Dict[str, List[float]]] = None,
                 fixed_in_order: Optional[Dict[str, int]] = None,
                 fixed_out_order: Optional[Dict[str, int]] = None,
                 mode: str = "discrete"):
        super().__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.config = config
        self.mode = mode  # "continuous" or "discrete"
        
        # 高级参数初始化
        self._init_parameters()
        
        # 约束处理
        self.fixed_axes_in = fixed_axes_in or {}
        self.fixed_axes_out = fixed_axes_out or {}
        self.fixed_in_order = fixed_in_order or {}
        self.fixed_out_order = fixed_out_order or {}
        
        # 基础轴映射
        self.register_buffer("axis_map", torch.tensor([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]
        ], dtype=OPT_CONFIG.dtype))
        
        # 损失函数
        self.loss_fn = AdvancedLosses(config.loss_type)
        
        # 缓存机制
        self._cache = {}
        self._cache_valid = False
        
        # 性能监控
        self.forward_time = 0.0
        self.forward_calls = 0
    
    def _init_parameters(self):
        """高级参数初始化"""
        if self.mode == "continuous":
            # Kaiming初始化用于连续模式
            self.in_axes_param = nn.Parameter(torch.empty(self.n_in, 3))
            self.out_axes_param = nn.Parameter(torch.empty(self.n_out, 3))
            nn.init.kaiming_normal_(self.in_axes_param, mode='fan_out')
            nn.init.kaiming_normal_(self.out_axes_param, mode='fan_out')
        else:
            # 离散模式：更智能的初始化
            self.in_axes_logits = nn.Parameter(torch.randn(self.n_in, 6) * 0.1)
            self.out_axes_logits = nn.Parameter(torch.randn(self.n_out, 6) * 0.1)
            
            # 给主要轴方向更高的初始权重
            with torch.no_grad():
                self.in_axes_logits[:, :3] += 0.5  # X, Y, Z轴
                self.out_axes_logits[:, :3] += 0.5
        
        # 顺序参数：使用正交初始化
        self.in_order_logits = nn.Parameter(torch.empty(self.n_in, self.n_in))
        self.out_order_logits = nn.Parameter(torch.empty(self.n_out, self.n_out))
        nn.init.orthogonal_(self.in_order_logits, gain=0.1)
        nn.init.orthogonal_(self.out_order_logits, gain=0.1)
        
        # 谱归一化（可选）
        if self.config.spectral_norm:
            self.in_order_logits = nn.utils.spectral_norm(self.in_order_logits)
            self.out_order_logits = nn.utils.spectral_norm(self.out_order_logits)
    
    def _apply_constraints(self):
        """应用约束条件"""
        with torch.no_grad():
            # 固定顺序约束
            for pos_str, idx in self.fixed_in_order.items():
                pos = int(pos_str)
                self.in_order_logits[pos, :] = -100.0
                self.in_order_logits[pos, int(idx)] = 100.0
            
            for pos_str, idx in self.fixed_out_order.items():
                pos = int(pos_str)
                self.out_order_logits[pos, :] = -100.0
                self.out_order_logits[pos, int(idx)] = 100.0
    
    def _get_rotation_axes(self, P_in: torch.Tensor, P_out: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取旋转轴"""
        if hasattr(self, 'in_axes_param'):  # 连续模式
            axes_in = P_in @ self.in_axes_param
            axes_out = P_out @ self.out_axes_param
        else:  # 离散模式
            # 使用低温度Gumbel-Softmax增强离散性
            temp = 0.1
            axes_in_soft = P_in @ (F.gumbel_softmax(self.in_axes_logits, tau=temp, hard=False) @ self.axis_map)
            axes_out_soft = P_out @ (F.gumbel_softmax(self.out_axes_logits, tau=temp, hard=False) @ self.axis_map)
            
            # 训练时使用软轴保持梯度，推理时离散化
            if training:
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
            axes_in[idx, :] = torch.tensor(vec, device=axes_in.device, dtype=axes_in.dtype)
        
        for idx_str, vec in self.fixed_axes_out.items():
            idx = int(idx_str)
            axes_out[idx, :] = torch.tensor(vec, device=axes_out.device, dtype=axes_out.dtype)
        
        # 高精度归一化
        axes_in = F.normalize(axes_in, dim=-1, eps=1e-8)
        axes_out = F.normalize(axes_out, dim=-1, eps=1e-8)
        
        return axes_in, axes_out
    
    def forward(self, in_angles: torch.Tensor, out_angles: torch.Tensor,
                probe_vectors: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """超高性能前向传播"""
        start_time = time.time()
        
        B = in_angles.shape[0]
        device = in_angles.device
        
        # 应用约束
        self._apply_constraints()
        
        # 动态温度调度
        if self.config.sinkhorn_temp_schedule and training:
            temp = max(0.1, self.config.sinkhorn_temp * (0.95 ** (self.forward_calls // 1000)))
        else:
            temp = self.config.sinkhorn_temp
        
        # 计算置换矩阵（JIT编译版本）
        P_in = sinkhorn_ultimate(self.in_order_logits, self.config.sinkhorn_iters, temp)
        P_out = sinkhorn_ultimate(self.out_order_logits, self.config.sinkhorn_iters, temp)
        
        # 获取旋转轴
        axes_in, axes_out = self._get_rotation_axes(P_in, P_out, training)
        
        # 超高效旋转矩阵计算
        R_in = torch.eye(3, device=device, dtype=in_angles.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
        R_out = torch.eye(3, device=device, dtype=out_angles.dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
        
        # 批量Rodrigues计算
        for i in range(self.n_in):
            axis_batch = axes_in[i].unsqueeze(0).expand(B, -1)
            angle_batch = in_angles[:, i]
            R_step = rodrigues_ultimate(axis_batch, angle_batch)
            R_in = torch.bmm(R_step, R_in)
        
        for i in range(self.n_out):
            axis_batch = axes_out[i].unsqueeze(0).expand(B, -1)
            angle_batch = out_angles[:, i]
            R_step = rodrigues_ultimate(axis_batch, angle_batch)
            R_out = torch.bmm(R_step, R_out)
        
        # 超高效探测向量处理
        if probe_vectors.dim() == 2:
            probe_batch = probe_vectors.t().unsqueeze(0).expand(B, -1, -1)  # [B, 3, n_probes]
        else:
            probe_batch = probe_vectors
        
        # 批量矩阵乘法
        v_in = torch.bmm(R_in, probe_batch)
        v_out = torch.bmm(R_out, probe_batch)
        
        # 性能统计
        self.forward_time += time.time() - start_time
        self.forward_calls += 1
        
        return {
            'v_in': v_in,
            'v_out': v_out,
            'R_in': R_in,
            'R_out': R_out,
            'P_in': P_in,
            'P_out': P_out,
            'axes_in': axes_in,
            'axes_out': axes_out,
            'temperature': temp
        }

# -----------------------------
# 终极训练器
# -----------------------------
class UltimateTrainer:
    """终极训练器 - 集成所有先进训练技术"""
    
    def __init__(self, model: RotationFitterUltimate, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # 初始化组件
        self._init_optimizer()
        self._init_scheduler()
        self._init_logging()
        
        # 混合精度训练
        if OPT_CONFIG.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'temperature': []
        }
        
        logger.info(f"初始化终极训练器: 优化器={config.optimizer_type}, 调度器={config.scheduler_type}")
        logger.info(f"混合精度训练: {self.use_amp}, 设备: {self.device}")
    
    def _init_optimizer(self):
        """初始化优化器"""
        params = self.model.parameters()
        
        if self.config.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                params, lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                params, lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=self.config.learning_rate,
                momentum=0.9, weight_decay=self.config.weight_decay,
                nesterov=True
            )
        else:
            # 默认使用AdamW
            self.optimizer = torch.optim.AdamW(
                params, lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _init_scheduler(self):
        """初始化学习率调度器"""
        if self.config.scheduler_type == "cosine_warm":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=self.config.epochs // 4,
                T_mult=2,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == "reduce_on_plateau":
            try:
                # 尝试使用verbose参数（新版本PyTorch）
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=self.config.early_stop_patience // 4,
                    min_lr=self.config.min_lr,
                    verbose=True
                )
            except TypeError:
                # 兼容旧版本PyTorch
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=self.config.early_stop_patience // 4,
                    min_lr=self.config.min_lr
                )
        else:
            self.scheduler = None
    
    def _init_logging(self):
        """初始化日志系统"""
        if TENSORBOARD_AVAILABLE:
            log_dir = Path("runs") / f"rotfit_{int(time.time())}"
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard日志目录: {log_dir}")
        else:
            self.writer = None
            logger.info("使用基础日志模式（TensorBoard不可用）")
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失"""
        v_in, v_out = outputs['v_in'], outputs['v_out']
        
        # 主损失
        main_loss = self.model.loss_fn(v_in, v_out)
        
        # 正则化损失
        reg_loss = 0.0
        
        # L2正则化（已在优化器中）
        
        # 旋转矩阵正交性约束
        if hasattr(self.config, 'orthogonal_reg') and self.config.orthogonal_reg > 0:
            R_in, R_out = outputs['R_in'], outputs['R_out']
            I = torch.eye(3, device=R_in.device).expand_as(R_in)
            
            ortho_loss_in = F.mse_loss(torch.bmm(R_in, R_in.transpose(-1, -2)), I)
            ortho_loss_out = F.mse_loss(torch.bmm(R_out, R_out.transpose(-1, -2)), I)
            
            reg_loss += self.config.orthogonal_reg * (ortho_loss_in + ortho_loss_out)
        
        # 置换矩阵双随机约束
        if hasattr(self.config, 'doubly_stochastic_reg') and self.config.doubly_stochastic_reg > 0:
            P_in, P_out = outputs['P_in'], outputs['P_out']
            
            # 行和列的和应该接近1
            row_sum_loss = F.mse_loss(P_in.sum(dim=-1), torch.ones(P_in.shape[0], device=P_in.device))
            col_sum_loss = F.mse_loss(P_in.sum(dim=-2), torch.ones(P_in.shape[1], device=P_in.device))
            
            reg_loss += self.config.doubly_stochastic_reg * (row_sum_loss + col_sum_loss)
        
        return main_loss + reg_loss
    
    def train_epoch(self, train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        in_angles, out_angles, probe_vectors = train_data
        
        # 数据增强
        if self.config.angle_noise_std > 0:
            noise = torch.randn_like(in_angles) * self.config.angle_noise_std
            in_angles = in_angles + noise
            
            noise = torch.randn_like(out_angles) * self.config.angle_noise_std
            out_angles = out_angles + noise
        
        self.optimizer.zero_grad()
        
        # 前向传播（混合精度）
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(in_angles, out_angles, probe_vectors, training=True)
                loss = self._compute_loss(outputs)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(in_angles, out_angles, probe_vectors, training=True)
            loss = self._compute_loss(outputs)
            
            loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
        
        train_loss = loss.item()
        
        # 验证
        val_loss = None
        if val_data is not None:
            val_loss = self._validate(val_data)
        
        # 学习率调度
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                self.scheduler.step()
        
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
            'temperature': outputs.get('temperature', self.config.sinkhorn_temp)
        }
    
    def _validate(self, val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """验证"""
        self.model.eval()
        
        with torch.no_grad():
            in_angles, out_angles, probe_vectors = val_data
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(in_angles, out_angles, probe_vectors, training=False)
                    loss = self._compute_loss(outputs)
            else:
                outputs = self.model(in_angles, out_angles, probe_vectors, training=False)
                loss = self._compute_loss(outputs)
        
        return loss.item()
    
    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
             val_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """完整训练流程"""
        logger.info("开始终极训练流程")
        logger.info(f"训练数据: {train_data[0].shape}, 验证数据: {'有' if val_data else '无'}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # 训练一个epoch
            metrics = self.train_epoch(train_data, val_data)
            
            # 记录历史
            self.history['train_loss'].append(metrics['train_loss'])
            if metrics['val_loss'] is not None:
                self.history['val_loss'].append(metrics['val_loss'])
            self.history['learning_rate'].append(metrics['lr'])
            self.history['temperature'].append(metrics['temperature'])
            
            # TensorBoard日志
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', metrics['train_loss'], epoch)
                if metrics['val_loss'] is not None:
                    self.writer.add_scalar('Loss/Validation', metrics['val_loss'], epoch)
                self.writer.add_scalar('LearningRate', metrics['lr'], epoch)
                self.writer.add_scalar('Temperature', metrics['temperature'], epoch)
            
            # 早停检查
            current_loss = metrics['val_loss'] if metrics['val_loss'] is not None else metrics['train_loss']
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
                self._save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.early_stop_patience:
                logger.info(f"早停于epoch {epoch}, 最佳损失: {self.best_loss:.6e}")
                break
            
            # 进度报告
            if epoch % 100 == 0 or epoch == self.config.epochs - 1:
                elapsed = time.time() - start_time
                avg_time = self.model.forward_time / max(1, self.model.forward_calls)
                
                logger.info(f"Epoch [{epoch}/{self.config.epochs}] "
                          f"Train Loss: {metrics['train_loss']:.6e} "
                          f"Val Loss: {metrics['val_loss']:.6e if metrics['val_loss'] else 'N/A'} "
                          f"LR: {metrics['lr']:.2e} "
                          f"Temp: {metrics['temperature']:.3f} "
                          f"Time: {elapsed:.1f}s "
                          f"Avg Forward: {avg_time*1000:.1f}ms")
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"训练完成! 总时间: {total_time:.1f}s, 最佳损失: {self.best_loss:.6e}")
        
        if self.writer is not None:
            self.writer.close()
        
        return {
            'history': self.history,
            'best_loss': self.best_loss,
            'total_time': total_time,
            'final_epoch': self.epoch
        }
    
    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'history': self.history,
            'config': asdict(self.config)
        }
        torch.save(checkpoint, filename)

# -----------------------------
# 终极主函数
# -----------------------------
def train_ultimate(config_file: str = "config.json"):
    """终极训练函数"""
    logger.info("=== 启动终极旋转拟合器 ===")
    
    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)
    
    # 解析配置，移除不支持的参数
    training_dict = raw_config.get('training', {})
    # 移除device参数，它不属于TrainingConfig
    training_dict.pop('device', None)
    training_config = TrainingConfig(**training_dict)
    
    # 数据处理
    processor = DataProcessor(OPT_CONFIG)
    in_angles_rad, out_angles_rad, in_cols, out_cols = processor.read_dataset(
        raw_config['data']['file']
    )
    
    # 数据分割
    N = len(in_angles_rad)
    if training_config.validation_split > 0:
        val_size = int(N * training_config.validation_split)
        indices = np.random.permutation(N)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_in = torch.tensor(in_angles_rad[train_indices], dtype=OPT_CONFIG.dtype, device=DEVICE)
        train_out = torch.tensor(out_angles_rad[train_indices], dtype=OPT_CONFIG.dtype, device=DEVICE)
        val_in = torch.tensor(in_angles_rad[val_indices], dtype=OPT_CONFIG.dtype, device=DEVICE)
        val_out = torch.tensor(out_angles_rad[val_indices], dtype=OPT_CONFIG.dtype, device=DEVICE)
        
        logger.info(f"数据分割: 训练={len(train_indices)}, 验证={len(val_indices)}")
    else:
        train_in = torch.tensor(in_angles_rad, dtype=OPT_CONFIG.dtype, device=DEVICE)
        train_out = torch.tensor(out_angles_rad, dtype=OPT_CONFIG.dtype, device=DEVICE)
        val_in = val_out = None
    
    # 探测向量
    probe_vectors = torch.tensor(
        raw_config['rotation'].get('vectors', [[1, 0, 0]]),
        dtype=OPT_CONFIG.dtype, device=DEVICE
    )
    
    # 创建模型
    model = RotationFitterUltimate(
        n_in=len(in_cols),
        n_out=len(out_cols),
        config=training_config,
        fixed_axes_in=raw_config['rotation'].get('fixed_axes', {}).get('in', {}),
        fixed_axes_out=raw_config['rotation'].get('fixed_axes', {}).get('out', {}),
        fixed_in_order=raw_config['rotation'].get('fixed_order', {}).get('in', {}),
        fixed_out_order=raw_config['rotation'].get('fixed_order', {}).get('out', {}),
        mode=raw_config['rotation'].get('mode', 'discrete')
    ).to(DEVICE)
    
    # 模型编译（PyTorch 2.0+）
    if OPT_CONFIG.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("模型编译成功")
        except Exception as e:
            logger.warning(f"模型编译失败: {e}")
    
    # 创建训练器
    trainer = UltimateTrainer(model, training_config)
    
    # 训练
    train_data = (train_in, train_out, probe_vectors)
    val_data = (val_in, val_out, probe_vectors) if val_in is not None else None
    
    results = trainer.train(train_data, val_data)
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        final_outputs = model(train_in, train_out, probe_vectors, training=False)
    
    # 后处理结果
    def to_numpy_safe(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy()
    
    # 保存结果
    final_result = {
        'config': asdict(training_config),
        'data_info': {
            'input_file': raw_config['data']['file'],
            'input_columns': in_cols,
            'output_columns': out_cols,
            'total_samples': N,
            'training_samples': len(train_in),
            'validation_samples': len(val_in) if val_in is not None else 0
        },
        'model_info': {
            'n_in': len(in_cols),
            'n_out': len(out_cols),
            'mode': raw_config['rotation'].get('mode', 'continuous'),
            'probe_vectors': probe_vectors.cpu().tolist(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        'training_results': {
            'best_loss': results['best_loss'],
            'final_epoch': results['final_epoch'],
            'total_time': results['total_time'],
            'avg_epoch_time': results['total_time'] / max(1, results['final_epoch']),
            'history': {k: v[-100:] for k, v in results['history'].items()}  # 只保存最后100个点
        },
        'final_outputs': {
            'input_axes': to_numpy_safe(final_outputs['axes_in']).tolist(),
            'output_axes': to_numpy_safe(final_outputs['axes_out']).tolist(),
            'input_permutation': softperm_to_perm_optimized(to_numpy_safe(final_outputs['P_in'])),
            'output_permutation': softperm_to_perm_optimized(to_numpy_safe(final_outputs['P_out'])),
            'final_temperature': final_outputs.get('temperature', training_config.sinkhorn_temp)
        },
        'performance_stats': {
            'forward_calls': model.forward_calls,
            'total_forward_time': model.forward_time,
            'avg_forward_time': model.forward_time / max(1, model.forward_calls),
            'device': str(DEVICE),
            'mixed_precision': trainer.use_amp
        }
    }
    
    # 保存到文件
    result_file = raw_config['output'].get('result_file', 'result_ultimate.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"终极训练完成! 结果保存至: {result_file}")
    logger.info(f"最佳损失: {results['best_loss']:.6e}")
    logger.info(f"训练时间: {results['total_time']:.1f}s")
    logger.info(f"平均前向时间: {model.forward_time/max(1, model.forward_calls)*1000:.1f}ms")
    
    return final_result

# 高级置换转换（复用优化版本的函数）
def softperm_to_perm_optimized(P: np.ndarray) -> List[int]:
    """优化的软置换到硬置换转换"""
    if P is None:
        return []
        
    n = P.shape[0]
    
    if SCIPY_AVAILABLE:
        try:
            row_ind, col_ind = linear_sum_assignment(-P)
            return col_ind.tolist()
        except Exception:
            pass
    
    # 贪心算法
    perm = [-1] * n
    used_cols = set()
    
    candidates = [(P[i, j], i, j) for i in range(n) for j in range(n)]
    candidates.sort(reverse=True)
    
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

if __name__ == "__main__":
    train_ultimate("configs/ultimate.json")
