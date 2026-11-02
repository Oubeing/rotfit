#!/usr/bin/env python3
# rotation_fitter_config_multi_probe.py
# 2025-08 - 支持 config.json 多 probe vector 自动平均损失

import json
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Optional: 使用匈牙利算法确定置换
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# -----------------------------
# 环境
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True

# -----------------------------
# 读取数据
# -----------------------------
def read_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine="openpyxl")
    cols = list(df.columns)
    in_cols = [c for c in cols if c.lower().startswith("in")]
    out_cols = [c for c in cols if c.lower().startswith("out")]
    if len(in_cols) == 0 or len(out_cols) == 0:
        raise ValueError("数据文件必须包含 in/out 列")
    in_arr = df[in_cols].to_numpy(dtype=float)
    out_arr = df[out_cols].to_numpy(dtype=float)
    return np.deg2rad(in_arr), np.deg2rad(out_arr), in_cols, out_cols

# -----------------------------
# Sinkhorn 近似置换
# -----------------------------
def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20, temp: float = 1.0):
    K = torch.exp(log_alpha / temp)
    for _ in range(n_iters):
        K = K / (K.sum(dim=1, keepdim=True) + 1e-12)
        K = K / (K.sum(dim=0, keepdim=True) + 1e-12)
    return K

# -----------------------------
# Rodrigues 批量
# -----------------------------
def axis_angle_to_matrix_batch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
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
        c + x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s,
        y*x*(1-c)+z*s, c + y*y*(1-c), y*z*(1-c)-x*s,
        z*x*(1-c)-y*s, z*y*(1-c)+x*s, c + z*z*(1-c)
    ], dim=-1).reshape(-1, 3, 3)
    return R

# -----------------------------
# 软置换 -> 确定顺序
# -----------------------------
def softperm_to_perm(P: np.ndarray) -> List[int]:
    n = P.shape[0]
    if SCIPY_AVAILABLE:
        cost = -P
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = [-1]*n
        for r,c in zip(row_ind, col_ind):
            perm[r]=int(c)
        return perm
    else:
        perm = [-1]*n
        used_cols = set()
        flat=[]
        for i in range(n):
            for j in range(n):
                flat.append((P[i,j],i,j))
        flat.sort(reverse=True)
        for val,i,j in flat:
            if perm[i]==-1 and j not in used_cols:
                perm[i]=int(j)
                used_cols.add(j)
            if all(x!=-1 for x in perm):
                break
        remaining = [c for c in range(n) if c not in used_cols]
        for i in range(n):
            if perm[i]==-1:
                perm[i]=remaining.pop(0)
        return perm

# -----------------------------
# Helper permute rows
# -----------------------------
def permute_rows(arr: np.ndarray, perm: List[int]) -> np.ndarray:
    out = np.zeros_like(arr)
    for i in range(len(perm)):
        out[i,:] = arr[perm[i],:]
    return out

# -----------------------------
# 模型
# -----------------------------
class RotationFitter(torch.nn.Module):
    def __init__(self, n_in:int, n_out:int, mode:str="continuous",
                 fixed_axes_in:Optional[Dict[int,List[float]]]=None,
                 fixed_axes_out:Optional[Dict[int,List[float]]]=None,
                 fixed_in_order:Optional[Dict[int,int]]=None,
                 fixed_out_order:Optional[Dict[int,int]]=None):
        super().__init__()
        self.n_in=n_in; self.n_out=n_out; self.mode=mode
        if mode=="continuous":
            self.in_axes_param = torch.nn.Parameter(torch.randn(n_in,3)*0.5)
            self.out_axes_param = torch.nn.Parameter(torch.randn(n_out,3)*0.5)
        else:
            # 使用小的随机初始化而不是全零，避免对称性
            self.in_axes_logits = torch.nn.Parameter(torch.randn(n_in,6)*0.1)
            self.out_axes_logits = torch.nn.Parameter(torch.randn(n_out,6)*0.1)
        # 使用小的随机初始化打破对称性，避免Sinkhorn不收敛
        self.in_order_logits = torch.nn.Parameter(torch.randn(n_in,n_in)*0.1)
        self.out_order_logits = torch.nn.Parameter(torch.randn(n_out,n_out)*0.1)
        self.register_buffer("axis_map", torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]))
        self.fixed_axes_in = fixed_axes_in or {}
        self.fixed_axes_out = fixed_axes_out or {}
        self.fixed_in_order = fixed_in_order or {}
        self.fixed_out_order = fixed_out_order or {}

    def forward(self, in_angles:torch.Tensor, out_angles:torch.Tensor,
        probe_v:torch.Tensor, sinkhorn_iters:int=32, sinkhorn_temp:float=0.5):
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
        if self.mode=="continuous":
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
def train_and_save(config_file:str):
    global DEVICE
    with open(config_file,"r",encoding="utf-8") as f:
        cfg=json.load(f)
    data_file=cfg["data"]["file"]
    mode=cfg["rotation"].get("mode","continuous")
    probe_vectors=cfg["rotation"].get("vectors",[[1,0,0]])
    probe_v = torch.tensor(probe_vectors,dtype=torch.float64,device=DEVICE)
    fixed_axes_in=cfg["rotation"].get("fixed_axes",{}).get("in",{})
    fixed_axes_out=cfg["rotation"].get("fixed_axes",{}).get("out",{})
    fixed_in_order=cfg["rotation"].get("fixed_order",{}).get("in",{})
    fixed_out_order=cfg["rotation"].get("fixed_order",{}).get("out",{})
    epochs=cfg["training"].get("epochs",2000)
    lr=cfg["training"].get("learning_rate",1e-2)
    sinkhorn_iters=cfg["training"].get("sinkhorn_iters",32)
    sinkhorn_temp=cfg["training"].get("sinkhorn_temp",0.5)
    device=cfg["training"].get("device",None)
    if device:
        DEVICE=torch.device(device)
    result_file=cfg["output"].get("result_file","rotation_fit_result.json")

    # read data
    in_angles_rad,out_angles_rad,in_cols,out_cols = read_dataset(data_file)
    N,n_in=in_angles_rad.shape
    _,n_out=out_angles_rad.shape
    in_angles=torch.tensor(in_angles_rad,dtype=torch.float64,device=DEVICE)
    out_angles=torch.tensor(out_angles_rad,dtype=torch.float64,device=DEVICE)

    # model
    model=RotationFitter(n_in,n_out,mode,fixed_axes_in,fixed_axes_out,fixed_in_order,fixed_out_order).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)

    for ep in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        
        # 对每个probe vector计算损失并平均
        for i in range(probe_v.shape[0]):
            current_probe = probe_v[i:i+1]  # 保持维度
            vin,vout,_,_,_,_,_,_ = model(in_angles,out_angles,current_probe,sinkhorn_iters,sinkhorn_temp)
            probe_loss = (vin-vout).pow(2).sum()/N
            total_loss += probe_loss
        
        loss = total_loss / probe_v.shape[0]  # 平均损失
        loss.backward()
        optimizer.step()
        if ep%500==0 or ep==epochs-1:
            print(f"[{ep}/{epochs}] Loss={loss.item():.6e}")

    # 获取最终结果 - 使用第一个probe vector作为代表
    first_probe = probe_v[0:1]  # 保持维度
    vin,vout,Rin,Rout,P_in,P_out,axes_in,axes_out = model(in_angles,out_angles,first_probe,sinkhorn_iters,sinkhorn_temp)
    # Sinkhorn 输出 NaN/inf 处理
    P_in_np = P_in.detach().cpu().numpy()
    P_in_np = np.nan_to_num(P_in_np, nan=1e-6, posinf=1e6, neginf=-1e6)
    P_out_np = P_out.detach().cpu().numpy()
    P_out_np = np.nan_to_num(P_out_np, nan=1e-6, posinf=1e6, neginf=-1e6)
    in_order = softperm_to_perm(P_in_np)
    out_order = softperm_to_perm(P_out_np)
    result={
        "input_file": data_file,
        "output_file": "output.csv",
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
    with open(result_file,"w",encoding="utf-8") as f:
        json.dump(result,f,indent=4)
    print(f"结果保存至 {result_file}")

# -----------------------------
# main
# -----------------------------
if __name__=="__main__":
    train_and_save("config.json")
