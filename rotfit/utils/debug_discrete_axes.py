#!/usr/bin/env python3
# 调试discrete模式下轴不是标准组合的问题

import torch
import torch.nn.functional as F
import numpy as np

def demonstrate_discrete_problem():
    """演示discrete模式下轴生成的问题"""
    
    print("=== Discrete模式轴生成问题分析 ===\n")
    
    # 模拟参数
    n_axes = 3
    
    # 基础轴映射 [+X, +Y, +Z, -X, -Y, -Z]
    axis_map = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]
    ])
    print("基础轴映射 (axis_map):")
    for i, axis in enumerate(axis_map):
        direction = ['+X', '+Y', '+Z', '-X', '-Y', '-Z'][i]
        print(f"  {i}: {axis.numpy()} ({direction})")
    print()
    
    # 模拟学习到的轴logits (随机初始化后的状态)
    axes_logits = torch.randn(n_axes, 6) * 0.1
    print("学习到的轴logits:")
    print(axes_logits)
    print()
    
    # 第一步：softmax选择 (这里就开始不是纯粹的选择了)
    softmax_weights = F.softmax(axes_logits, dim=-1)
    print("Softmax权重 (每行应该选择一个轴，但实际是概率分布):")
    for i, weights in enumerate(softmax_weights):
        print(f"  轴{i}: {weights.numpy()}")
        max_idx = torch.argmax(weights)
        max_prob = weights[max_idx].item()
        direction = ['+X', '+Y', '+Z', '-X', '-Y', '-Z'][max_idx]
        print(f"       最可能: {direction} (概率: {max_prob:.3f})")
    print()
    
    # 第二步：与axis_map相乘得到实际轴
    discrete_axes_soft = softmax_weights @ axis_map
    print("Softmax后的轴 (不是纯粹的标准轴):")
    for i, axis in enumerate(discrete_axes_soft):
        print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
        norm = torch.norm(axis).item()
        print(f"       模长: {norm:.6f}")
    print()
    
    # 第三步：Sinkhorn置换矩阵的影响
    print("=== Sinkhorn置换矩阵的影响 ===")
    
    # 模拟一个不完美的置换矩阵
    order_logits = torch.randn(n_axes, n_axes) * 0.1
    
    # 简化的sinkhorn (几次迭代)
    K = torch.exp(order_logits / 0.5)
    for _ in range(10):
        K = K / (K.sum(dim=1, keepdim=True) + 1e-8)
        K = K / (K.sum(dim=0, keepdim=True) + 1e-8)
    
    print("Sinkhorn置换矩阵 (应该接近单位矩阵的置换，但实际是软置换):")
    print(K.numpy())
    print()
    
    # 第四步：最终的轴 = 置换矩阵 × 离散轴
    final_axes = K @ discrete_axes_soft
    print("最终输出的轴 (经过Sinkhorn置换后):")
    for i, axis in enumerate(final_axes):
        print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
        norm = torch.norm(axis).item()
        print(f"       模长: {norm:.6f}")
    print()
    
    # 对比：理想的discrete轴应该是什么样
    print("=== 理想的discrete轴应该是 ===")
    ideal_axes = torch.tensor([
        [1., 0., 0.],  # +X
        [0., 1., 0.],  # +Y  
        [0., 0., 1.]   # +Z
    ])
    print("理想轴:")
    for i, axis in enumerate(ideal_axes):
        print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
    print()
    
    # 分析差异
    print("=== 问题总结 ===")
    print("1. Softmax产生概率分布而不是one-hot选择")
    print("2. Sinkhorn产生软置换矩阵而不是硬置换")
    print("3. 两次软操作的叠加导致最终轴偏离标准方向")
    print("4. 即使在discrete模式下，轴也不是严格的[1,0,0]组合")

def show_solution():
    """展示解决方案"""
    print("\n=== 解决方案 ===")
    
    print("方案1: 使用hard=True的Gumbel-Softmax")
    axes_logits = torch.randn(3, 6) * 0.1
    axis_map = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]
    ])
    
    # Hard Gumbel-Softmax
    hard_selection = F.gumbel_softmax(axes_logits, tau=0.1, hard=True)
    hard_axes = hard_selection @ axis_map
    
    print("Hard Gumbel-Softmax结果:")
    for i, axis in enumerate(hard_axes):
        print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
    print()
    
    print("方案2: 后处理强制离散化")
    soft_axes = torch.tensor([
        [0.95, 0.02, 0.03],
        [0.01, 0.97, 0.02], 
        [0.02, 0.01, 0.97]
    ])
    
    # 找到最接近的标准轴
    standard_axes = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]
    ])
    
    discretized_axes = []
    for axis in soft_axes:
        # 计算与所有标准轴的相似度
        similarities = torch.cosine_similarity(axis.unsqueeze(0), standard_axes, dim=1)
        best_idx = torch.argmax(similarities)
        discretized_axes.append(standard_axes[best_idx])
    
    discretized_axes = torch.stack(discretized_axes)
    
    print("强制离散化结果:")
    for i, axis in enumerate(discretized_axes):
        print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")

if __name__ == "__main__":
    demonstrate_discrete_problem()
    show_solution()
