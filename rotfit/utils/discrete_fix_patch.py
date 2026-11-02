#!/usr/bin/env python3
# discrete_fix_patch.py
# 修复discrete模式轴不是标准组合的补丁

"""
使用方法：
1. 将这个文件放在与你的主程序相同的目录
2. 在你的主程序中导入这个修复函数
3. 在获取轴后调用修复函数

或者直接复制相关函数到你的代码中
"""

import numpy as np

def fix_discrete_axes_numpy(axes_array, verbose=True):
    """
    修复discrete模式下的轴，确保输出标准轴组合
    
    Args:
        axes_array: numpy数组，形状为 [n_axes, 3]
        verbose: 是否打印修复过程
    
    Returns:
        修复后的标准轴数组
    """
    
    # 标准轴定义
    standard_axes = np.array([
        [1.0, 0.0, 0.0],   # +X
        [0.0, 1.0, 0.0],   # +Y
        [0.0, 0.0, 1.0],   # +Z
        [-1.0, 0.0, 0.0],  # -X
        [0.0, -1.0, 0.0],  # -Y
        [0.0, 0.0, -1.0]   # -Z
    ])
    
    direction_names = ['+X', '+Y', '+Z', '-X', '-Y', '-Z']
    
    if verbose:
        print("=== Discrete轴修复过程 ===")
        print("修复前的轴:")
    
    fixed_axes = []
    
    for i, axis in enumerate(axes_array):
        if verbose:
            print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
        
        # 计算与所有标准轴的余弦相似度
        axis_normalized = axis / (np.linalg.norm(axis) + 1e-8)
        similarities = []
        
        for std_axis in standard_axes:
            similarity = np.dot(axis_normalized, std_axis)
            similarities.append(similarity)
        
        # 找到最相似的标准轴
        similarities = np.array(similarities)
        best_idx = np.argmax(np.abs(similarities))  # 使用绝对值找最相似
        best_similarity = similarities[best_idx]
        
        # 选择对应的标准轴
        chosen_axis = standard_axes[best_idx].copy()
        
        if verbose:
            print(f"       最接近: {direction_names[best_idx]} (相似度: {best_similarity:.6f})")
            print(f"       修复为: [{chosen_axis[0]:.6f}, {chosen_axis[1]:.6f}, {chosen_axis[2]:.6f}]")
        
        fixed_axes.append(chosen_axis)
    
    fixed_axes = np.array(fixed_axes)
    
    if verbose:
        print("\n修复后的轴:")
        for i, axis in enumerate(fixed_axes):
            print(f"  轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
        print("=== 修复完成 ===\n")
    
    return fixed_axes

def analyze_discrete_problem(axes_array):
    """
    分析discrete模式下轴的问题
    """
    print("=== Discrete轴问题分析 ===")
    
    standard_axes = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]
    ])
    direction_names = ['+X', '+Y', '+Z', '-X', '-Y', '-Z']
    
    print("当前轴分析:")
    for i, axis in enumerate(axes_array):
        print(f"\n轴{i}: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")
        
        # 计算模长
        norm = np.linalg.norm(axis)
        print(f"  模长: {norm:.6f}")
        
        # 检查是否接近标准轴
        axis_normalized = axis / (norm + 1e-8)
        similarities = []
        
        for j, std_axis in enumerate(standard_axes):
            similarity = np.dot(axis_normalized, std_axis)
            similarities.append((similarity, direction_names[j]))
        
        # 排序找到最相似的
        similarities.sort(key=lambda x: abs(x[0]), reverse=True)
        
        print(f"  最接近的标准轴:")
        for sim, name in similarities[:3]:
            print(f"    {name}: {sim:.6f}")
        
        # 判断是否是"纯"轴
        best_sim = abs(similarities[0][0])
        if best_sim > 0.99:
            print(f"  ✓ 接近标准轴 {similarities[0][1]}")
        elif best_sim > 0.9:
            print(f"  ⚠ 较接近标准轴 {similarities[0][1]}")
        else:
            print(f"  ✗ 不接近任何标准轴")
    
    print("\n=== 分析完成 ===")

# 示例用法
def example_usage():
    """示例用法"""
    
    # 模拟一些非标准的轴（来自discrete模式的输出）
    problematic_axes = np.array([
        [0.95, 0.02, 0.03],   # 接近+X但不完全是
        [0.01, 0.97, -0.02],  # 接近+Y但不完全是
        [-0.03, 0.01, 0.98],  # 接近+Z但不完全是
        [0.1, 0.2, 0.3]       # 完全不是标准轴
    ])
    
    print("=== 示例：修复非标准轴 ===")
    
    # 分析问题
    analyze_discrete_problem(problematic_axes)
    
    # 修复轴
    fixed_axes = fix_discrete_axes_numpy(problematic_axes)
    
    return fixed_axes

if __name__ == "__main__":
    # 运行示例
    example_usage()
    
    print("\n" + "="*50)
    print("如何在你的代码中使用这个修复：")
    print("""
1. 在你的代码中找到轴生成的部分（大约第293-294行）
2. 在获取axes_in和axes_out后，添加：

    # 修复discrete轴
    if self.mode == "discrete":
        axes_in_np = axes_in.detach().cpu().numpy()
        axes_out_np = axes_out.detach().cpu().numpy()
        
        axes_in_fixed = fix_discrete_axes_numpy(axes_in_np)
        axes_out_fixed = fix_discrete_axes_numpy(axes_out_np)
        
        axes_in = torch.tensor(axes_in_fixed, device=axes_in.device, dtype=axes_in.dtype)
        axes_out = torch.tensor(axes_out_fixed, device=axes_out.device, dtype=axes_out.dtype)

3. 或者直接修改forward函数中的轴生成逻辑
    """)
