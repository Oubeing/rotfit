# 🔧 如何修复Discrete模式轴不是标准组合的问题

## 🎯 问题总结

**问题**: config.json设置`"mode": "discrete"`，但输出的轴不是[1,0,0]、[0,1,0]这样的标准组合

**根本原因**: 
1. Softmax产生概率分布而不是one-hot选择
2. Sinkhorn产生软置换矩阵而不是硬置换
3. 两次"软操作"叠加导致结果偏离标准轴

## 🛠️ 立即可用的修复方案

### **方案1: 最简单修复（推荐）**

在你的代码中找到第293-294行（或类似的轴生成代码）：

```python
# 原代码
axes_in = P_in @ (F.softmax(self.in_axes_logits, dim=-1) @ self.axis_map)
axes_out = P_out @ (F.softmax(self.out_axes_logits, dim=-1) @ self.axis_map)
```

**替换为**：

```python
# 修复代码
if self.mode == "discrete":
    # 使用更低温度使选择更确定
    temp = 0.1
    axes_in_soft = P_in @ (F.softmax(self.in_axes_logits / temp, dim=-1) @ self.axis_map)
    axes_out_soft = P_out @ (F.softmax(self.out_axes_logits / temp, dim=-1) @ self.axis_map)
    
    # 强制离散化为最接近的标准轴
    def discretize_axis(axis):
        similarities = F.cosine_similarity(axis.unsqueeze(0), self.axis_map, dim=1)
        best_idx = torch.argmax(similarities)
        return self.axis_map[best_idx]
    
    axes_in = torch.stack([discretize_axis(axis) for axis in axes_in_soft])
    axes_out = torch.stack([discretize_axis(axis) for axis in axes_out_soft])
else:
    # 连续模式保持原样
    axes_in = P_in @ self.in_axes_param
    axes_out = P_out @ self.out_axes_param
```

### **方案2: 使用预制修复函数**

1. 将`discrete_fix_patch.py`放在同一目录
2. 在你的代码中添加：

```python
# 在文件开头导入
from discrete_fix_patch import fix_discrete_axes_numpy

# 在轴生成后添加（约第306行之前）
if self.mode == "discrete":
    # 转换为numpy进行修复
    axes_in_np = axes_in.detach().cpu().numpy()
    axes_out_np = axes_out.detach().cpu().numpy()
    
    # 修复为标准轴
    axes_in_fixed = fix_discrete_axes_numpy(axes_in_np, verbose=False)
    axes_out_fixed = fix_discrete_axes_numpy(axes_out_np, verbose=False)
    
    # 转换回tensor
    axes_in = torch.tensor(axes_in_fixed, device=axes_in.device, dtype=axes_in.dtype)
    axes_out = torch.tensor(axes_out_fixed, device=axes_out.device, dtype=axes_out.dtype)
```

### **方案3: 使用Gumbel-Softmax（终极版本适用）**

如果你使用的是终极版本，修改第440-441行：

```python
# 原代码
axes_in = P_in @ (F.gumbel_softmax(self.in_axes_logits, tau=temp, hard=False) @ self.axis_map)

# 修复代码
axes_in = P_in @ (F.gumbel_softmax(self.in_axes_logits, tau=0.1, hard=True) @ self.axis_map)
```

## 📋 具体修复步骤

### **对于1_compatible.py用户**：

1. **找到第293-294行**：
```python
axes_in = P_in @ (F.softmax(self.in_axes_logits, dim=-1) @ self.axis_map)
axes_out = P_out @ (F.softmax(self.out_axes_logits, dim=-1) @ self.axis_map)
```

2. **替换为修复代码**：
```python
# 生成软轴
axes_in_soft = P_in @ (F.softmax(self.in_axes_logits / 0.1, dim=-1) @ self.axis_map)
axes_out_soft = P_out @ (F.softmax(self.out_axes_logits / 0.1, dim=-1) @ self.axis_map)

# 强制离散化
def discretize_axes(soft_axes, axis_map):
    discretized = []
    for axis in soft_axes:
        similarities = torch.cosine_similarity(axis.unsqueeze(0), axis_map, dim=1)
        best_idx = torch.argmax(similarities)
        discretized.append(axis_map[best_idx])
    return torch.stack(discretized)

if self.mode == "discrete":
    axes_in = discretize_axes(axes_in_soft, self.axis_map)
    axes_out = discretize_axes(axes_out_soft, self.axis_map)
else:
    axes_in = axes_in_soft
    axes_out = axes_out_soft
```

### **对于1_optimized.py用户**：

类似的修改，找到对应的轴生成代码并应用相同的修复。

### **对于1_ultimate.py用户**：

1. 找到第440-441行
2. 将`hard=False`改为`hard=True`
3. 将温度设置为更小的值（如0.1）

## ✅ 验证修复效果

修复后，你的结果应该显示：

```json
{
  "input_axes": [
    [1.000000, 0.000000, 0.000000],  // 精确的+X轴
    [0.000000, 1.000000, 0.000000],  // 精确的+Y轴
    [0.000000, 0.000000, 1.000000],  // 精确的+Z轴
    [-1.000000, 0.000000, 0.000000]  // 精确的-X轴
  ],
  "output_axes": [
    [0.000000, 0.000000, 1.000000],  // 精确的+Z轴
    [1.000000, 0.000000, 0.000000]   // 精确的+X轴
  ]
}
```

而不是之前的：
```json
{
  "input_axes": [
    [0.987234, 0.023451, -0.012345],  // 接近但不精确
    [0.012345, 0.976543, 0.034567],   // 接近但不精确
    ...
  ]
}
```

## 🎯 选择哪个方案？

- **新手用户**: 使用方案2（预制修复函数）
- **有经验用户**: 使用方案1（直接修改代码）
- **终极版本用户**: 使用方案3（Gumbel-Softmax）

## ⚠️ 注意事项

1. **训练可能稍慢**: 强制离散化会让训练稍微慢一点，但结果更准确
2. **梯度问题**: 如果使用硬离散化，可能会有梯度问题，可以考虑在训练时使用软版本，推理时使用硬版本
3. **备份代码**: 修改前请备份原代码

现在你就可以获得真正的标准轴输出了！🚀
