# 🔍 Discrete模式轴不是标准组合的原因分析

## ❓ 问题描述

**用户疑问**: 为什么config.json设置`"mode": "discrete"`，但输出的`input_axes`和`output_axes`不是[1,0,0]、[0,1,0]、[0,0,1]这样的标准组合？

## 🧮 数学原理分析

### **Discrete模式的实现流程**

```python
# 关键代码（第293-294行）
axes_in = P_in @ (F.softmax(self.in_axes_logits, dim=-1) @ self.axis_map)
axes_out = P_out @ (F.softmax(self.out_axes_logits, dim=-1) @ self.axis_map)
```

### **问题分解**

#### 1. **基础轴映射 (axis_map)**
```
axis_map = [
  [1,  0,  0],  # +X轴
  [0,  1,  0],  # +Y轴  
  [0,  0,  1],  # +Z轴
  [-1, 0,  0],  # -X轴
  [0, -1,  0],  # -Y轴
  [0,  0, -1]   # -Z轴
]
```

#### 2. **Softmax选择 (第一个问题)**

**理想情况**：
```
in_axes_logits[0] = [5, -2, -2, -2, -2, -2]  # 强烈偏好+X轴
softmax结果 ≈ [0.99, 0.002, 0.002, 0.002, 0.002, 0.002]
选择的轴 ≈ [1, 0, 0]  # 接近+X轴
```

**实际情况**：
```
in_axes_logits[0] = [0.1, 0.05, -0.02, 0.08, -0.01, 0.03]  # 训练中的状态
softmax结果 ≈ [0.18, 0.17, 0.16, 0.18, 0.16, 0.17]  # 接近均匀分布！
选择的轴 = 0.18*[1,0,0] + 0.17*[0,1,0] + ... 
         = [0.18-0.18, 0.17-0.17, 0.16-0.16]
         = [0.00, 0.00, 0.00] + 噪声
         ≈ [0.02, -0.01, 0.03]  # 不是标准轴！
```

#### 3. **Sinkhorn置换 (第二个问题)**

**理想的置换矩阵**：
```
P_in = [
  [1, 0, 0, 0],  # 第0个位置选择第0个轴
  [0, 1, 0, 0],  # 第1个位置选择第1个轴
  [0, 0, 1, 0],  # 第2个位置选择第2个轴
  [0, 0, 0, 1]   # 第3个位置选择第3个轴
]
```

**实际的软置换矩阵**：
```
P_in ≈ [
  [0.7, 0.1, 0.1, 0.1],  # 软选择，不是硬选择
  [0.1, 0.6, 0.2, 0.1],
  [0.1, 0.1, 0.7, 0.1], 
  [0.2, 0.1, 0.1, 0.6]
]
```

#### 4. **两次软操作的叠加效应**

```
最终轴 = P_in @ (softmax_weights @ axis_map)
       = 软置换 @ (软选择 @ 标准轴)
       = 软 @ 软 @ 硬
       = 更软的结果！
```

## 📊 数值示例

### **假设我们有3个输入轴**

#### **Step 1: Softmax权重**
```
轴0的logits: [0.2, 0.1, -0.1, 0.0, -0.05, 0.05]
轴0的softmax: [0.21, 0.19, 0.15, 0.17, 0.14, 0.18]
轴0选择的方向 = 0.21*[1,0,0] + 0.19*[0,1,0] + ... 
              = [0.21-0.17, 0.19-0.14, 0.15-0.18]
              = [0.04, 0.05, -0.03]  # 不是标准轴！
```

#### **Step 2: Sinkhorn置换**
```
P_in = [
  [0.6, 0.2, 0.2],  # 软置换，不是[1,0,0]
  [0.1, 0.7, 0.2],
  [0.3, 0.1, 0.6]
]
```

#### **Step 3: 最终结果**
```
最终轴0 = P_in[0] @ 所有软选择轴
        = 0.6 * [0.04, 0.05, -0.03] + 0.2 * [...] + 0.2 * [...]
        = [0.03, 0.02, -0.01]  # 进一步偏离标准轴！
```

## 🔧 解决方案

### **方案1: 硬选择 (Gumbel-Softmax with hard=True)**
```python
# 修改第440-441行
axes_in = P_in @ (F.gumbel_softmax(self.in_axes_logits, tau=0.1, hard=True) @ self.axis_map)
axes_out = P_out @ (F.gumbel_softmax(self.out_axes_logits, tau=0.1, hard=True) @ self.axis_map)
```

### **方案2: 后处理强制离散化**
```python
def discretize_axes(soft_axes, axis_map):
    """将软轴强制离散化为最接近的标准轴"""
    discretized = []
    for axis in soft_axes:
        # 计算与所有标准轴的余弦相似度
        similarities = cosine_similarity(axis, axis_map)
        best_idx = argmax(similarities)
        discretized.append(axis_map[best_idx])
    return discretized

# 在获取轴后应用
axes_in = discretize_axes(axes_in, self.axis_map)
axes_out = discretize_axes(axes_out, self.axis_map)
```

### **方案3: 约束优化**
```python
# 在训练过程中添加离散化损失
discrete_loss = 0
for axis in axes_in:
    # 计算与最近标准轴的距离
    distances = [||axis - std_axis||² for std_axis in axis_map]
    discrete_loss += min(distances)

total_loss = main_loss + lambda * discrete_loss
```

## 🎯 推荐解决方案

### **立即可用的修复**：

在你的代码中找到第293-294行，修改为：

```python
# 原代码
axes_in = P_in @ (F.softmax(self.in_axes_logits, dim=-1) @ self.axis_map)
axes_out = P_out @ (F.softmax(self.out_axes_logits, dim=-1) @ self.axis_map)

# 修复代码
# 使用更低的温度参数使选择更加确定
temp = 0.1
axes_in_soft = P_in @ (F.softmax(self.in_axes_logits / temp, dim=-1) @ self.axis_map)
axes_out_soft = P_out @ (F.softmax(self.out_axes_logits / temp, dim=-1) @ self.axis_map)

# 强制离散化
def discretize_axis(axis):
    similarities = F.cosine_similarity(axis.unsqueeze(0), self.axis_map, dim=1)
    best_idx = torch.argmax(similarities)
    return self.axis_map[best_idx]

axes_in = torch.stack([discretize_axis(axis) for axis in axes_in_soft])
axes_out = torch.stack([discretize_axis(axis) for axis in axes_out_soft])
```

## 📋 总结

**问题根源**：
1. ✗ Softmax产生概率分布而不是one-hot选择
2. ✗ Sinkhorn产生软置换而不是硬置换  
3. ✗ 两次软操作叠加导致结果偏离标准轴

**解决方案**：
1. ✅ 使用更低温度的Softmax
2. ✅ 使用Gumbel-Softmax的hard模式
3. ✅ 后处理强制离散化
4. ✅ 添加离散化正则项

**最简单的修复**：在轴生成后添加强制离散化步骤，确保输出真正的标准轴方向。
