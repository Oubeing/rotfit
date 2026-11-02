# API 文档

## 核心函数说明

### 1. 数据处理

#### `read_dataset(path: str)`
**功能**: 读取角度数据文件
**参数**: 
- `path`: 文件路径（支持CSV和Excel）
**返回**: 
- `in_angles_rad`: 输入角度数组（弧度）
- `out_angles_rad`: 输出角度数组（弧度）
- `in_cols`: 输入列名列表
- `out_cols`: 输出列名列表
**说明**: 自动识别以"in"和"out"开头的列，并将角度从度转换为弧度

### 2. 算法核心

#### `sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20, temp: float = 1.0)`
**功能**: Sinkhorn算法实现软置换矩阵
**参数**:
- `log_alpha`: 输入的对数概率矩阵
- `n_iters`: 迭代次数
- `temp`: 温度参数（越小越接近硬置换）
**返回**: 软置换矩阵（双随机矩阵）
**原理**: 通过交替的行列归一化逼近置换矩阵

#### `axis_angle_to_matrix_batch(axis: torch.Tensor, angle: torch.Tensor)`
**功能**: 批量轴角到旋转矩阵转换
**参数**:
- `axis`: 旋转轴向量 [N, 3]
- `angle`: 旋转角度 [N]
**返回**: 旋转矩阵 [N, 3, 3]
**原理**: 使用Rodrigues公式进行转换

### 3. 置换处理

#### `softperm_to_perm(P: np.ndarray) -> List[int]`
**功能**: 将软置换矩阵转换为确定置换
**参数**: 
- `P`: 软置换矩阵
**返回**: 置换列表
**算法**: 优先使用匈牙利算法，否则使用贪心算法

#### `permute_rows(arr: np.ndarray, perm: List[int]) -> np.ndarray`
**功能**: 按照置换重排数组行
**参数**:
- `arr`: 输入数组
- `perm`: 置换列表
**返回**: 重排后的数组

## 主要类

### `RotationFitter(torch.nn.Module)`

**初始化参数**:
- `n_in`: 输入旋转轴数量
- `n_out`: 输出旋转轴数量  
- `mode`: 模式 ("continuous" 或 "discrete")
- `fixed_axes_in/out`: 固定轴约束字典
- `fixed_in/out_order`: 固定顺序约束字典

**可学习参数**:
- `in_axes_param/logits`: 输入旋转轴参数
- `out_axes_param/logits`: 输出旋转轴参数
- `in_order_logits`: 输入顺序logits
- `out_order_logits`: 输出顺序logits

#### `forward(in_angles, out_angles, probe_v, sinkhorn_iters=32, sinkhorn_temp=0.5)`
**功能**: 前向传播计算
**参数**:
- `in_angles`: 输入角度 [B, n_in]
- `out_angles`: 输出角度 [B, n_out]
- `probe_v`: 探测向量 [n_vectors, 3]
- `sinkhorn_iters`: Sinkhorn迭代次数
- `sinkhorn_temp`: Sinkhorn温度

**返回**:
- `vin`: 输入旋转后的向量
- `vout`: 输出旋转后的向量
- `Rin`: 输入旋转矩阵
- `Rout`: 输出旋转矩阵
- `P_in/P_out`: 软置换矩阵
- `axes_in/out`: 旋转轴方向

**计算流程**:
1. 通过Sinkhorn算法计算软置换矩阵
2. 应用固定约束条件
3. 构建旋转轴方向向量
4. 计算复合旋转矩阵
5. 应用到探测向量

## 主函数

### `train_and_save(config_file: str)`
**功能**: 完整的训练和保存流程
**参数**: 
- `config_file`: 配置文件路径
**流程**:
1. 加载配置和数据
2. 初始化模型和优化器
3. 训练循环（支持多探测向量）
4. 保存结果到JSON文件

**训练循环**:
```python
for ep in range(epochs):
    total_loss = 0.0
    # 对每个probe vector计算损失
    for i in range(probe_v.shape[0]):
        current_probe = probe_v[i:i+1]
        vin, vout, ... = model(in_angles, out_angles, current_probe, ...)
        probe_loss = (vin - vout).pow(2).sum() / N
        total_loss += probe_loss
    
    loss = total_loss / probe_v.shape[0]  # 平均损失
    loss.backward()
    optimizer.step()
```

## 配置参数详解

### 旋转配置
- `mode`: 
  - `"continuous"`: 连续模式，轴可以是任意3D方向
  - `"discrete"`: 离散模式，轴限制为±X,±Y,±Z六个方向
- `vectors`: 探测向量列表，用于损失计算
- `fixed_axes`: 固定轴约束
- `fixed_order`: 固定顺序约束

### 训练配置
- `epochs`: 训练轮数
- `learning_rate`: Adam优化器学习率
- `sinkhorn_iters`: Sinkhorn算法迭代次数（影响置换质量）
- `sinkhorn_temp`: 温度参数（影响软硬程度）
- `device`: 计算设备

## 数学原理

### 旋转复合
输入旋转: R_in = R_n × ... × R_2 × R_1
输出旋转: R_out = R_m × ... × R_2 × R_1

### 损失函数
L = (1/N) * Σ ||R_in * v - R_out * v||²

对于多探测向量:
L = (1/K) * Σ_k [(1/N) * Σ ||R_in * v_k - R_out * v_k||²]

### Sinkhorn算法
迭代更新:
1. K = exp(log_alpha / temp)
2. K = K / K.sum(dim=1, keepdim=True)  # 行归一化
3. K = K / K.sum(dim=0, keepdim=True)  # 列归一化
4. 重复步骤2-3

## 约束处理

### 固定轴约束
在forward过程中直接覆盖对应轴的方向：
```python
for idx, vec in fixed_axes.items():
    axes[int(idx), :] = torch.tensor(vec, device=device)
```

### 固定顺序约束
通过设置logits的极值实现：
```python
self.order_logits[pos, :] = -50.0      # 其他位置设为很小值
self.order_logits[pos, idx] = 50.0     # 目标位置设为很大值
```

这样设计确保了模型的可微分性，同时满足各种约束条件。
