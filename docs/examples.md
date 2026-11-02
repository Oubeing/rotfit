# 使用示例

## 示例1: 基本使用 - 机械臂关节校准

### 场景描述
有一个4轴机械臂，需要校准输入关节角度和实际输出角度的关系。

### 数据准备
```csv
in1,in2,in3,in4,out1,out2,out3
0,0,0,0,0,0,0
10,0,0,0,9.8,0.2,0.1
0,15,0,0,0.1,14.9,-0.3
0,0,20,0,-0.2,0.1,19.7
0,0,0,25,0.3,-0.1,0.2
...
```

### 配置文件
```json
{
  "data": {"file": "robot_calibration.csv"},
  "rotation": {
    "mode": "discrete",
    "vectors": [[1,0,0], [0,1,0], [0,0,1]]
  },
  "training": {
    "epochs": 5000,
    "learning_rate": 0.005
  },
  "output": {"result_file": "robot_result.json"}
}
```

### 预期结果
程序会学习到每个关节的实际旋转轴方向和旋转顺序，用于后续的运动控制校正。

## 示例2: 固定约束 - 云台系统

### 场景描述
一个2轴云台系统，已知第1轴是垂直轴（Z轴），需要学习第2轴方向。

### 配置文件
```json
{
  "data": {"file": "gimbal_data.csv"},
  "rotation": {
    "mode": "continuous",
    "vectors": [[1,0,0]],
    "fixed_axes": {
      "in": {"0": [0,0,1]},
      "out": {"0": [0,0,1]}
    }
  },
  "training": {
    "epochs": 3000,
    "learning_rate": 0.01
  }
}
```

### 数据格式
```csv
in1,in2,out1,out2
0,0,0,0
30,0,29.8,0.5
0,45,0.2,44.7
30,45,29.6,45.1
...
```

## 示例3: 多探测向量 - 高精度校准

### 场景描述
需要高精度校准的旋转系统，使用多个探测向量提高拟合精度。

### 配置文件
```json
{
  "data": {"file": "precision_data.csv"},
  "rotation": {
    "mode": "continuous",
    "vectors": [
      [1,0,0], [0,1,0], [0,0,1],
      [0.707,0.707,0], [0.707,0,0.707], [0,0.707,0.707]
    ]
  },
  "training": {
    "epochs": 10000,
    "learning_rate": 0.001,
    "sinkhorn_temp": 0.1
  }
}
```

### 特点
- 使用6个探测向量覆盖更多方向
- 较低的温度参数获得更确定的置换
- 更多的训练轮数和较小的学习率

## 示例4: 复杂约束 - 部分已知系统

### 场景描述
一个复杂的旋转系统，部分轴方向和顺序已知。

### 配置文件
```json
{
  "data": {"file": "complex_system.csv"},
  "rotation": {
    "mode": "discrete",
    "vectors": [[1,0,0], [0,1,0]],
    "fixed_axes": {
      "in": {
        "0": [1,0,0],
        "2": [0,0,1]
      },
      "out": {
        "1": [0,1,0]
      }
    },
    "fixed_order": {
      "in": {
        "0": 2,
        "3": 0
      }
    }
  },
  "training": {
    "epochs": 8000,
    "learning_rate": 0.002
  }
}
```

### 约束说明
- 输入第0轴固定为X轴，第2轴固定为Z轴
- 输出第1轴固定为Y轴
- 输入旋转顺序：第0个位置使用原第2轴，第3个位置使用原第0轴

## 示例5: 故障排除 - 收敛问题

### 问题描述
训练过程中损失不收敛或收敛很慢。

### 解决方案配置
```json
{
  "data": {"file": "difficult_data.csv"},
  "rotation": {
    "mode": "continuous",
    "vectors": [[1,0,0]]
  },
  "training": {
    "epochs": 20000,
    "learning_rate": 0.0005,
    "sinkhorn_iters": 50,
    "sinkhorn_temp": 1.0
  }
}
```

### 调优策略
1. **降低学习率**: 从0.01降到0.0005
2. **增加训练轮数**: 增加到20000轮
3. **增加Sinkhorn迭代**: 提高到50次
4. **调高温度**: 使用更软的置换矩阵

## 示例6: 数据预处理脚本

### Python脚本示例
```python
import pandas as pd
import numpy as np

# 生成测试数据
def generate_test_data():
    """生成测试用的角度数据"""
    np.random.seed(42)
    n_samples = 1000
    
    # 输入角度（度）
    in_angles = np.random.uniform(-180, 180, (n_samples, 3))
    
    # 模拟输出角度（添加噪声）
    out_angles = np.zeros((n_samples, 2))
    for i in range(n_samples):
        # 简单的映射关系 + 噪声
        out_angles[i, 0] = in_angles[i, 0] * 0.9 + in_angles[i, 1] * 0.1 + np.random.normal(0, 0.5)
        out_angles[i, 1] = in_angles[i, 2] * 0.8 + np.random.normal(0, 0.3)
    
    # 保存为CSV
    df = pd.DataFrame({
        'in1': in_angles[:, 0],
        'in2': in_angles[:, 1], 
        'in3': in_angles[:, 2],
        'out1': out_angles[:, 0],
        'out2': out_angles[:, 1]
    })
    df.to_csv('test_data.csv', index=False)
    print("测试数据已生成: test_data.csv")

if __name__ == "__main__":
    generate_test_data()
```

## 结果分析示例

### 结果文件解读
```json
{
  "input_axes": [
    [0.998, 0.062, 0.001],    // 接近X轴
    [0.001, 0.999, 0.045],    // 接近Y轴  
    [0.045, -0.001, 0.999]    // 接近Z轴
  ],
  "output_axes": [
    [1.000, 0.000, 0.000],    // 精确X轴
    [0.000, 0.000, 1.000]     // 精确Z轴
  ],
  "input_order": [0, 2, 1],   // 顺序: 第0轴->第2轴->第1轴
  "output_order": [1, 0],     // 顺序: 第1轴->第0轴
  "final_loss": 2.34e-07      // 很小的损失值，表示拟合良好
}
```

### Python分析脚本
```python
import json
import numpy as np

def analyze_result(result_file):
    """分析拟合结果"""
    with open(result_file, 'r') as f:
        result = json.load(f)
    
    print(f"最终损失: {result['final_loss']:.2e}")
    print(f"训练轮数: {result['epochs']}")
    
    # 分析轴方向
    in_axes = np.array(result['input_axes'])
    print("\n输入轴分析:")
    for i, axis in enumerate(in_axes):
        print(f"轴{i}: {axis}, 模长: {np.linalg.norm(axis):.3f}")
    
    # 分析顺序
    print(f"\n输入旋转顺序: {result['input_order']}")
    print(f"输出旋转顺序: {result['output_order']}")

if __name__ == "__main__":
    analyze_result("result.json")
```

这些示例涵盖了从基本使用到复杂约束的各种场景，可以作为不同应用的参考模板。
