# 快速使用指南

## 1. 准备数据文件

创建 `angles.csv` 文件，格式如下：
```csv
in1,in2,in3,out1,out2
10,20,-30,45.5,-60.2
15,25,-35,50.1,-65.8
...
```

## 2. 配置参数

编辑 `config.json`：
```json
{
  "data": {"file": "angles.csv"},
  "rotation": {
    "mode": "discrete",
    "vectors": [[1,0,0]]
  },
  "training": {
    "epochs": 2000,
    "learning_rate": 0.01
  }
}
```

## 3. 运行程序

```bash
python 1.py
```

## 4. 查看结果

程序完成后会生成 `result.json`，包含：
- 学习到的旋转轴方向
- 旋转顺序
- 最终损失值

## 常用配置

### 连续模式（任意轴方向）
```json
"rotation": {"mode": "continuous"}
```

### 离散模式（只允许XYZ轴）
```json
"rotation": {"mode": "discrete"}
```

### 固定某个轴
```json
"rotation": {
  "fixed_axes": {
    "in": {"0": [1,0,0]}  // 第0个输入轴固定为X轴
  }
}
```

### 多探测向量
```json
"rotation": {
  "vectors": [
    [1,0,0], [0,1,0], [0,0,1]
  ]
}
```

就这么简单！程序会自动优化旋转参数以最小化输入输出的差异。
