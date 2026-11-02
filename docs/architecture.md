# 架构与扩展指南

本文面向要**改代码 / 加模型**的开发者。只想跑通流程请看根目录 `README.md`；
配置与排错细节看 `docs/usage.md`。

## 1. 目录结构

```
rotfit/                             # 项目根（所有命令都在这里执行）
├── README.md                       # 安装 + 使用 + 排错
├── requirements.txt
├── pyproject.toml                  # 包元信息（可选：pip install -e .）
├── rotfit/                         # 核心 Python 包
│   ├── __init__.py                 #   版本号与包说明
│   ├── core/                       #   与模型无关的共享层
│   │   ├── rotation.py             #     sinkhorn / Rodrigues / 软置换→硬置换 / 行置换
│   │   └── data.py                 #     read_dataset（CSV / Excel → 弧度角度）
│   ├── models/                     #   实现变体（每个都能独立运行）
│   │   ├── baseline.py             #     基准实现，依赖 core，推荐扩展起点
│   │   ├── compatible.py           #     兼容性优先
│   │   ├── optimized.py            #     性能优先
│   │   ├── ultimate.py             #     工程化（compile / JIT / 混合精度）
│   │   └── fixed_discrete.py       #     离散化诊断工具
│   └── utils/                      #   辅助脚本
│       ├── discrete_fix_patch.py   #     离散轴后处理补丁
│       └── debug_discrete_axes.py  #     离散化问题诊断
├── configs/                        # 配置（data/output 路径相对项目根）
├── data/                           # 示例数据
├── results/                        # 训练输出；historical/ 存历史产物
├── scripts/
│   ├── train.py                    # 统一训练入口（CLI）
│   └── verify.py                   # 结果校验 + 塌缩诊断
├── docs/                           # 文档
├── reference/                      # 项目起源材料（原始对话存档、初版实现）
├── tests/                          # 预留测试目录
└── logs/                           # 运行日志
```

## 2. 分层设计

```
scripts/train.py ──► rotfit/models/*      （训练编排：读配置 → 建模型 → 优化 → 写结果）
                        │
                        └──► rotfit/core  （纯函数：Sinkhorn、Rodrigues、置换、读数据）
scripts/verify.py  ──► rotfit/core        （独立重建，不复用模型代码，避免自证）
```

**约定**

1. `core` 只放纯函数，不依赖任何 `models`；不得反向 import。
2. `models/*` 之间互不 import，允许重复少量样板代码，保证单个文件可独立运行。
3. 配置键一律用 JSON 表达，不在代码里硬编码常量。
4. 路径统一「相对项目根」，由 `scripts/train.py` 负责切换工作目录。

## 3. core 提供的原语

| 函数 | 作用 |
|---|---|
| `sinkhorn(log_alpha, n_iters, temp)` | 分数矩阵 → 近似双随机（软置换）矩阵，表示旋转顺序 |
| `axis_angle_to_matrix_batch(axis, angle)` | Rodrigues：轴 + 弧度 → 旋转矩阵 `(B,3,3)` |
| `softperm_to_perm(P)` | 软置换 → 硬置换（优先 Hungarian，退化贪心） |
| `permute_rows(arr, perm)` | 按置换重排数组行 |
| `read_dataset(path)` | 读 CSV/Excel，返回 `(in_rad, out_rad, in_cols, out_cols)` |

## 4. 模型契约

新增变体必须满足以下契约，才能被 CLI 与校验脚本正确调用：

1. 入口函数签名：`train_*(config_file: str) -> dict`（返回结果字典，同时写入 JSON）。
2. 读取配置的键：`data.file`、`rotation.mode / vectors / fixed_axes / fixed_order`、
   `training.*`、`output.result_file`。
3. 结果 JSON 至少包含：`input_axes`、`output_axes`、`input_order`、`output_order`、`final_loss`。
   `scripts/verify.py` 依赖这些字段独立复算。
4. `discrete` 模式：训练时用软轴保留梯度，推理时用 `argmax` 吸附到 ±X/±Y/±Z。
5. 轴数组按**位置**排列，`axes[i]` 与第 i 列角度对应；`order` 只是用于解释的置换记录。

## 5. 扩展步骤

### A. 换数据 / 新实验

1. 数据放 `data/`，列名以 `in` / `out` 开头；
2. 复制 `configs/baseline.json`，改 `data.file`、`mode`、`vectors`、`output.result_file`；
3. `python scripts/train.py --config configs/你的.json`；
4. `python scripts/verify.py --config configs/你的.json`。

### B. 新增模型变体

```bash
cp rotfit/models/baseline.py rotfit/models/my_variant.py
```

- 只 import `rotfit.core.*`；
- 改写 `RotationFitter.forward` 与 `train_and_save`；
- 在 `scripts/train.py` 的 `MODELS` 里加一行：

```python
MODELS = {
    ...,
    "my_variant": ("rotfit.models.my_variant", "train_and_save"),
}
```

- 再补一份 `configs/my_variant.json` 即可从 CLI 调用。

### C. 新增损失 / 正则 / 调度器

- 通用数学放进 `core/rotation.py` 或新建 `core/losses.py`；
- 模型相关逻辑留在对应 `models/*`；
- 新增超参必须走配置文件，并在本文档的配置表里登记。

### D. 新增数据格式

`core/data.py::read_dataset` 是唯一数据入口：目前按列名前缀 `in` / `out` 解析 CSV / Excel。
支持新格式（如 JSON、Parquet）时，在这里加分支并保持返回值结构不变。

## 6. 已知限制

- **可辨识性依赖探针与样本**：单探针 + 极少样本会塌缩到平凡解（所有轴同向）。
  `scripts/verify.py` 会给出诊断，不要只看 `final_loss`。
- **不含平移**：模型只描述纯旋转链，直线轴需要另建参数化。
- **`ultimate` 变体**：`torch.compile` / JIT / 混合精度路径未做端到端验证，生产前先小样本试跑。
- **`baseline` 的轴是软化值**：`discrete` 模式下保存的是 softmax 软化后的单位向量，
  要严格标准轴请用 `verify.py --snap-discrete`。
- **无固定随机种子**：多次运行结果有差异，正式结论建议多次取共识。

## 7. 相关问题档案

历史问题的成因与修复记录见：

- `docs/discrete_axes_analysis.md` —— 离散模式轴为何偏离标准方向；
- `docs/gradient_fix_summary.md` —— 梯度断流、`device` 参数等修复；
- `docs/project_overview.md` —— 项目沿革与历次问题梳理；
- `docs/changelog.md` —— 版本变更。
