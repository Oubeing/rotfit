# RotFit · 旋转轴与旋转顺序反演拟合

> 给定一台「多轴串联旋转机构」的观测数据 —— 每个样本记录了 **输入侧 N 个转角** 与
> **输出侧 M 个转角**，但不知道每一步到底绕哪根轴、以什么顺序叠加 —— RotFit 用可微优化
> 把这套**旋转轴方向**与**旋转顺序**反演出来。

```
R_in  = R_n · … · R_2 · R_1        （输入侧，每个 R 由「轴 + 转角」构成）
R_out = R_m · … · R_2 · R_1        （输出侧）
目标：对所有探针向量 v，使  R_in · v ≈ R_out · v
```

两个待求量全部由模型自己学出来：
- **轴方向**：`continuous` 模式为任意 3D 单位向量；`discrete` 模式约束为 ±X/±Y/±Z 六选一；
- **旋转顺序**：用 Sinkhorn 迭代得到近似置换矩阵，再取硬置换。

---

## 目录

1. [能做什么 / 不能做什么](#1-能做什么--不能做什么)
2. [工作原理](#2-工作原理)
3. [环境要求](#3-环境要求)
4. [安装](#4-安装)
5. [数据准备](#5-数据准备)
6. [快速开始（三步跑通）](#6-快速开始三步跑通)
7. [配置文件详解](#7-配置文件详解)
8. [模型变体与运行命令](#8-模型变体与运行命令)
9. [结果解读与可信度校验](#9-结果解读与可信度校验)
10. [常见问题](#10-常见问题)
11. [项目结构](#11-项目结构)
12. [二次开发](#12-二次开发)
13. [文档索引](#13-文档索引)

---

## 1. 能做什么 / 不能做什么

**适合**
- 多轴转台、串联机械臂末端、摆动/回转机构等**旋转链**的运动学参数反演；
- 已知每级转角数值，但不确定各关节轴向（例如装配后轴线有偏差）或叠加次序；
- 有两组观测（输入链 / 输出链）需要互相标定、对齐的场景。

**不适合**
- 含平移（直线轴）的机构 —— 本模型只有旋转；
- 转角几乎全为 0、或样本数量极少（< 5 条）的数据 —— 信息量不足，必然出现平凡解；
- 需要实时在线辨识的场合 —— 这是离线批处理拟合工具。

## 2. 工作原理

| 环节 | 做法 |
|---|---|
| 轴参数化 | `continuous`：直接优化单位向量；`discrete`：6 维 logits 经 softmax 选 ±X/±Y/±Z |
| 顺序参数化 | 对 `n×n` 分数矩阵做 Sinkhorn 迭代，得到近似双随机（软置换）矩阵 |
| 轴角 → 矩阵 | Rodrigues 公式（`rotfit/core/rotation.py`） |
| 损失 | 各探针向量上 `‖R_in·v − R_out·v‖²` 的均值 |
| 离散化 | 训练时用软轴（保梯度），推理时 `argmax` 吸附到最近的标准轴 |

> 关键点：**探针向量决定问题是否可辨识**。只用一个探针时，模型有大量等价解，
> 极易塌缩成「所有轴都指向同一方向」的平凡解（详见 [第 9 节](#9-结果解读与可信度校验)）。

## 3. 环境要求

- Python **≥ 3.9**（本项目在 3.13 + torch 2.15 CPU 上验证通过）
- PyTorch **≥ 2.0**（CPU 版即可运行；有 CUDA 会自动用 GPU）
- 其余依赖：`numpy`、`pandas`、`scipy`（硬置换用匈牙利算法，缺失时自动退化为贪心）、`openpyxl`（读 xlsx）
- 无强制 GPU 要求；数据量小时几秒到几分钟即可完成训练

## 4. 安装

在项目根目录（本文件所在目录）依次执行：

```bash
# 1) 进入项目根目录
cd rotfit

# 2) 建虚拟环境（推荐，避免污染全局环境）
python -m venv .venv
# Windows (Git Bash / PowerShell)
source .venv/Scripts/activate
# macOS / Linux
# source .venv/bin/activate

# 3) 升级安装器并安装依赖
python -m pip install -U pip
python -m pip install -r requirements.txt
```

需要 GPU 版 PyTorch 时，用官方源单独装一次（版本号按你的 CUDA 调整）：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

验证安装：

```bash
python -c "import torch, numpy, pandas, scipy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

> **Windows 提示**：若 `pip` 命令报错或者装到了别的 Python 上，统一改用 `python -m pip`。

## 5. 数据准备

### 5.1 文件格式

CSV（或 xlsx），**第一行为列名**，一行一个样本：

```csv
in1,in2,in3,in4,out1,out2
12.5,-3.2,45.0,8.0,7.5,-19.4
-30.0,11.1,0.0,22.6,-5.1,9.8
...
```

现有示例数据 `data/angles.csv`（4 输入轴 / 2 输出轴 / 4 条样本）可直接用于跑通流程。

### 5.2 列名规则

| 规则 | 说明 |
|---|---|
| 输入列 | 列名以 `in` 开头（`in1`、`in2`…），按出现顺序对应第 1、2… 级旋转 |
| 输出列 | 列名以 `out` 开头（`out1`、`out2`…） |
| 单位 | **度**（读取时自动转弧度），不接受弧度混用 |
| 顺序 | 列顺序 = 从最内层到最外层的叠加顺序，务必与物理链一致 |

### 5.3 数据质量要求（直接决定结果可信度）

1. **样本数量**：建议 ≥ 20 条；少于 5 条基本无法辨识（示例数据仅用于演示流程）。
2. **姿态多样性**：转角不要集中在 0 附近，各轴都要有「明显非零」的样本。
3. **避免全 0 行**：全 0 样本对轴方向没有任何约束。

### 5.4 换用自己的数据

把文件放进 `data/`，然后复制一份配置并改 `data.file` 即可（见下一节）。

## 6. 快速开始（三步跑通）

```bash
# 第 1 步：确认数据就位
ls data/angles.csv

# 第 2 步：训练（默认 baseline 变体 + configs/baseline.json + 离散模式）
python scripts/train.py

# 第 3 步：独立校验结果（重建旋转矩阵、复核损失、诊断是否塌缩）
python scripts/verify.py
```

训练过程会打印损失曲线，结束后自动打印结果摘要：

```
[train] 变体=baseline  配置=configs/baseline.json
[0/8000] Loss=7.950223e-01
[500/8000] Loss=2.1e-02
...
[7999/8000] Loss=4.241151e-09

====================================================================
[baseline] 训练完成，结果文件: results/baseline.json
  final_loss : 4.241150908755626e-09
  input_axes : [+1.000, +0.023, -0.006] | ...
  output_axes: [-1.000, +0.000, -0.004] | ...
  input_order: [3, 1, 2, 0]   output_order: [0, 1]
====================================================================
```

校验脚本会额外告诉你「结果是否可信」：

```
轴间最大夹角 : 0.0°
⚠️  诊断：所有轴几乎同向 → 结果极可能塌缩为平凡解，不可信。
```

> 上面这条警告是**示例数据（4 条样本 + 单个探针）的必然结果**，不是代码 bug。
> 换成多探针后「损失接近 0」的假象会消失（损失回到 1e-1 量级），
> 但 4 条样本仍不足以让轴分开 —— 见 9.4 的实测对照。

## 7. 配置文件详解

配置文件放在 `configs/`，JSON 格式。所有路径都**相对项目根目录**书写。
以 `configs/baseline.json` 为例：

```json
{
  "data": { "file": "data/angles.csv" },

  "rotation": {
    "mode": "discrete",
    "vectors": [[1.0, 0.0, 0.0]],
    "fixed_axes":  { "in": { "2": [1, 0, 0] }, "out": {} },
    "fixed_order": { "in": {}, "out": {} }
  },

  "training": {
    "epochs": 8000,
    "learning_rate": 0.001,
    "sinkhorn_iters": 32,
    "sinkhorn_temp": 0.5,
    "device": null
  },

  "output": { "result_file": "results/baseline.json" }
}
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `data.file` | str | 数据文件路径（`.csv` / `.xlsx`） |
| `rotation.mode` | str | `discrete`（轴限 ±X/±Y/±Z）或 `continuous`（任意方向） |
| `rotation.vectors` | list | 探针向量列表，每个是 3 维向量；**至少给 3 个互不平行的**才能有效约束轴 |
| `rotation.fixed_axes.in/out` | dict | 固定某级轴，键是**位置索引**（从 0 开始），值是 3 维向量 |
| `rotation.fixed_order.in/out` | dict | 固定顺序：键是位置，值是被放到该位置的轴索引 |
| `training.epochs` | int | 训练轮数；8000 对 4 条样本足够，样本变多需上调 |
| `training.learning_rate` | float | 学习率，默认 1e-3 |
| `training.sinkhorn_iters` | int | Sinkhorn 迭代次数，越大置换越硬 |
| `training.sinkhorn_temp` | float | Sinkhorn 温度，越小越接近硬置换 |
| `training.device` | str/null | `null` 自动选择；也可写 `"cpu"` / `"cuda"` |
| `output.result_file` | str | 结果 JSON 保存路径 |

### 常用改法

**换数据**：改 `data.file`。
**改成连续模式**：`"mode": "continuous"`。
**加探针（强烈推荐）**：`"vectors": [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]`。
**已知某级轴**（例如输入第 3 级确定绕 X）：`"fixed_axes": { "in": { "2": [1,0,0] }, "out": {} }`。

## 8. 模型变体与运行命令

```bash
python scripts/train.py --list                 # 查看所有变体
python scripts/train.py                        # baseline（推荐起点）
python scripts/train.py --model compatible     # 兼容性优先
python scripts/train.py --model optimized      # 性能优先
python scripts/train.py --model ultimate       # 工程化（compile/JIT/混合精度）
python scripts/train.py --model fixed_discrete # 离散化诊断对比
python scripts/train.py --config configs/mine.json   # 指定任意配置
```

| 变体 | 定位 | 主要差异 |
|---|---|---|
| `baseline` | **推荐起点**，逻辑最干净 | float64、逐探针计算、训练软轴 / 推理硬离散 |
| `compatible` | 兼容性优先 | 检测 torch 版本与可选依赖，全链路 try-except 容错 |
| `optimized` | 性能优先 | float32、批量探针、数值稳定 Sinkhorn、AdamW + 早停 + 梯度裁剪 |
| `ultimate` | 工程化 | torch.compile / JIT / 混合精度 / TensorBoard；特性未端到端验证，生产前先小样本试跑 |
| `fixed_discrete` | 诊断工具 | 连续跑「强制离散」与「软轴」两遍做对比，不产出正式结果 |

## 9. 结果解读与可信度校验

### 9.1 结果字段

`results/<变体>.json`：

| 字段 | 含义 |
|---|---|
| `input_axes` / `output_axes` | 各级旋转轴（已按位置排好，`input_axes[i]` 对应第 i 级） |
| `input_order` / `output_order` | 硬置换结果：`order[i]` 表示第 i 级用的是第 `order[i]` 个学到的轴 |
| `final_loss` | 最后一次迭代的损失（各探针均方误差的平均） |
| `vector_v` | 本次使用的探针向量 |
| `mode` 相关 | 配置回显（`fix_axes` / `fix_order` / `epochs` / `learning_rate`） |

> `discrete` 模式下轴是 softmax 软化后的近似单位向量（如 `[+1.000, +0.023, -0.006]`）。
> 想看真正离散的轴，给 `verify.py` 加 `--snap-discrete`。

### 9.2 结果是否可信？看三个信号

```bash
python scripts/verify.py                    # 用结果里的轴独立重算损失
python scripts/verify.py --snap-discrete    # 吸附到标准轴后再算一遍
```

1. **重建损失 ≈ 训练损失** —— 说明结果文件自洽，代码没有算错。
2. **轴间最大夹角不能接近 0°** —— 所有轴同向 = 塌缩到平凡解，结果无意义。
3. **损失小但轴乱** —— 说明约束不足，需要加探针或补样本。

### 9.3 为什么会出现「损失极小但结果无效」

若所有轴都塌到 +X，而探针恰好是 `[1,0,0]`，那么绕 X 的旋转不改变 `[1,0,0]`，
于是 `R_in·v ≡ R_out·v ≡ v`，损失天然为 0 —— 无论真实几何是什么。

**解决顺序**（从低成本到高成本）：
1. 探针改为多个互不平行方向：`[[1,0,0],[0,1,0],[0,0,1],[1,1,1]]`；
2. 固定已知的轴/顺序（`fixed_axes` / `fixed_order`）缩小解空间；
3. 补充真实标定样本（不同姿态、更多数量）—— **样本量才是决定性因素**；
4. 换 `continuous` 模式看轴是否能分散开。

### 9.4 实测对照（示例数据，仅 4 条样本）

| 配置 | `final_loss` | 轴间最大夹角 | 判定 |
|---|---|---|---|
| `configs/baseline.json`（1 探针，离散） | 4.24e-09 | 0.0° | 损失极小但全塌到 ±X，**不可信** |
| `configs/multiprobe.json`（6 探针，离散） | 1.25e-01 | 0.0° | 假象被戳破（损失不再趋 0），但仍塌到同一根轴 |

**结论**：单探针会制造「损失完美」的假象；加探针能让假象暴露，但**样本不足时仍会塌缩**。
这组示例数据的定位只是「跑通流程」，不要把它当作可用的标定结果。

## 10. 常见问题

**Q1. 报错 `ModuleNotFoundError: No module named 'rotfit'`**
没有在项目根目录运行。`scripts/train.py` 会自动切根目录，但用 `python -m rotfit.…` 时必须在根目录。

**Q2. 报错找不到 `data/angles.csv`**
配置里的路径是相对**项目根目录**的。请确认当前目录是本 README 所在目录，或改用绝对路径。

**Q3. `pip install` 没反应 / 装到了别的 Python**
Windows 上 `pip` 常指向另一个解释器。统一用 `python -m pip install …`。

**Q4. 训练很慢**
调小 `training.epochs`（如 2000）先看趋势；或换 `optimized` 变体；有 GPU 时把 `device` 设为 `"cuda"`。

**Q5. 损失降到 1e-9 但轴全是 +X**
典型塌缩，见 9.3。示例数据只有 4 条样本，出现这种情况属正常。

**Q6. `discrete` 模式下轴不是严格的标准方向**
训练阶段为保梯度使用软轴。加 `--snap-discrete` 校验，或参考 `docs/discrete_axes_analysis.md` 了解原理。

**Q7. 换成 `continuous` 模式后结果差异很大**
正常：`continuous` 解空间大得多，对数据量要求更高；两者结论应互相印证再采信。

**Q8. 能读 Excel 吗？**
可以，`data.file` 指向 `.xlsx` 即可（需要 `openpyxl`）。`data/angles.xlsx`、`data/angles_labeled.xlsx` 是同一份示例数据的两种格式/列名版本。

## 11. 项目结构

```
rotfit/
├── README.md                  # 本文（安装 + 使用 + 排错）
├── requirements.txt           # 依赖清单
├── pyproject.toml             # 包元信息（可选：pip install -e .）
├── rotfit/                    # 核心 Python 包
│   ├── core/                  #   共享数学原语 + 数据读取（与模型无关）
│   │   ├── rotation.py        #     sinkhorn / Rodrigues / 软置换→硬置换
│   │   └── data.py            #     read_dataset（CSV/Excel → 弧度）
│   ├── models/                #   5 个实现变体（每个都能独立跑）
│   └── utils/                 #   离散轴后处理补丁 / 诊断脚本
├── configs/                   # 配置：baseline / compatible / optimized /
│                              #       ultimate / sanity_continuous / multiprobe
├── data/                      # 示例数据（angles.csv / angles.xlsx / angles_labeled.xlsx）
├── results/                   # 训练输出（historical/ 为历史产物存档）
├── scripts/
│   ├── train.py               # 统一训练入口
│   └── verify.py              # 结果校验 + 塌缩诊断
├── docs/                      # 架构、原理与历史文档
├── reference/                 # 项目起源材料（原始对话存档、初版实现）
├── tests/                     # 预留测试目录
└── logs/                      # 运行日志（ultimate 变体）
```

## 12. 二次开发

```python
from rotfit.models.baseline import train_and_save
from rotfit.core.rotation import sinkhorn, axis_angle_to_matrix_batch, softperm_to_perm
from rotfit.core.data import read_dataset

train_and_save("configs/baseline.json")
```

新增模型的标准动作：
1. 复制 `rotfit/models/baseline.py` → `rotfit/models/my_variant.py`，改写 `RotationFitter`；
2. 只依赖 `rotfit.core`，不要跨模型互相 import；
3. 在 `scripts/train.py` 的 `MODELS` 字典加一行即可接入 CLI；
4. 参数一律走配置文件，不要硬编码。

详细约定见 [`docs/architecture.md`](docs/architecture.md)。

## 13. 文档索引

| 文档 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 文档总索引（现行 / 历史文档区分） |
| [docs/architecture.md](docs/architecture.md) | 架构、模块约定、扩展步骤 |
| [docs/usage.md](docs/usage.md) | 使用手册（配置实战、调参、结果判定、排错） |
| [docs/changelog.md](docs/changelog.md) | 版本变更记录 |
| [docs/project_overview.md](docs/project_overview.md) | 项目沿革与问题档案 |
| [docs/discrete_axes_analysis.md](docs/discrete_axes_analysis.md) | `discrete` 模式轴偏离标准方向的根因 |
| [docs/gradient_fix_summary.md](docs/gradient_fix_summary.md) | 梯度 / 离散相关修复记录 |
| [docs/comparison_matrix.md](docs/comparison_matrix.md) | 各变体对比矩阵 |
| [docs/api.md](docs/api.md) · [docs/examples.md](docs/examples.md) | 早期接口与示例（部分口径已过时） |

---

**版本**：1.0.0 ｜ **许可**：MIT
