# 变更记录（Changelog）

版本号遵循语义化版本；本项目在重构阶段以「日期 + 版本」双标记。

---

## v1.0.0 — 2026-11-02（项目定名与文档完善）

### 重命名

- 包名 `rotation_fitter` → **`rotfit`**，项目正式定名为 **RotFit（旋转轴与旋转顺序反演拟合）**。
- 全部 import、docstring、文档、日志文件名（`logs/rotfit.log`、`runs/rotfit_*`）同步更新。
- 新增 `pyproject.toml`：项目名、版本、依赖、可选依赖、`rotfit` 命令入口。

### 文档

- 重写 `README.md`：按「安装 → 数据准备 → 快速开始 → 配置 → 运行 → 结果解读 → FAQ → 扩展」组织，
  每一步都有可直接复制的命令与期望输出。
- 新增 `docs/usage.md`：使用手册（数据接入检查项、配置实战、结果判定流程、调参手册、故障排查表）。
- 重写 `docs/architecture.md`：分层设计、模型契约、扩展步骤、已知限制。
- 重写 `docs/project_overview.md`：转为「项目沿革与问题档案」，去掉过时的文件清单。
- 移除所有与本工具无关的第三方软件/行业背景描述，示例数据改为中性列名。

### 工具

- 新增 `scripts/verify.py`：独立重建旋转矩阵，复核损失并输出**塌缩诊断**
  （轴间最大夹角按「+X 与 -X 同轴」口径计算）。
- `scripts/train.py`：新增 `--list`、缺失配置的友好报错、训练结束后自动打印结果摘要。
- 新增 `configs/multiprobe.json`：6 个互不平行探针的推荐配置模板。

### 数据

- `data/一后.xlsx` → `data/angles_labeled.xlsx`（同一份示例数据、带列名的版本），文件名中性化。

---

## v0.2.0 — 2026-11-02（结构化重组 + bug 修复）

### 目录重组

将散落在根目录的 28+ 个文件整理为标准 Python 项目：

- 抽取 `core`（共享旋转数学原语 + 数据读取），消除各变体重复实现；
- 5 个实现变体统一收入 `models/`（baseline / compatible / optimized / ultimate / fixed_discrete）；
- 配置 → `configs/`、数据 → `data/`、结果 → `results/`、文档 → `docs/`、参考 → `reference/`；
- 新增统一入口 `scripts/train.py`、`requirements.txt`、`docs/architecture.md`、顶层 `README.md`。

### 修复 `ultimate` 变体两处致命 bug

1. **模式判断错误**：`_init_parameters()` 用 `self.config.optimizer_type == "continuous"` 判断模式
   （应为 `mode`），且 `mode` 从未传入模型 → 连续模式参数 `in_axes_param / out_axes_param` 永不创建。
   - 修复：`RotationFitterUltimate.__init__` 增加 `mode` 参数并保存为 `self.mode`；
     判断改为 `if self.mode == "continuous":`；`train_ultimate()` 传入 `raw_config['rotation']['mode']`。
2. **未定义变量**：`_get_rotation_axes()` 引用未定义的 `training` → 首次 forward 必抛 `NameError`。
   - 修复：方法签名增加 `training: bool = True`，`forward()` 调用处传入。
- 验证：离散 / 连续两种模式均可构造并前向。

### 运行验证

- `python scripts/train.py --model baseline`：8000 epoch，损失 1.10 → 2.87e-8，
  与重组前行为一致，证明重构未破坏功能。
- 结果仍塌缩到平凡解（所有轴 ≈ +X），属实验设计问题（单探针与轴平行 + 仅 4 样本），
  非代码 bug，详见 `README.md` 第 9 节。

---

## v0.1.0 — 2025-08（初版）

- 5 个独立脚本 `1.py` / `1_compatible.py` / `1_optimized.py` / `1_ultimate.py` / `1_fixed_discrete.py`。
- 核心思路：Sinkhorn 学旋转顺序、Rodrigues 转旋转矩阵、软轴训练 + 推理硬离散。

### 历史要点

- `discrete` 模式天生存在「软操作叠加导致轴偏离标准方向」的问题，
  各修复文档围绕此展开；`baseline` 采用「低温度 + 训练软轴 / 推理离散轴」缓解。
- 早期 `result.json`（来自 optimized 变体）`final_loss` 仅 1.05e-5，但 `best_loss` 为 6.65e-9，
  说明末段训练跑飞、早停未生效，属不稳定训练。
