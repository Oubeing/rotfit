# 文档索引

## 现行文档（与当前代码一致）

| 文档 | 读者 | 内容 |
|---|---|---|
| [`../README.md`](../README.md) | 所有人 | 安装、数据准备、快速开始、配置、结果解读、FAQ |
| [`usage.md`](usage.md) | 使用者 | 使用手册：数据接入检查、配置实战、判定流程、调参、排错 |
| [`architecture.md`](architecture.md) | 开发者 | 分层设计、模型契约、扩展步骤、已知限制 |
| [`changelog.md`](changelog.md) | 维护者 | 版本变更记录 |
| [`project_overview.md`](project_overview.md) | 维护者 | 项目沿革、踩过的坑、结论 |
| [`discrete_axes_analysis.md`](discrete_axes_analysis.md) | 开发者 | 离散模式轴偏离标准方向的根因分析 |

## 历史文档（早期版本，口径可能已过时）

下列文档写于项目重构前，保留是为了追溯问题与设计取舍。**其中的文件名（如 `1.py`、
`config.json`）、参数名与结论请对照现行代码阅读。**

| 文档 | 说明 |
|---|---|
| [`00_original_README.md`](00_original_README.md) | 初版 README |
| [`quickstart.md`](quickstart.md) | 早期快速开始 |
| [`documentation_index.md`](documentation_index.md) | 早期文档索引 |
| [`api.md`](api.md) | 早期 API 说明 |
| [`examples.md`](examples.md) | 早期示例（含随机数据生成示例） |
| [`comparison_matrix.md`](comparison_matrix.md) | 变体对比矩阵（性能数字未经复测） |
| [`compatibility_guide.md`](compatibility_guide.md) | 兼容性说明 |
| [`optimization_guide.md`](optimization_guide.md) | 优化指南 |
| [`how_to_fix_discrete.md`](how_to_fix_discrete.md) | 离散化修复操作说明 |
| [`gradient_fix_summary.md`](gradient_fix_summary.md) | 梯度 / device 等问题修复记录 |
| [`ultimate_optimization.md`](ultimate_optimization.md) | 工程化变体的设计说明 |

## 阅读顺序建议

- **第一次接触**：`README.md` → `usage.md`
- **要改代码**：`architecture.md` → `models/baseline.py`
- **结果不对劲**：`README.md` 第 9 节 → `usage.md` 第 5 节 → `discrete_axes_analysis.md`
- **想了解历史**：`project_overview.md` → `changelog.md`
