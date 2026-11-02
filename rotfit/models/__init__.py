"""模型变体集合。

每个变体都是一个独立、自包含的训练入口，便于横向对比与逐步演进：

- baseline        —— 基准实现（原 1.py），逻辑最干净，依赖 core。推荐从这里扩展。
- compatible      —— 兼容性最高（多版本/可选依赖容错）。
- optimized       —— 平衡性能（float32、稳定 Sinkhorn、AdamW+早停+梯度裁剪）。
- ultimate        —— 工程化（已修复两处致命 bug，但 torch.compile/JIT 等未端到端验证）。
- fixed_discrete  —— 调试专用，用于诊断 discrete 模式下的轴离散化问题。

统一训练入口见项目根目录的 scripts/train.py（默认调用 baseline）。
"""

from . import baseline

__all__ = ["baseline"]
