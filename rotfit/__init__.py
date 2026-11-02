"""RotFit — 旋转轴与旋转顺序反演拟合工具包。

从「输入角度序列 → 输出角度序列」的观测数据中，用可微优化反推出一个多轴
旋转系统的**旋转轴方向**与**旋转顺序**：

    continuous 模式    旋转轴可以是任意 3D 方向（单位向量）
    discrete   模式    旋转轴被约束为 6 个标准方向 ±X / ±Y / ±Z

包结构
------
    rotfit.core     共享数学原语：Sinkhorn 软置换、Rodrigues 轴角转换、
                    软置换→硬置换、数据读取（与具体模型无关）
    rotfit.models   各个实现变体：baseline / compatible / optimized /
                    ultimate / fixed_discrete
    rotfit.utils    辅助脚本：离散轴后处理补丁、离散化问题诊断

推荐入口
--------
命令行（在项目根目录执行）::

    python scripts/train.py --model baseline

代码内调用::

    from rotfit.models.baseline import train_and_save
    train_and_save("configs/baseline.json")

扩展新模型时从 ``rotfit.models.baseline`` 复制改写即可，它依赖 core，
逻辑最干净。
"""

__version__ = "1.0.0"

__all__ = ["core", "models", "utils"]
