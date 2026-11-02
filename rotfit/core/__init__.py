"""核心数学原语与数据读取，供各模型变体复用。"""

from .rotation import (
    sinkhorn,
    axis_angle_to_matrix_batch,
    softperm_to_perm,
    permute_rows,
)
from .data import read_dataset

__all__ = [
    "sinkhorn",
    "axis_angle_to_matrix_batch",
    "softperm_to_perm",
    "permute_rows",
    "read_dataset",
]
