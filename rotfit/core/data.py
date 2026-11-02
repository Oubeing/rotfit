"""数据读取：把 angles 表格读成 (输入角度弧度, 输出角度弧度, 输入列名, 输出列名)。

约定：
- 输入列名以 `in` 开头，输出列名以 `out` 开头（不区分大小写）。
- 角度以「度」存储，这里统一转成「弧度」。
- 路径解析：若给定路径不存在，会依次尝试 `<cwd>/<path>`、`<cwd>/data/<basename>`、
  `<project_root>/data/<basename>`，方便从项目根目录运行。
"""

import os
import numpy as np
import pandas as pd

# 项目根目录：rotfit/core/data.py -> 上两级
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    basename = os.path.basename(path)
    candidates = [
        path,
        os.path.join("data", basename),
        os.path.join(_PROJECT_ROOT, "data", basename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path  # 仍返回原路径，由调用方报错


def read_dataset(path: str):
    path = _resolve(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine="openpyxl")
    cols = list(df.columns)
    in_cols = [c for c in cols if c.lower().startswith("in")]
    out_cols = [c for c in cols if c.lower().startswith("out")]
    if len(in_cols) == 0 or len(out_cols) == 0:
        raise ValueError("数据文件必须包含 in/out 列")
    in_arr = df[in_cols].to_numpy(dtype=float)
    out_arr = df[out_cols].to_numpy(dtype=float)
    return np.deg2rad(in_arr), np.deg2rad(out_arr), in_cols, out_cols
