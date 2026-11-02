"""结果校验：独立重建旋转矩阵，复核损失并诊断「塌缩 / 平凡解」。

训练出的轴是否可信，不能只看 final_loss —— 当所有轴都塌到同一个方向、且该
方向恰好与探针向量平行时，损失会天然接近 0，但结果毫无意义。本脚本用
**结果文件里保存的轴**独立重算一遍误差，并给出轴多样性诊断。

用法（项目根目录下）：

    python scripts/verify.py                                  # 校验 configs/baseline.json 指向的结果
    python scripts/verify.py --config configs/optimized.json  # 校验另一份配置的结果
    python scripts/verify.py --result results/baseline.json --data data/angles.csv
    python scripts/verify.py --snap-discrete                  # 把轴吸附到 ±X/±Y/±Z 后再算一遍
"""

import argparse
import itertools
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rotfit.core.data import read_dataset  # noqa: E402
from rotfit.core.rotation import axis_angle_to_matrix_batch  # noqa: E402

STANDARD_AXES = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
    dtype=np.float64,
)


def build_R(angles, axes):
    """按位置顺序累乘旋转矩阵：R = Rn @ ... @ R1（与模型内部实现一致）。"""
    n = angles.shape[0]
    R = np.tile(np.eye(3), (n, 1, 1))
    for pos in range(angles.shape[1]):
        ax = np.asarray(axes[pos], dtype=np.float64)
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        ax_batch = torch.tensor(np.tile(ax, (n, 1)), dtype=torch.float64)
        ang_batch = torch.tensor(angles[:, pos], dtype=torch.float64)
        R = axis_angle_to_matrix_batch(ax_batch, ang_batch).numpy() @ R
    return R


def snap(axes):
    """把每个轴吸附到最近的 ±X/±Y/±Z。"""
    out = []
    for ax in axes:
        ax = np.asarray(ax, dtype=np.float64)
        sims = STANDARD_AXES @ (ax / (np.linalg.norm(ax) + 1e-12))
        out.append(STANDARD_AXES[int(np.argmax(sims))].tolist())
    return out


def axis_spread(axes):
    """所有轴「轴向」两两夹角的最大值（度）。

    旋转轴是**有向线段**：+X 与 -X 表示同一根轴（只是转角反号），
    因此用 |cos| 计算夹角。结果为 0 表示全部塌到同一根轴上。
    """
    axes = [np.asarray(a, dtype=np.float64) / (np.linalg.norm(a) + 1e-12) for a in axes]
    worst = 0.0
    for a, b in itertools.combinations(axes, 2):
        cos = min(1.0, abs(float(a @ b)))
        worst = max(worst, math.degrees(math.acos(cos)))
    return worst


def main():
    ap = argparse.ArgumentParser(description="RotFit 结果校验与塌缩诊断")
    ap.add_argument("--config", default="configs/baseline.json", help="配置文件；用于取数据路径与模式")
    ap.add_argument("--result", default=None, help="结果 JSON；默认取配置里的 output.result_file")
    ap.add_argument("--data", default=None, help="数据文件；默认取配置里的 data.file")
    ap.add_argument("--snap-discrete", action="store_true", help="先把轴吸附到 ±X/±Y/±Z 再校验")
    args = ap.parse_args()

    os.chdir(ROOT)

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)

    result_path = args.result or cfg.get("output", {}).get("result_file", "results/baseline.json")
    data_path = args.data or cfg.get("data", {}).get("file", "data/angles.csv")
    mode = cfg.get("rotation", {}).get("mode", "discrete")
    probes = cfg.get("rotation", {}).get("vectors", [[1, 0, 0]])

    for p, label in [(result_path, "结果文件"), (data_path, "数据文件")]:
        if not os.path.exists(p):
            sys.exit(f"[verify] 找不到{label}: {p}")

    with open(result_path, encoding="utf-8") as f:
        res = json.load(f)

    in_angles, out_angles, in_cols, out_cols = read_dataset(data_path)
    axes_in = res["input_axes"]
    axes_out = res["output_axes"]

    if args.snap_discrete:
        axes_in, axes_out = snap(axes_in), snap(axes_out)

    R_in = build_R(in_angles, axes_in)
    R_out = build_R(out_angles, axes_out)

    print("=" * 70)
    print(f"结果文件 : {result_path}")
    print(f"数据文件 : {data_path}   (样本 {in_angles.shape[0]} 条, "
          f"输入 {in_angles.shape[1]} 轴, 输出 {out_angles.shape[1]} 轴)")
    print(f"模式     : {mode}{'  (已吸附到标准轴)' if args.snap_discrete else ''}")
    print(f"训练损失 : {res.get('final_loss')}")
    print("-" * 70)

    total = 0.0
    for v in probes:
        v = np.asarray(v, dtype=np.float64).reshape(3, 1)
        vin = R_in @ v
        vout = R_out @ v
        err = np.linalg.norm(vin - vout, axis=1)
        loss = float((err ** 2).mean())
        total += loss
        print(f"探针 {np.asarray(v).ravel().tolist()}  →  重建损失 {loss:.3e}  "
              f"最大单点误差 {err.max():.3e}")
    total /= len(probes)

    print("-" * 70)
    print("输入轴 :", " | ".join("[" + ", ".join(f"{x:+.3f}" for x in a) + "]" for a in axes_in))
    print("输出轴 :", " | ".join("[" + ", ".join(f"{x:+.3f}" for x in a) + "]" for a in axes_out))
    print(f"顺序   : in={res.get('input_order')}  out={res.get('output_order')}")
    print("-" * 70)

    spread = axis_spread(list(axes_in) + list(axes_out))
    print(f"轴间最大夹角 : {spread:.1f}°")
    collapsed = spread < 5.0
    if collapsed:
        print("⚠️  诊断：所有轴几乎同向 → 结果极可能塌缩为平凡解，不可信。")
        if len(probes) <= 1:
            print("    当前只有 1 个探针，先加探针：[[1,0,0],[0,1,0],[0,0,1],[1,1,1]]")
        else:
            print("    探针已足够，问题多半在样本：增加样本数量与姿态多样性，")
            print("    或把已知的轴/顺序写进 fixed_axes / fixed_order 缩小解空间。")
    else:
        print("✅ 诊断：轴方向具有多样性，未检测到整体塌缩（仍需结合样本量判断可信度）。")

    if res.get("final_loss") is not None and total > 0:
        ratio = total / max(res["final_loss"], 1e-30)
        print(f"重建损失 / 训练损失 = {ratio:.2f}x"
              f"{'（量级一致，结果文件自洽）' if 0.1 <= ratio <= 10 else '（差异较大，注意轴可能被软化或未离散）'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
