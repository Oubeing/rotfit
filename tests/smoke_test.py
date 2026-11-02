"""冒烟测试：不依赖 pytest，直接运行即可。

    python tests/smoke_test.py

覆盖：core 的三个数学原语、数据读取、baseline 模型前向形状。
全部通过会打印 "ALL CHECKS PASSED"。
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

from rotfit.core.rotation import (
    sinkhorn, axis_angle_to_matrix_batch, softperm_to_perm, permute_rows,
)
from rotfit.core.data import read_dataset
from rotfit.models.baseline import RotationFitter


def check(name, fn):
    fn()
    print(f"  ok  {name}")


def test_sinkhorn():
    torch.manual_seed(0)
    logits = torch.randn(4, 4, dtype=torch.float64)
    P = sinkhorn(logits, n_iters=32, temp=0.5)
    assert P.shape == (4, 4)
    assert torch.allclose(P.sum(dim=1), torch.ones(4, dtype=torch.float64), atol=1e-6), "行和应为 1"
    assert torch.allclose(P.sum(dim=0), torch.ones(4, dtype=torch.float64), atol=1e-6), "列和应为 1"
    assert (P >= 0).all()


def test_rodrigues():
    # 角度为 0 时应为单位矩阵
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    angle = torch.zeros(1, dtype=torch.float64)
    R = axis_angle_to_matrix_batch(axis, angle)
    assert torch.allclose(R[0], torch.eye(3, dtype=torch.float64), atol=1e-9)

    # 绕 Z 轴 90° 应把 X 轴转到 Y 轴
    R90 = axis_angle_to_matrix_batch(axis, torch.tensor([np.pi / 2], dtype=torch.float64))
    v = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    out = (R90[0] @ v).squeeze(-1)
    assert torch.allclose(out, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-9)


def test_perm():
    P = np.eye(3)
    P[0, 0] = 0; P[0, 2] = 1
    P[2, 2] = 0; P[2, 0] = 1
    assert softperm_to_perm(P) == [2, 1, 0]
    arr = np.array([[10], [20], [30]])
    assert permute_rows(arr, [2, 1, 0]).ravel().tolist() == [30, 20, 10]


def test_data():
    in_angle, out_angle, in_cols, out_cols = read_dataset("data/angles.csv")
    assert in_angle.ndim == 2 and out_angle.ndim == 2
    assert in_angle.shape[1] == len(in_cols)
    assert out_angle.shape[1] == len(out_cols)
    assert np.isfinite(in_angle).all() and np.isfinite(out_angle).all()


def test_model_forward():
    in_angle, out_angle, _, _ = read_dataset("data/angles.csv")
    n_in, n_out = in_angle.shape[1], out_angle.shape[1]
    model = RotationFitter(n_in, n_out, mode="discrete").to(torch.float64)
    model.train()
    vin, vout, Rin, Rout, _, _, axes_in, _ = model(
        torch.tensor(in_angle, dtype=torch.float64),
        torch.tensor(out_angle, dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    n = in_angle.shape[0]
    assert vin.shape == (n, 3, 1) and vout.shape == (n, 3, 1)
    assert Rin.shape == (n, 3, 3) and Rout.shape == (n, 3, 3)
    assert axes_in.shape == (n_in, 3)

    # 推理态：轴应吸附到标准方向（坐标为 0 / ±1）
    model.eval()
    with torch.no_grad():
        _, _, _, _, _, _, axes_in_hard, _ = model(
            torch.tensor(in_angle, dtype=torch.float64),
            torch.tensor(out_angle, dtype=torch.float64),
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        )
    assert torch.allclose(axes_in_hard.abs().sum(dim=1), torch.ones(n_in, dtype=torch.float64), atol=1e-6), \
        "离散模式下推理态应为标准轴"
    assert torch.allclose(axes_in_hard.abs().max(dim=1).values, torch.ones(n_in, dtype=torch.float64), atol=1e-6)


if __name__ == "__main__":
    print("RotFit 冒烟测试")
    check("sinkhorn: 双随机性", test_sinkhorn)
    check("rodrigues: 恒等与 90° 旋转", test_rodrigues)
    check("置换: 软→硬 与 行重排", test_perm)
    check("数据: read_dataset", test_data)
    check("模型: 前向形状与离散轴", test_model_forward)
    print("\nALL CHECKS PASSED")
