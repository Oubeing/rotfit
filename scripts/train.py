"""RotFit 统一训练 / 推理入口。

在项目根目录执行（脚本会自动切换工作目录，无需手动 cd）：

    python scripts/train.py                            # 默认 baseline + configs/baseline.json
    python scripts/train.py --model optimized          # 换变体（自动用 configs/optimized.json）
    python scripts/train.py --config configs/mine.json # 用指定配置（变体默认 baseline）
    python scripts/train.py --model fixed_discrete     # 离散化诊断对比
    python scripts/train.py --list                     # 查看可用变体

约定：配置文件中的 ``data.file`` 与 ``output.result_file`` 一律写
「相对项目根目录」的路径（如 ``data/angles.csv``、``results/baseline.json``）。
"""

import argparse
import json
import os
import sys

# 让项目根目录可被 import（脚本位于 <root>/scripts/）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --model 名称 -> (模块路径, 训练函数名)
MODELS = {
    "baseline": ("rotfit.models.baseline", "train_and_save"),
    "compatible": ("rotfit.models.compatible", "train_compatible"),
    "optimized": ("rotfit.models.optimized", "train_and_save_optimized"),
    "ultimate": ("rotfit.models.ultimate", "train_ultimate"),
}

MODEL_HELP = {
    "baseline": "基准实现，逻辑最干净，推荐从这里扩展",
    "compatible": "兼容性优先（多版本 / 可选依赖容错）",
    "optimized": "性能优先（float32、批量探针、稳定 Sinkhorn、早停）",
    "ultimate": "工程化（torch.compile / JIT / 混合精度，需自行验证）",
    "fixed_discrete": "离散化诊断（强制离散 vs 软轴对比，不产出正式结果）",
}


def _print_result_summary(result_file: str, model: str) -> None:
    """训练结束后回读结果文件，打印关键指标（失败则静默跳过）。"""
    if not result_file or not os.path.exists(result_file):
        return
    try:
        with open(result_file, encoding="utf-8") as f:
            res = json.load(f)
    except Exception:
        return

    def fmt_axes(axes):
        return " | ".join(
            "[" + ", ".join(f"{v:+.3f}" for v in ax) + "]" for ax in axes
        )

    print("\n" + "=" * 68)
    print(f"[{model}] 训练完成，结果文件: {result_file}")
    print(f"  final_loss : {res.get('final_loss')}")
    if res.get("best_loss") is not None:
        print(f"  best_loss  : {res.get('best_loss')}")
    if res.get("input_axes"):
        print(f"  input_axes : {fmt_axes(res['input_axes'])}")
    if res.get("output_axes"):
        print(f"  output_axes: {fmt_axes(res['output_axes'])}")
    print(f"  input_order: {res.get('input_order')}   output_order: {res.get('output_order')}")
    print("=" * 68)
    print("提示：所有轴都接近同一个方向（如全为 [+1,0,0]）时属于塌缩/平凡解，")
    print("      说明数据或探针约束不足，结果不可信 —— 见 README「结果校验」。\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RotFit — 旋转轴与旋转顺序反演训练入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--model",
        default="baseline",
        choices=list(MODELS.keys()) + ["fixed_discrete"],
        help="选择实现变体（默认 baseline）",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="配置文件路径；默认 configs/<model>.json",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用变体后退出",
    )
    args = ap.parse_args()

    if args.list:
        print("可用变体：")
        for name, desc in MODEL_HELP.items():
            print(f"  {name:<15} {desc}")
        return

    # 统一切到项目根目录，保证 data/、results/ 等相对路径正确解析
    if os.getcwd() != ROOT:
        print(f"[train] 工作目录切换到项目根: {ROOT}")
        os.chdir(ROOT)

    config_path = args.config or f"configs/{args.model}.json"
    if not os.path.exists(config_path):
        sys.exit(f"[train] 找不到配置文件: {config_path}")

    print(f"[train] 变体={args.model}  配置={config_path}")

    if args.model == "fixed_discrete":
        from rotfit.models import fixed_discrete

        # fixed_discrete 依次跑「强制离散」和「软轴」两种方式做对比
        fixed_discrete.train_fixed_discrete(config_path, force_discrete=True)
        fixed_discrete.train_fixed_discrete(config_path, force_discrete=False)
        return

    mod_name, fn_name = MODELS[args.model]
    module = __import__(mod_name, fromlist=[fn_name])
    train_fn = getattr(module, fn_name)
    train_fn(config_path)

    # 回读结果并打印摘要
    result_file = ""
    try:
        with open(config_path, encoding="utf-8") as f:
            result_file = json.load(f).get("output", {}).get("result_file", "")
    except Exception:
        pass
    _print_result_summary(result_file, args.model)


if __name__ == "__main__":
    main()
