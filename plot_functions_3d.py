"""plot_functions_3d.py

用 3D 曲面图可视化 benchmark.py 中的 10 个测试函数。

说明
- 3D 图只能展示 2 个自变量，因此这里固定使用 D=2（x,y），绘制 z=f(x,y)。
- f7 含噪声项 random[0,1)，为保证可复现，脚本会固定随机种子。

用法
  python plot_functions_3d.py

输出
- 默认保存：output/test_functions_3d.png
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

from benchmark import FUNCTIONS


def eval_on_grid(func, x_range: tuple[float, float], y_range: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    xx, yy = np.meshgrid(xs, ys)

    X = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    zz = func(X).reshape(xx.shape)
    return xx, yy, zz


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制 10 个测试函数的 3D 曲面图（D=2）")
    parser.add_argument("--n", type=int, default=120, help="网格分辨率（每维点数），默认 120")
    parser.add_argument(
        "--out",
        default=os.path.join("output", "test_functions_3d.png"),
        help="输出图片路径，默认 output/test_functions_3d.png",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # f7 含噪声：固定随机种子，保证每次生成相同的 3D 图
    np.random.seed(0)

    fig = plt.figure(figsize=(20, 10))

    for i, func_info in enumerate(FUNCTIONS, start=1):
        name = func_info["name"]
        lower, upper = func_info["range"]

        ax = fig.add_subplot(2, 5, i, projection="3d")

        # 注意：对范围很大的函数（如 f8: [-500,500]）高分辨率会较慢。
        # 这里仍使用论文/定义给的范围，但你可以用 --n 调小分辨率。
        xx, yy, zz = eval_on_grid(func_info["func"], (lower, upper), (lower, upper), args.n)

        ax.plot_surface(
            xx,
            yy,
            zz,
            rstride=1,
            cstride=1,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("f(x,y)", fontsize=8)

        # 让子图更紧凑
        ax.tick_params(axis="both", which="major", labelsize=7)

    fig.suptitle("10 个测试函数的 3D 曲面图（D=2 截面）", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"已保存：{args.out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
