"""
实验主程序

实验设置：
  - 粒子数：20
  - c1 = c2 = 2.0
  - 最大迭代：2000 代
  - 每个函数独立运行 100 次
  - 维度：10维 和 30维
  - 记录：均值 + 标准差
"""

import numpy as np
import pickle
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmark import FUNCTIONS
from pso import pso, METHODS, METHOD_LABELS


class Tee:
    """同时向终端和日志文件写入输出"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def fmt_val(v):
    """紧凑格式化浮点数（4位有效数字）"""
    if v == 0:
        return '0'
    av = abs(v)
    if 1e-3 <= av < 1e4:
        return f'{v:.4g}'
    return f'{v:.4e}'

# ── 实验参数 ──
N_RUNS      = 100
T_MAX       = 2000
N_PARTICLES = 20
C1 = C2     = 2.0
DIMS        = [10, 30]

OUTPUT_ROOT  = "output"
RESULTS_FILE = "results.pkl"


def next_experiment_dir():
    """在 output/ 下找到下一个可用的 exN 目录并创建"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    idx = 0
    while os.path.exists(os.path.join(OUTPUT_ROOT, f"ex{idx}")):
        idx += 1
    out_dir = os.path.join(OUTPUT_ROOT, f"ex{idx}")
    os.makedirs(out_dir)
    return out_dir


def run_single(args):
    """单次运行（用于并行）—— 接收索引而非对象，避免 pickle 问题"""
    fi, method, D, seed, record_curve = args
    func_info = FUNCTIONS[fi]
    np.random.seed(seed)  # 在子进程中严格设置随机种子
    return pso(
        func=func_info["func"],
        bounds=func_info["range"],
        D=D,
        method=method,
        n_particles=N_PARTICLES,
        T_max=T_MAX,
        c1=C1, c2=C2,
        record_curve=record_curve
    )


def run_all():
    """
    运行全部实验（均值/方差 + 进化曲线一体化，真正并行）

    种子规则：seed = fi * 10000 + run
      - 只与"测试函数索引"和"运行批次"绑定，不含 method
      - 同一 run 内所有方法使用相同种子 → 控制变量，公平对比

    并行策略：在 N_RUNS 层级上使用 ProcessPoolExecutor
      - 10D 顺便记录进化曲线，与统计数据完全同源

    Returns
    -------
    results : dict  results[D][fname][method] = {"mean", "std", "all"}
    curves  : dict  curves[fname][method] = array(T_MAX+1,)  仅 10D
    """
    results = {}
    curves = {}
    total_tasks = len(DIMS) * len(FUNCTIONS) * len(METHODS)
    task_idx = 0
    max_workers = os.cpu_count() or 4

    for D in DIMS:
        results[D] = {}
        record_curve = (D == 10)  # 仅 10D 记录曲线
        dim_start = time.time()
        print(f"\n{'─'*60}", flush=True)
        print(f"  维度 = {D}D", flush=True)
        print(f"{'─'*60}", flush=True)

        for fi, func_info in enumerate(FUNCTIONS):
            fname = func_info["name"]
            results[D][fname] = {}
            if record_curve:
                curves[fname] = {}

            for method in METHODS:
                task_idx += 1
                label = METHOD_LABELS[method]
                t0 = time.time()
                print(f"  [{task_idx:>3}/{total_tasks}] [{D}D] {fname:<6}  {label:<20} ", end='', flush=True)

                # 种子只与 (fi, run) 绑定，不含 method —— 控制变量法
                tasks = [
                    (fi, method, D, fi * 10000 + run, record_curve)
                    for run in range(N_RUNS)
                ]

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    run_results = list(executor.map(run_single, tasks))

                if record_curve:
                    vals = np.array([r[0] for r in run_results])
                    all_curves = [r[1] for r in run_results]
                    curves[fname][method] = np.mean(all_curves, axis=0)
                else:
                    vals = np.array(run_results)

                elapsed = time.time() - t0
                print(
                    f"完成  {elapsed:5.1f}s | "
                    f"mean={fmt_val(vals.mean())}  std={fmt_val(vals.std())}  "
                    f"min={fmt_val(vals.min())}  max={fmt_val(vals.max())}  "
                    f"median={fmt_val(np.median(vals))}",
                    flush=True
                )
                GROUP = 10
                for g in range(0, N_RUNS, GROUP):
                    row = vals[g:g + GROUP]
                    nums = "  ".join(f"{r+g+1:>3}:{fmt_val(v)}" for r, v in enumerate(row))
                    print(f"    {nums}", flush=True)

                results[D][fname][method] = {
                    "mean": vals.mean(),
                    "std":  vals.std(),
                    "all":  vals,
                }

        dim_elapsed = time.time() - dim_start
        print(f"  [{D}D] 全部完成，耗时 {dim_elapsed/60:.1f} 分钟", flush=True)

    return results, curves

if __name__ == "__main__":
    # 确定本次实验输出目录（output/ex0, ex1, ...）
    out_dir = next_experiment_dir()

    # 日志文件保存到实验目录
    log_filename = os.path.join(out_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    tee = Tee(log_filename)
    sys.stdout = tee

    wall_start = time.time()
    start_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 60)
    print(f"  PSO 惯量权重对比实验")
    print(f"  开始时间: {start_str}")
    print(f"  参数: N_RUNS={N_RUNS}, T_MAX={T_MAX}, N_PARTICLES={N_PARTICLES}, c1=c2={C1}")
    print(f"  维度: {DIMS}")
    print(f"  输出目录: {out_dir}")
    print(f"  日志文件: {log_filename}")
    print("=" * 60)

    results, curves = run_all()

    # 保存结果到实验目录
    results_path = os.path.join(out_dir, RESULTS_FILE)
    with open(results_path, "wb") as f:
        pickle.dump({"results": results, "curves": curves}, f)

    total_elapsed = time.time() - wall_start
    end_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"  结束时间: {end_str}")
    print(f"  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  实验结果已保存至 {results_path}")
    print(f"  日志已保存至 {log_filename}")
    print("=" * 60)
    print(f"运行 plot_results.py 生成图表和表格（默认读取最新实验目录）")

    sys.stdout = tee.terminal
    tee.close()
