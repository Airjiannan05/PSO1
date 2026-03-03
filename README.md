# PSO 惯量权重对比实验

基于粒子群优化（PSO）的四种惯量权重策略对比实现，在 10 个标准测试函数上进行实验，对应论文 Table 2 / Table 3 / Fig. 4。

## 四种惯量权重策略

| 标识 | 名称 | 公式 |
|------|------|------|
| `LDW` | 线性递减 | ω = ω_max − (ω_max − ω_min) · t |
| `RIW` | 随机权重 | ω ~ U(ω_min, ω_max(t))，上界随迭代线性递减 |
| `CONCAVE` | 凹函数递减 | ω = −(ω_max − ω_min) · t² + ω_max |
| `CONVEX` | 凸函数递减 | ω = (ω_max − ω_min) · (t−1)² + ω_min |

其中 t = T / T_max ∈ [0, 1] 为归一化迭代进度，ω_min = 0.4，ω_max = 0.9。

## 测试函数

| 编号 | 函数名 | 搜索范围 | 最优值 | 类型 |
|------|--------|----------|--------|------|
| f1  | Sphere        | [−100, 100]^D  | 0                  | 单峰 |
| f2  | Schwefel 2.22 | [−10, 10]^D    | 0                  | 单峰 |
| f3  | Schwefel 1.2  | [−100, 100]^D  | 0                  | 单峰 |
| f4  | Schwefel 2.21 | [−100, 100]^D  | 0                  | 单峰 |
| f5  | Rosenbrock    | [−30, 30]^D    | 0                  | 单峰 |
| f6  | Step          | [−100, 100]^D  | 0                  | 单峰 |
| f7  | Quartic+Noise | [−1.28, 1.28]^D | 0                 | 单峰 |
| f8  | Schwefel 2.26 | [−500, 500]^D  | −418.9829 × D      | 多峰 |
| f9  | Rastrigin     | [−5.12, 5.12]^D | 0                 | 多峰 |
| f10 | Ackley        | [−32, 32]^D    | 0                  | 多峰 |

## 环境准备

**Python 3.8+**，安装依赖：

```bash
pip install numpy matplotlib
```

## 使用步骤

### 第一步：（可选）可视化测试函数三维形貌

```bash
python plot_functions_3d.py
```

输出 `output/test_functions_3d.png`，展示所有 10 个函数在 D=2 时的 3D 曲面图。

### 第二步：运行实验

```bash
python experiment.py
```

- 自动在 **10 维**和 **30 维**上各跑 100 次独立重复实验
- `ProcessPoolExecutor` 多核并行加速（自动使用 CPU 核心数）
- 种子规则：`seed = fi × 10000 + run`，与方法无关，保证四种方法在同一 run 下使用相同随机初始化（控制变量）
- 结果自动保存至 `output/exN/results.pkl`（N 自增，不覆盖历史结果）
- 同步生成带时间戳的日志文件 `output/exN/experiment_YYYYMMDD_HHMMSS.log`
- 生成 `output/exN/notice.md`，记录本次实验的全部关键配置

### 第三步：生成图表和表格

```bash
# 自动选择最新 exN 目录
python plot_results.py

# 或手动指定目录
python plot_results.py --exp-dir output/ex6

# 不弹出图窗（仅保存文件，适合批处理）
python plot_results.py --no-show
```

在指定的 `output/exN/` 目录下生成：

| 文件 | 内容 |
|------|------|
| `table_10D.png` | 论文 Table 2：10 维均值 / 标准差结果表，最优值粗体蓝色标注 |
| `table_30D.png` | 论文 Table 3：30 维均值 / 标准差结果表 |
| `evolution_curves.png` | 论文 Fig. 4：f1/f6/f9/f10 四个函数的进化曲线（100 次均值，对数坐标）|
| `omega_comparison.png` | 四种惯量权重随迭代的变化曲线 |

## 项目结构

```
PSO1/
├── pso.py               # PSO 核心算法 + 4 种 ω 计算策略
├── benchmark.py         # 10 个标准测试函数（向量化，支持批量粒子输入）
├── experiment.py        # 实验主程序（并行运行、数据收集、结果保存）
├── plot_results.py      # 结果可视化（Table 2/3 + Fig. 4 + ω 变化图）
├── plot_functions_3d.py # 10 个测试函数的 3D 曲面可视化
└── output/
    ├── test_functions_3d.png   # 测试函数曲面图（plot_functions_3d.py 生成）
    └── exN/                    # 每次 experiment.py 运行自动创建（N 自增）
        ├── results.pkl              # 实验数据（results + curves）
        ├── notice.md                # 本次实验配置记录
        ├── experiment_*.log         # 完整运行日志
        ├── table_10D.png
        ├── table_30D.png
        ├── evolution_curves.png
        └── omega_comparison.png
```

## 实验参数

| 参数 | 值 |
|------|----|
| 粒子数 N | 20 |
| 最大迭代次数 T_max | 2000 |
| 学习因子 c1, c2 | 2.0 |
| 惯量权重范围 [ω_min, ω_max] | [0.4, 0.9] |
| 速度上限 v_max | (upper − lower) × 0.2 |
| 边界处理策略 | random（超界截断位置，超界后随机重置位置和速度，增强探索能力）|
| RIW 模式 | decay_upper（随机上界随迭代线性递减）|
| 独立运行次数 | 100 |
| 测试维度 | 10D, 30D |

## 进化曲线数据说明

Fig. 4 展示的每条曲线为 **100 次独立运行的逐代全局最优值均值**，仅取 10 维结果：

```
mean_curve[t] = (1/100) × Σ gbest_i(t),  t = 0, 1, ..., 2000
```

其中 `gbest_i(t)` 为第 i 次运行在第 t 代结束时的全局最优适应值（单调不增）。  
Fig. 4 选取的四个函数为：**f1 Sphere**、**f6 Step**、**f9 Rastrigin**、**f10 Ackley**。
