"""
PSO 核心实现 + 4种惯量权重控制方法
  'LDW'    - 线性递减惯量权重
  'RIW'    - 随机惯量权重
  'CONCAVE'- 凹函数递减惯量权重
  'CONVEX' - 凸函数递减惯量权重
"""

import numpy as np


# ──────────────────────────────────────────────
# 4种 ω 计算函数
# ──────────────────────────────────────────────

def omega_ldw(T, T_max, w_min=0.4, w_max=0.9):
    """线性递减法 (LDW)"""
    return w_max - (w_max - w_min) * T / T_max


def omega_riw(n_particles, D=1, w_min=0.4, w_max=0.6):
    """随机惯量权重法 (RIW) - 每个粒子独立随机取值，shape=(n_particles,1)"""
    return np.random.uniform(w_min, w_max, (n_particles, 1))


def omega_concave(T, T_max, w_min=0.4, w_max=0.9):
    """凹函数递减法 - 公式(6)"""
    t = T / T_max
    return -(w_max - w_min) * t ** 2 + w_max


def omega_convex(T, T_max, w_min=0.4, w_max=0.9):
    """凸函数递减法 - 公式(7)"""
    t = T / T_max
    return (w_max - w_min) * (t - 1) ** 2 + w_min

# ──────────────────────────────────────────────
# PSO 主体
# ──────────────────────────────────────────────

def pso(func, bounds, D, method='LDW',
        n_particles=20, T_max=2000,
        c1=2.0, c2=2.0,
        w_min=0.4, w_max=0.9,
        record_curve=False):
    """
    PSO 优化器

    Parameters
    ----------
    func        : 目标函数（最小化）
    bounds      : (lower, upper) 搜索范围
    D           : 维度
    method      : 'LDW' | 'RIW' | 'CONCAVE' | 'CONVEX'
    n_particles : 粒子数（默认20）
    T_max       : 最大迭代次数（默认2000）
    c1, c2      : 学习因子（默认2.0）
    w_min,w_max : 惯量权重上下界
    record_curve: 是否记录每代最优值（用于绘制进化曲线）

    Returns
    -------
    gbest_val   : 全局最优适应值
    curve       : 每代最优值列表（record_curve=True时返回）
    """
    lower, upper = bounds
    v_max = (upper - lower) * 0.2   # Vmax = 搜索范围的20%

    # ── 初始化 ──
    X = np.random.uniform(lower, upper, (n_particles, D))
    V = np.random.uniform(-v_max, v_max, (n_particles, D))

    pbest_pos = X.copy()
    pbest_val = func(X)

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    curve = []
    if record_curve:
        curve.append(gbest_val)  # 第0代：初始状态

    # ── 迭代 ──
    for T in range(1, T_max + 1):

        # 计算当前 ω
        if method == 'LDW':
            w = omega_ldw(T, T_max, w_min, w_max)
        elif method == 'RIW':
            w = omega_riw(n_particles, 1, 0.4, 0.6)   # 每个粒子分配1个独立的权重 (shape=(n_particles, 1))
        elif method == 'CONCAVE':
            w = omega_concave(T, T_max, w_min, w_max)
        elif method == 'CONVEX':
            w = omega_convex(T, T_max, w_min, w_max)
        else:
            raise ValueError(f"Unknown method: {method}")

        r1 = np.random.uniform(0, 1, (n_particles, D))
        r2 = np.random.uniform(0, 1, (n_particles, D))

        # 速度更新
        V = (w * V
             + c1 * r1 * (pbest_pos - X)
             + c2 * r2 * (gbest_pos - X))

        # 速度限幅
        V = np.clip(V, -v_max, v_max)

        # 位置更新
        X = X + V

        # 位置限幅（边界处理：截断）
        X = np.clip(X, lower, upper)

        
        # 速度修正：让撞墙的粒子停下来，不要继续往墙外飞
        mask_lower = X <= lower
        mask_upper = X >= upper
        V[mask_lower | mask_upper] = 0.0  

        # 更新个体最优
        vals = func(X)
        improved = vals < pbest_val
        pbest_pos[improved] = X[improved]
        pbest_val[improved] = vals[improved]

        # 更新全局最优
        best_idx = np.argmin(pbest_val)
        if pbest_val[best_idx] < gbest_val:
            gbest_val = pbest_val[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()

        if record_curve:
            curve.append(gbest_val)

    if record_curve:
        return gbest_val, curve
    return gbest_val


METHODS = ['LDW', 'RIW', 'CONCAVE', 'CONVEX']
METHOD_LABELS = {
    'LDW':     '线性递减法',
    'RIW':     '随机权重法',
    'CONCAVE': '凹函数递减法',
    'CONVEX':  '凸函数递减法',
}
METHOD_COLORS = {
    'LDW':     'blue',
    'RIW':     'red',
    'CONCAVE': 'green',
    'CONVEX':  'orange',
}
METHOD_MARKERS = {
    'LDW':     's',
    'RIW':     'o',
    'CONCAVE': '^',
    'CONVEX':  'D',
}
