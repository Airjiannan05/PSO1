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
        boundary='random',
    record_curve=False,
    v_max_factor=0.2,
    riw_mode='decay_upper',
    riw_w_min=None,
    riw_w_max=None):
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
    boundary    : 边界处理策略
                  'absorb'  - 吸收：超界后截断位置，速度清零（原方案）
                  'reflect' - 反弹：超界后镜像位置，速度取反（保留动能）
                  'random'  - 随机：超界后随机重置位置和速度（最强探索）
    record_curve: 是否记录每代最优值（用于绘制进化曲线）
    v_max_factor : 速度上限比例，v_max=(upper-lower)*v_max_factor
    riw_mode     : RIW 版本
                   'uniform'     - 每代在 [riw_w_min, riw_w_max] 里均匀采样
                   'decay_upper' - 上界随迭代线性递减到下界
    riw_w_min/max: RIW 的采样区间；为 None 时沿用 w_min/w_max

    Returns
    -------
    gbest_val   : 全局最优适应值
    curve       : 每代最优值列表（record_curve=True时返回）
    """
    lower, upper = bounds
    v_max = (upper - lower) * float(v_max_factor)   # Vmax = 搜索范围的一定比例

    # RIW 的权重区间（若未单独指定，则沿用通用 w_min/w_max）
    riw_min = w_min if riw_w_min is None else riw_w_min
    riw_max = w_max if riw_w_max is None else riw_w_max

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
            if riw_mode == 'uniform':
                # 每个粒子分配1个独立的权重 (shape=(n_particles, 1))
                w = omega_riw(n_particles, 1, riw_min, riw_max)
            elif riw_mode == 'decay_upper':
                # 随着迭代进展，逐渐降低随机权重的上界，使其整体趋势递减
                w_upper = riw_max - (riw_max - riw_min) * T / T_max
                w = omega_riw(n_particles, 1, riw_min, w_upper)
            else:
                raise ValueError(f"Unknown riw_mode: {riw_mode}")
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

   
        # 速度修正
        mask_lower = X <= lower
        mask_upper = X >= upper
        out_of_bounds = mask_lower | mask_upper
        # 1. 吸收 (Absorb): 贴在墙上，速度归零
        if boundary == 'absorb':
            X = np.clip(X, lower, upper)
            V[out_of_bounds] = 0.0

        # 2. 反弹 (Reflect): 像镜面一样折返位置，动能保留但方向取反
        elif boundary == 'reflect':
            X[mask_lower] = 2 * lower - X[mask_lower]
            X[mask_upper] = 2 * upper - X[mask_upper]

            # 极小概率下，如果速度极大导致反弹后依然越界，做一次截断兜底
            X = np.clip(X, lower, upper)
            
            # 速度反向
            V[out_of_bounds] *= -1.0

        # 3. 随机 (Random): 超界后随机重置位置和速度，增强探索能力
        elif boundary == 'random':
            # np.sum(mask_lower) 能求出有多少个维度越界，直接生成对应数量的随机数
            n_lower = np.sum(mask_lower)
            n_upper = np.sum(mask_upper)
            n_out   = np.sum(out_of_bounds)
            
            if n_lower > 0:
                X[mask_lower] = np.random.uniform(lower, upper, n_lower)
            if n_upper > 0:
                X[mask_upper] = np.random.uniform(lower, upper, n_upper)
            if n_out > 0:
                V[out_of_bounds] = np.random.uniform(-v_max, v_max, n_out)
                
        else:
            raise ValueError(f"未知的边界处理策略: {boundary}")

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
