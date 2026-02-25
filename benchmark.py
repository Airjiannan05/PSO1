"""
10个标准测试函数（向量化版本）
所有函数接受 shape=(N, D) 的矩阵，返回 shape=(N,) 的结果数组
"""
import numpy as np


# ──────────────────────────────────────────────
# 单峰函数 f1~f7
# ──────────────────────────────────────────────

def f1(X):
    """Sphere  |  range: [-100,100]^D  |  fmin=0"""
    return np.sum(X ** 2, axis=1)


def f2(X):
    """Schwefel 2.22  |  range: [-10,10]^D  |  fmin=0"""
    return np.sum(np.abs(X), axis=1) + np.prod(np.abs(X), axis=1)


def f3(X):
    """Schwefel 1.2  |  range: [-100,100]^D  |  fmin=0"""
    return np.sum(np.cumsum(X, axis=1) ** 2, axis=1)


def f4(X):
    """Schwefel 2.21  |  range: [-100,100]^D  |  fmin=0"""
    return np.max(np.abs(X), axis=1)


def f5(X):
    """Rosenbrock  |  range: [-30,30]^D  |  fmin=0"""
    return np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (X[:, :-1] - 1) ** 2, axis=1)


def f6(X):
    """Step  |  range: [-100,100]^D  |  fmin=0"""
    return np.sum(np.floor(X + 0.5) ** 2, axis=1)


def f7(X):
    """Quartic + Noise  |  range: [-1.28,1.28]^D  |  fmin=0"""
    N, D = X.shape
    i = np.arange(1, D + 1)
    return np.sum(i * X ** 4, axis=1) + np.random.uniform(0, 1, N)


# ──────────────────────────────────────────────
# 多峰函数 f8~f10
# ──────────────────────────────────────────────

def f8(X):
    """Schwefel 2.26  |  range: [-500,500]^D  |  fmin=-418.9829*D"""
    return np.sum(-X * np.sin(np.sqrt(np.abs(X))), axis=1)


def f9(X):
    """Rastrigin  |  range: [-5.12,5.12]^D  |  fmin=0"""
    return np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)


def f10(X):
    """Ackley  |  range: [-32,32]^D  |  fmin=0"""
    N, D = X.shape
    a, b, c = 20, 0.2, 2 * np.pi
    s1 = -a * np.exp(-b * np.sqrt(np.sum(X ** 2, axis=1) / D))
    s2 = -np.exp(np.sum(np.cos(c * X), axis=1) / D)
    return s1 + s2 + a + np.e


# ──────────────────────────────────────────────
# 函数元数据（名称、范围、最优值）
# ──────────────────────────────────────────────

FUNCTIONS = [
    {"func": f1,  "name": "f1  Sphere",        "range": (-100, 100), "fmin": 0.0,          "unimodal": True},
    {"func": f2,  "name": "f2  Schwefel 2.22", "range": (-10,  10),  "fmin": 0.0,          "unimodal": True},
    {"func": f3,  "name": "f3  Schwefel 1.2",  "range": (-100, 100), "fmin": 0.0,          "unimodal": True},
    {"func": f4,  "name": "f4  Schwefel 2.21", "range": (-100, 100), "fmin": 0.0,          "unimodal": True},
    {"func": f5,  "name": "f5  Rosenbrock",    "range": (-30,  30),  "fmin": 0.0,          "unimodal": True},
    {"func": f6,  "name": "f6  Step",          "range": (-100, 100), "fmin": 0.0,          "unimodal": True},
    {"func": f7,  "name": "f7  Quartic+Noise", "range": (-1.28,1.28),"fmin": 0.0,          "unimodal": True},
    {"func": f8,  "name": "f8  Schwefel 2.26", "range": (-500, 500), "fmin": lambda D: -418.9829 * D, "unimodal": False},
    {"func": f9,  "name": "f9  Rastrigin",     "range": (-5.12,5.12),"fmin": 0.0,          "unimodal": False},
    {"func": f10, "name": "f10 Ackley",        "range": (-32,  32),  "fmin": 0.0,          "unimodal": False},
]
