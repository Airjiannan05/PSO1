"""
结果可视化
  - Table 2: 10维结果表
  - Table 3: 30维结果表
  - Fig 4: 四个典型函数的进化曲线（f1, f6, f9, f10）
"""

import pickle
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib import font_manager

from benchmark import FUNCTIONS
from pso import METHODS, METHOD_LABELS, METHOD_COLORS, METHOD_MARKERS

RESULTS_FILE = "results.pkl"

# 论文中函数的简写标签
FUNC_LABELS = ["$f_1$","$f_2$","$f_3$","$f_4$","$f_5$",
               "$f_6$","$f_7$","$f_8$","$f_9$","$f_{10}$"]

METHOD_CN = {
    'LDW':     '线性递减法',
    'RIW':     '随机权重法',
    'CONCAVE': '凹函数递减法',
    'CONVEX':  '凸函数递减法',
}


def load_data():
    with open(RESULTS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["results"], data["curves"]


def _fmt(val):
    """将数值格式化为论文样式的科学计数法字符串"""
    if val == 0.0:
        return "0"
    # 用 E 格式，去掉多余的0，保留4位有效数字
    s = f"{val:.4E}"
    # 如 1.6350E-44 → 1.635E-44
    mantissa, exp = s.split("E")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp_int = int(exp)
    return f"{mantissa}E{exp_int:+03d}".replace("+0", "+").replace("-0", "-").replace("E+", "E").replace("E-0", "E-").replace("E+0", "E")


def _fmt2(val):
    """4位有效数字科学计数法，与论文格式一致"""
    if val == 0.0:
        return "0"
    exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    if -4 <= exp <= 4:
        # 普通十进制，4位有效数字
        sig = 4 - exp - 1 if exp >= 0 else 4
        return f"{val:.{max(sig,0)}f}"
    else:
        return f"{val:.3E}"


# ──────────────────────────────────────────────
# 绘制论文格式表格图片（对应 Table 2 / Table 3）
# ──────────────────────────────────────────────

def plot_table(results, D, save_path=None):
    """
    绘制与论文完全一致格式的结果表格图片：
      - 双层表头：方法名 / 平均值+标准差
      - 粗体标注每行最优均值
      - 科学计数法数值
    """
    n_funcs = len(FUNCTIONS)

    # ── 准备单元格数据 ──
    cell_texts = []
    cell_bold  = []     # 哪些(行,列)需要粗体

    for fi, func_info in enumerate(FUNCTIONS):
        fname = func_info["name"]
        means = {m: results[D][fname][m]["mean"] for m in METHODS}
        stds  = {m: results[D][fname][m]["std"]  for m in METHODS}
        best_method = min(METHODS, key=lambda m: means[m])

        row_texts = [FUNC_LABELS[fi]]
        row_bold  = [False]

        def fmt_cell(v):
            if v == 0.0:
                return "0"
            abs_v = abs(v)
            if 0.001 <= abs_v < 100000:
                # 不需要科学计数法，用普通小数
                s = f"{v:.4g}"
                return s
            else:
                s = f"{v:.3E}"
                m_part, e_part = s.split("E")
                e_val = int(e_part)
                return f"{m_part}E{e_val}"

        for m in METHODS:
            mt = fmt_cell(means[m])
            st = fmt_cell(stds[m])
            is_best = (m == best_method)
            row_texts += [mt, st]
            row_bold  += [is_best, is_best]
        cell_texts.append(row_texts)
        cell_bold.append(row_bold)

    # ── 构建图形 ──
    fig_w = 18
    fig_h = 0.42 * (n_funcs + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    title_str = (f"表 {'2' if D==10 else '3'}  四种算法求得最优值的平均值和标准差"
                 f"（{D} 维函数）")
    ax.set_title(title_str, fontsize=13, fontweight='bold', pad=10, loc='center')

    col_widths = [0.055] + [0.117, 0.117] * 4

    header1 = ["函数",
               "线性递减法", "", "随机权重法", "", "凹函数递减法", "", "凸函数递减法", ""]
    header2 = ["", "平均值", "标准差", "平均值", "标准差",
               "平均值", "标准差", "平均值", "标准差"]

    all_rows = [header1, header2] + cell_texts
    n_cols = 9
    n_rows = len(all_rows)

    tbl = ax.table(
        cellText=all_rows,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#555555')
        cell.set_linewidth(0.6)

        if r == 0:
            cell.set_facecolor('#DDEEFF')
            cell.set_text_props(fontweight='bold', fontsize=9)
            if c in [1, 3, 5, 7]:
                cell.set_text_props(fontweight='bold', fontsize=9, ha='center')
            if c in [2, 4, 6, 8]:
                cell.get_text().set_text("")
        elif r == 1:
            cell.set_facecolor('#EEF4FF')
            cell.set_text_props(fontweight='bold', fontsize=8.5)
        else:
            fi = r - 2
            cell.set_facecolor('#FFFFFF' if fi % 2 == 0 else '#F8F8F8')
            if cell_bold[fi][c]:
                cell.set_text_props(fontweight='bold', color='#0033AA')

        if c == 0 and r >= 2:
            cell.set_text_props(ha='center')

    for r in range(n_rows):
        for c in range(n_cols):
            tbl[r, c].set_height(0.06 if r <= 1 else 0.055)

    plt.tight_layout(pad=0.5)
    save_path = save_path or f"table_{D}D.png"
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"表格已保存至 {save_path}")
    plt.close()


# ──────────────────────────────────────────────
# 保留原文本打印（快速查看）
# ──────────────────────────────────────────────

def print_table(results, D):
    print(f"\n{'='*100}")
    print(f"  表{'2' if D==10 else '3'}  四种算法 {D}维 结果（均值 / 标准差）  粗体=最优")
    print(f"{'='*100}")
    hdr = f"{'函数':<6}" + "".join(
        f"{'  '+METHOD_CN[m]+' 均值':>18}{'标准差':>16}" for m in METHODS)
    print(hdr)
    print("-" * 100)
    for fi, func_info in enumerate(FUNCTIONS):
        fname = func_info["name"]
        means = {m: results[D][fname][m]["mean"] for m in METHODS}
        stds  = {m: results[D][fname][m]["std"]  for m in METHODS}
        best  = min(means.values())
        row = f"f{fi+1:<5}"
        for m in METHODS:
            star = "*" if np.isclose(means[m], best, rtol=1e-3) else " "
            row += f"  {star}{means[m]:>14.4E}  {stds[m]:>14.4E}"
        print(row)
    print("=" * 100)
    print("* = 最优均值")


# ──────────────────────────────────────────────
# 绘制进化曲线（对应论文 Fig. 4）
# ──────────────────────────────────────────────

def plot_curves(curves):
    # 论文图4选的4个函数
    selected = ["f1  Sphere", "f6  Step", "f9  Rastrigin", "f10 Ackley"]
    labels   = ["(a) f1 Sphere", "(b) f6 Step", "(c) f9 Rastrigin", "(d) f10 Ackley"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    # 各子图合理的y轴下限，避免0/nan造成的极端动态范围
    y_mins = {
        "f1  Sphere":   1e-100,
        "f6  Step":     5e-3,    # Step函数整数值，100次均值最小约0.01
        "f9  Rastrigin":5e-1,
        "f10 Ackley":   1e-15,
    }

    for idx, (fname, label) in enumerate(zip(selected, labels)):
        ax = axes[idx]
        y_min = y_mins[fname]
        for method in METHODS:
            curve = np.array(curves[fname][method], dtype=float)
            # 将0或极小值替换为nan，让曲线在收敛到0时自然截断
            y = np.where(curve > y_min, curve, np.nan)
            ax.semilogy(
                range(1, len(y) + 1),
                y,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.5,
            )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("迭代次数", fontsize=10)
        ax.set_ylabel("最优适应值（对数）", fontsize=10)
        ax.set_ylim(bottom=y_min)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)

    fig.suptitle("图4  4种算法在典型10维函数上的进化曲线", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("evolution_curves.png", dpi=150, bbox_inches='tight')
    print("进化曲线已保存至 evolution_curves.png")
    plt.show()


# ──────────────────────────────────────────────
# 绘制 ω 随迭代变化图
# ──────────────────────────────────────────────

def plot_omega():
    T_max = 2000
    T_arr = np.arange(1, T_max + 1)
    t_arr = T_arr / T_max
    w_min, w_max = 0.4, 0.9

    omega = {
        'LDW':     w_max - (w_max - w_min) * t_arr,
        'CONCAVE': -(w_max - w_min) * t_arr ** 2 + w_max,
        'CONVEX':  (w_max - w_min) * (t_arr - 1) ** 2 + w_min,
    }
    # RIW 用均值线展示
    omega['RIW'] = np.full(T_max, 0.5)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        style = '--' if method == 'RIW' else '-'
        ax.plot(t_arr, omega[method],
                color=METHOD_COLORS[method],
                linestyle=style,
                label=METHOD_LABELS[method],
                linewidth=2)

    ax.set_xlabel("归一化迭代进度 t = T/T_max", fontsize=11)
    ax.set_ylabel("惯量权重 ω", fontsize=11)
    ax.set_title("4种惯量权重控制方法的变化曲线", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0.35, 0.95)
    plt.tight_layout()
    plt.savefig("omega_comparison.png", dpi=150)
    print("ω 变化曲线已保存至 omega_comparison.png")
    plt.show()


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":
    results, curves = load_data()

    # 打印文本结果（快速查看）
    print_table(results, 10)
    print_table(results, 30)

    # 绘制论文格式表格图片
    plot_table(results, 10, "table_10D.png")
    plot_table(results, 30, "table_30D.png")

    # 绘制 ω 变化图
    plot_omega()

    # 绘制进化曲线
    plot_curves(curves)
