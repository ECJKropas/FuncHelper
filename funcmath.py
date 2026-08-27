"""
FuncHelper —— 核心数学模块（不依赖 GUI / tkinter）

负责两件事：
1. 屏幕坐标 <-> 用户自定义局部坐标系 的可逆仿射变换。
   用户点击的三个点：原点 O(0,0)、X 轴单位点 (1,0)、Y 轴单位点 (0,1)。
   这三个点连成的两条向量通常既不垂直、长度也不相等，
   因此不能直接当正交基，而要用一个 2x2 线性系统求解。

2. 给定局部坐标系下的 n 个点 (xi, yi)，拟合一个 n-1 次多项式
   y = p(x)，使其严格穿过这 n 个点（多项式插值）。
"""

from __future__ import annotations

import re

import numpy as np

# 退化判定用的容差
_DET_EPS = 1e-9


def compute_basis(O, Px, Py):
    """由三个屏幕坐标点构造坐标系基向量。

    参数
    ----
    O  : 原点 (0,0) 的屏幕坐标 [x, y]
    Px : X 轴单位点 (1,0) 的屏幕坐标 [x, y]
    Py : Y 轴单位点 (0,1) 的屏幕坐标 [x, y]

    返回
    ----
    (O, ex, ey)
        O  : ndarray[2]
        ex : 局部 x 单位向量在屏幕空间中的表示（Px - O）
        ey : 局部 y 单位向量在屏幕空间中的表示（Py - O）
    """
    O = np.asarray(O, dtype=float).reshape(2)
    Px = np.asarray(Px, dtype=float).reshape(2)
    Py = np.asarray(Py, dtype=float).reshape(2)
    ex = Px - O
    ey = Py - O
    return O, ex, ey


def screen_to_local(S, O, ex, ey):
    """把屏幕坐标 S 转换为局部坐标 [a, b]。

    几何关系：屏幕点 = O + a * ex + b * ey
    求解线性方程组  M * [a, b]^T = (S - O)，其中
        M = [[ex.x, ey.x],
             [ex.y, ey.y]]
    """
    S = np.asarray(S, dtype=float).reshape(2)
    O = np.asarray(O, dtype=float).reshape(2)
    ex = np.asarray(ex, dtype=float).reshape(2)
    ey = np.asarray(ey, dtype=float).reshape(2)

    d = S - O
    det = ex[0] * ey[1] - ey[0] * ex[1]
    if abs(det) < _DET_EPS:
        raise ValueError(
            "坐标系退化：原点、X 轴单位点、Y 轴单位点共线，"
            "无法构造可逆变换。请重新选择三个不共线的点。"
        )
    a = (ey[1] * d[0] - ey[0] * d[1]) / det
    b = (-ex[1] * d[0] + ex[0] * d[1]) / det
    return np.array([a, b], dtype=float)


def local_to_screen(local, O, ex, ey):
    """局部坐标 [a, b] 还原为屏幕坐标（供绘制/校验使用）。"""
    local = np.asarray(local, dtype=float).reshape(2)
    O = np.asarray(O, dtype=float).reshape(2)
    ex = np.asarray(ex, dtype=float).reshape(2)
    ey = np.asarray(ey, dtype=float).reshape(2)
    return O + local[0] * ex + local[1] * ey


def fit_polynomial(local_points, deg=None):
    """对 n 个局部点做 n-1 次多项式插值，返回升幂系数。

    参数
    ----
    local_points :  iterable of (x, y)，局部坐标系下的点
    deg          :  多项式次数，默认 n-1

    返回
    ----
    coeffs : ndarray，升幂排列，即 p(x) = coeffs[0] + coeffs[1] x + ... + coeffs[k] x^k

    说明
    ----
    - 要求所有 x 坐标互不相同，否则无法表示为单值函数 y = f(x)。
    - 通过解 Vandermonde 线性方程组得到精确插值（非最小二乘）。
    """
    pts = [tuple(map(float, p)) for p in local_points]
    n = len(pts)
    if n < 1:
        raise ValueError("至少需要 1 个采集点才能构造函数。")

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # 检查 x 是否唯一（允许极小浮点误差）
    rounded = [round(x, 9) for x in xs]
    if len(set(rounded)) != n:
        # 找出重复 x 的索引，便于提示
        seen = {}
        dup = None
        for i, x in enumerate(rounded):
            if x in seen:
                dup = x
                break
            seen[x] = i
        raise ValueError(
            f"存在相同的 x 坐标（x={dup}），无法表示为单值函数 y=f(x)。"
            f"请保证每个采集点的局部 x 坐标互不相同。"
        )

    if deg is None:
        deg = n - 1
    if deg < 0:
        raise ValueError("多项式次数不能为负。")
    if deg > n - 1:
        # 次数过高无解（严格插值在 n 点时最高 n-1 次唯一）
        raise ValueError(
            f"采集了 {n} 个点，最多只能拟合 {n - 1} 次多项式；"
            f"请求的次数 {deg} 过大。"
        )

    # 用 n 个点构造 n 列 Vandermonde 矩阵（列: 1, x, x^2, ..., x^{n-1}），
    # 再取前 deg+1 列对应 <= deg 次项。
    A_full = np.vander(xs, N=n, increasing=True)  # n x n
    A = A_full[:, : deg + 1]
    b = np.asarray(ys, dtype=float)
    coeffs = np.linalg.solve(A, b)
    return coeffs


def poly_eval(coeffs, x):
    """用升幂系数计算多项式在 x 处的值（支持标量或数组）。"""
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        result = result + c * (x ** i)
    return result


def poly_to_str(coeffs, var: str = "x"):
    """把升幂系数格式化成易读表达式，例如 '2 x^3 - 1 x^2 + 0.5 x + 1'。"""
    coeffs = list(map(float, coeffs))
    deg = len(coeffs) - 1
    segs = []
    first = True
    for i in range(deg, -1, -1):
        c = coeffs[i]
        if abs(c) < 1e-9 and deg > 0:
            continue
        ac = abs(c)
        if i == 0:
            t = f"{ac:.6g}"
        elif i == 1:
            t = f"{ac:.6g}{var}"
        else:
            t = f"{ac:.6g}{var}^{i}"

        if first:
            seg = ("-" + t) if c < 0 else t
        else:
            seg = ("- " + t) if c < 0 else ("+ " + t)
        segs.append(seg)
        first = False
    return " ".join(segs) if segs else "0"


def multiline_expr(expr: str):
    """把单行表达式按加/减项拆成多行，便于悬浮球横条显示。

    例如 ' -0.5x^3 + 3x^2 - 2x + 1 ' 变成：
        -0.5x^3
         + 3x^2
         - 2x
         + 1
    折线段模式同理，每个 '+' 分隔的项各占一行。
    """
    # 按 ' + ' / ' - '（两侧带空格的运算符）切分并保留运算符
    parts = re.split(r"( [+-] )", expr)
    lines = []
    pending_op = ""
    for p in parts:
        if p == "":
            continue
        if re.fullmatch(r" [+-] ", p):  # 运算符，挂到下一行开头
            pending_op = p
            continue
        lines.append(pending_op + p)
        pending_op = ""
    return "\n".join(lines)


def _fmt(v):
    """数值格式化（与多项式保持一致的可读精度）。"""
    return f"{float(v):.6g}"


def _fmt_x_minus(xi):
    """把 'x - xi' 写成可读形式：xi>=0 时 'x-(xi)'，xi<0 时 'x+(|xi|)'，
    避免出现 'x-(-2)' 这种减去负数的丑陋写法。"""
    v = float(xi)
    if v >= 0:
        return f"x-({_fmt(v)})"
    return f"x+({_fmt(-v)})"


def _fmt_factor(c):
    """把数值 c 格式化为乘积因子：c 为 1 时省略（返回 ''），否则返回 'c*'。"""
    if abs(float(c) - 1.0) < 1e-9:
        return ""
    return _fmt(c) + "*"


def build_piecewise_expr(local_points):
    """构造穿过 n 个点的连续分段线性函数表达式（折线段模式）。

    思路（与用户给出的样例一致）：
    先按 x 排序，基础项为 y0 + m0*(x - x0)，
    之后对每个内部节点 xi，用 (mi - m_{i-1}) * max(x - xi, 0)
    修正斜率；而 max(x - xi, 0) = (x - xi + |x - xi|) / 2。

    系数一律取绝对值，符号作为前导 ' + ' / ' - '，避免出现 '(-5)*' 或
    'x-(-2)' 这类减去负数的丑陋写法。
    返回 'f(x) =' 之后的表达式字符串（不含 'y =' 前缀）。
    """
    pts = [tuple(map(float, p)) for p in local_points]
    n = len(pts)
    if n < 1:
        raise ValueError("至少需要 1 个采集点。")

    pts.sort(key=lambda p: p[0])
    xs = [p[0] for p in pts]
    rounded = [round(x, 9) for x in xs]
    if len(set(rounded)) != n:
        raise ValueError(
            "存在相同的 x 坐标，折线段模式无法构造（分段需要互异的 x）。"
        )

    # 单点 -> 常数
    if n == 1:
        return _fmt(pts[0][1])

    x0, y0 = pts[0]
    slopes = []
    for i in range(n - 1):
        xi, yi = pts[i]
        xj, yj = pts[i + 1]
        slopes.append((yj - yi) / (xj - xi))

    # parts: (sign, body)，sign 取 '' / '+' / '-'
    parts = []
    if abs(float(y0)) > 1e-9:
        parts.append(("", _fmt(y0)))

    if abs(slopes[0]) > 1e-12:
        body = f"{_fmt_factor(abs(slopes[0]))}({_fmt_x_minus(x0)})"
        parts.append(("+" if slopes[0] >= 0 else "-", body))

    for i in range(1, n - 1):  # 内部节点
        delta = slopes[i] - slopes[i - 1]
        if abs(delta) > 1e-12:
            xi = pts[i][0]
            # 把 max(x-xi,0)=(x-xi+|x-xi|)/2 的 /2 折进前面的系数
            coeff = abs(delta) / 2.0
            body = (
                f"{_fmt_factor(coeff)}(({_fmt_x_minus(xi)})"
                f"+abs({_fmt_x_minus(xi)}))"
            )
            parts.append(("+" if delta >= 0 else "-", body))

    if not parts:
        return "0"

    first_sign, first_body = parts[0]
    if first_sign == "-":
        segs = ["-" + first_body]
    else:
        # 首项不加前导 '+'（可能是被省去零常数项后的斜率项）
        segs = [first_body]
    for sign, body in parts[1:]:
        segs.append((" + " if sign == "+" else " - ") + body)
    return "".join(segs)


if __name__ == "__main__":
    # 简单自检
    O, ex, ey = compute_basis([100, 100], [200, 100], [100, 50])
    local = screen_to_local([300, -200], O, ex, ey)
    print("逆变换示例，应为 [2. 3.]:", local)
    print("正变换还原:", local_to_screen(local, O, ex, ey))

    pts = [(0.0, 1.0), (1.0, 0.0), (2.0, 1.0)]
    coeffs = fit_polynomial(pts)
    print("插值系数(升幂):", coeffs)
    print("表达式 y =", poly_to_str(coeffs))
    print("在 x=1 处的值(应为 0):", poly_eval(coeffs, 1.0))
