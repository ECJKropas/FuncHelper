"""
FuncHelper 核心数学模块单元测试（无需 GUI / 显示器）。

运行：
    venv/bin/python test_math.py
"""

import math

import numpy as np

import funcmath as fm


def approx(a, b, tol=1e-6):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert np.allclose(a, b, atol=tol), f"期望 {b}，实际 {a}"


def test_basis_and_inverse():
    # 非正交、非等长基：x 单位向右 100px，y 单位向上 100px（屏幕 y 向下）
    O, ex, ey = fm.compute_basis([100, 100], [200, 100], [100, 50])

    # 局部 (2, 3) -> 屏幕（ey=[0,-50]，故 y=100+3*(-50)=-50）
    screen = fm.local_to_screen([2, 3], O, ex, ey)
    approx(screen, [300, -50])

    # 屏幕 -> 局部，应还原
    back = fm.screen_to_local(screen, O, ex, ey)
    approx(back, [2, 3])

    # 原点本身
    approx(fm.screen_to_local(O, O, ex, ey), [0, 0])
    # x 单位点
    approx(fm.screen_to_local([200, 100], O, ex, ey), [1, 0])
    # y 单位点
    approx(fm.screen_to_local([100, 50], O, ex, ey), [0, 1])


def test_degenerate_basis():
    # 三点共线 -> 应抛错
    O, ex, ey = fm.compute_basis([0, 0], [10, 10], [20, 20])
    try:
        fm.screen_to_local([5, 5], O, ex, ey)
        raise AssertionError("退化基未报错")
    except ValueError:
        pass


def test_fit_constant():
    # 1 个点 -> 0 次（常数）函数
    coeffs = fm.fit_polynomial([(3.0, 7.0)])
    approx(coeffs, [7.0])
    assert abs(fm.poly_eval(coeffs, 999.0) - 7.0) < 1e-9


def test_fit_line():
    # 2 个点 -> 线性：过 (0,1) 与 (2,5) => y = 2x + 1
    coeffs = fm.fit_polynomial([(0.0, 1.0), (2.0, 5.0)])
    approx(coeffs, [1.0, 2.0])
    assert abs(fm.poly_eval(coeffs, 1.0) - 3.0) < 1e-9


def test_fit_quadratic():
    # 3 个点 -> 二次：取 y = x^2 - x + 1 上的三点
    pts = [(-1.0, 3.0), (0.0, 1.0), (2.0, 3.0)]
    coeffs = fm.fit_polynomial(pts)
    approx(coeffs, [1.0, -1.0, 1.0], tol=1e-9)
    for x in (-1.0, 0.0, 0.5, 2.0):
        assert abs(fm.poly_eval(coeffs, x) - (x * x - x + 1)) < 1e-9


def test_fit_high_degree_extra_points():
    # 4 个点 -> 三次多项式插值
    pts = [(0.0, 1.0), (1.0, 0.0), (2.0, -1.0), (3.0, 4.0)]
    coeffs = fm.fit_polynomial(pts)
    for (x, y) in pts:
        assert abs(fm.poly_eval(coeffs, x) - y) < 1e-9


def test_duplicate_x_rejected():
    try:
        fm.fit_polynomial([(1.0, 2.0), (1.0, 5.0)])
        raise AssertionError("重复 x 未报错")
    except ValueError as e:
        assert "x" in str(e).lower() or "x" in str(e)


def test_poly_to_str():
    s = fm.poly_to_str([1.0, -2.0, 3.0])  # 3x^2 - 2x + 1
    assert s == "3x^2 - 2x + 1", s
    s2 = fm.poly_to_str([-1.0])  # -1
    assert s2 == "-1", s2


def test_build_piecewise_through_points():
    # 分段线性应穿过每个点（升/降序都应正确，内部会按 x 排序）
    pts = [(0.0, 1.0), (2.0, 5.0), (3.0, 2.0), (-1.0, 0.0)]
    expr = fm.build_piecewise_expr(pts)
    env = {"__builtins__": {}, "abs": abs}
    for x, y in pts:
        val = eval(expr, env, {"x": x})  # noqa: S307 (受控测试输入)
        assert abs(val - y) < 1e-9, f"x={x}: 期望 {y}，得到 {val}"
    # 单点 -> 常数
    assert fm.build_piecewise_expr([(4.0, 9.0)]) == "9", fm.build_piecewise_expr([(4.0, 9.0)])


def test_piecewise_duplicate_x_rejected():
    try:
        fm.build_piecewise_expr([(1.0, 2.0), (1.0, 5.0)])
        raise AssertionError("折线段重复 x 未报错")
    except ValueError:
        pass


def test_piecewise_large_coords_precision():
    # 回归：屏幕像素级大坐标 + 高精度格式化，字符串必须仍严格穿过各点。
    # 旧实现用 6 位有效数字，大坐标被斜率放大后不再穿点（误报「未穿过」）。
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(2, 6))
        xs = np.sort(rng.uniform(-2000, 2000, n))
        ys = rng.uniform(-2000, 2000, n)
        pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
        expr = fm.build_piecewise_expr(pts)
        env = {"__builtins__": {}, "abs": abs}
        for x, y in pts:
            val = eval(expr, env, {"x": x})  # noqa: S307 (受控测试输入)
            assert abs(val - y) < 1e-6, f"x={x}: 期望 {y}，得到 {val}\nexpr={expr}"


def main():
    tests = [
        test_basis_and_inverse,
        test_degenerate_basis,
        test_fit_constant,
        test_fit_line,
        test_fit_quadratic,
        test_fit_high_degree_extra_points,
        test_duplicate_x_rejected,
        test_poly_to_str,
        test_build_piecewise_through_points,
        test_piecewise_duplicate_x_rejected,
        test_piecewise_large_coords_precision,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过 ✅")
    # 顺带验证 :math 导入正常
    assert math.pi > 0


if __name__ == "__main__":
    main()
