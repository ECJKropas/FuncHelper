"""
FuncHelper —— 桌面悬浮球工具（tkinter）

使用方式
--------
    venv/bin/python funchelper.py

操作流程
--------
1. 屏幕上出现一个可拖动的蓝色悬浮球，以及一个顶部提示条。
2. 左键点击悬浮球「开始」：进入标定模式，出现一层几乎透明的全屏捕获层。
   - 第一次左键点击：原点 (0,0)
   - 第二次左键点击：X 轴单位点 (1,0)
   - 第三次左键点击：Y 轴单位点 (0,1)
   （这三点连成的向量通常既不垂直、长度也不相等，程序会自动做仿射变换）
3. 之后进入采集模式：每次左键点击记录一个数据点（可采集任意多个）。
4. 全部点完后，左键点击悬浮球「结束」：程序把各点转换到局部坐标系，
   拟合一个 n-1 次多项式 y = f(x) 穿过这 n 个点，并从悬浮球右侧伸出横条
   显示函数字符串（左键点横条复制、右键关闭），同时保存 fit_result.json。

说明
----
- 悬浮球上右键 / 标定过程中点击悬浮球 = 取消并重置。
- 捕获层是透明的，下方内容清晰可见，点击即记录屏幕坐标。
- 所有数学计算见 funcmath.py。
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import messagebox

import numpy as np

import funcmath as fm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class FuncHelper:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏主窗口，只用悬浮球 + 捕获层

        self.state = "IDLE"  # IDLE / GET_ORIGIN / GET_X / GET_Y / COLLECT
        self.mode = "poly"    # "poly"(多项式模式) / "piecewise"(折线段模式)
        # 标定点
        self.O = None
        self.Px = None
        self.Py = None
        self.ex = None
        self.ey = None
        # 采集的屏幕点（像素坐标）
        self.points_screen = []
        # 结果横条（从悬浮球右侧伸出）
        self.result_bar = None

        # ---- 顶部提示条 ----
        self.hint = tk.Toplevel(self.root)
        self.hint.overrideredirect(True)
        self.hint.attributes("-topmost", True)
        self.hint.configure(bg="#222222")
        self.hint_label = tk.Label(
            self.hint,
            text="点击悬浮球开始标定",
            bg="#222222",
            fg="#ffffff",
            font=("Arial", 11),
        )
        self.hint_label.pack(padx=12, pady=5)
        self._set_hint(self._idle_hint_text())

        # ---- 悬浮球 ----
        self.ball = tk.Toplevel(self.root)
        self.ball.overrideredirect(True)
        self.ball.attributes("-topmost", True)
        self.ball.geometry("92x92+220+220")
        self.ball.configure(bg="#2d2d2d")
        self.canvas = tk.Canvas(
            self.ball, width=92, height=92, bg="#2d2d2d", highlightthickness=0
        )
        self.canvas.create_oval(
            4, 4, 88, 88, fill="#3a7afe", outline="#ffffff", width=2
        )
        self.text_id = self.canvas.create_text(
            46, 46, text="开始", fill="white", font=("Arial", 13, "bold"),
            justify="center",
        )
        self.canvas.pack()
        # 拖动 + 点击判定
        self.ball.bind("<ButtonPress-1>", self._on_press)
        self.ball.bind("<B1-Motion>", self._on_drag)
        self.ball.bind("<ButtonRelease-1>", self._on_release)
        # 右键：空闲时切换模式；标定时取消
        self.ball.bind("<Button-3>", self._on_right_click)
        self.ball.bind("<Button-2>", self._on_right_click)
        self.ball.bind("<Escape>", lambda e: self.cancel())

        # ---- 全屏捕获层（透明，用于记录屏幕点击）----
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.05)  # 极淡，便于看到下方内容
        self.overlay.configure(bg="#1e90ff")
        self.overlay.geometry(f"{sw}x{sh}+0+0")
        self.overlay.bind("<Button-1>", self._on_capture_click)
        # 标定时右键取消（Button-2 兼容 macOS 触控板双指点按）
        self.overlay.bind("<Button-3>", lambda e: self.cancel())
        self.overlay.bind("<Button-2>", lambda e: self.cancel())
        self.overlay.withdraw()

        self.root.mainloop()

    # ------------------------------------------------------------------ #
    # 布局辅助
    # ------------------------------------------------------------------ #
    def _place_hint(self):
        self.hint.update_idletasks()
        w = self.hint.winfo_width()
        h = self.hint.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        self.hint.geometry(f"{w}x{h}+{x}+8")

    def _set_hint(self, text: str):
        self.hint_label.config(text=text)
        self._place_hint()

    def _set_ball(self, text: str):
        self.canvas.itemconfig(self.text_id, text=text)

    # ------------------------------------------------------------------ #
    # 模式（多项式 / 折线段）
    # ------------------------------------------------------------------ #
    MODE_NAMES = {"poly": "多项式模式", "piecewise": "折线段模式"}

    def _idle_hint_text(self):
        other = "折线段模式" if self.mode == "poly" else "多项式模式"
        return f"点击悬浮球开始（当前: {self.MODE_NAMES[self.mode]}，右键切换为{other}）"

    def _toggle_mode(self):
        self.mode = "piecewise" if self.mode == "poly" else "poly"
        self._set_hint(self._idle_hint_text())

    def _on_right_click(self, event=None):
        # 空闲时右键切换模式；标定时右键取消
        if self.state == "IDLE":
            self._toggle_mode()
        else:
            self.cancel()

    def _show_overlay(self):
        self.overlay.deiconify()
        # 明确把悬浮球与提示条放在捕获层之上，确保「结束」能被点到
        self.ball.lift(self.overlay)
        self.hint.lift(self.overlay)

    def _hide_overlay(self):
        self.overlay.withdraw()

    # ------------------------------------------------------------------ #
    # 悬浮球拖动 / 点击
    # ------------------------------------------------------------------ #
    def _on_press(self, event):
        self._sx, self._sy = event.x_root, event.y_root
        self._ox, self._oy = self.ball.winfo_x(), self.ball.winfo_y()
        self._moved = False

    def _on_drag(self, event):
        dx = event.x_root - self._sx
        dy = event.y_root - self._sy
        if abs(dx) > 3 or abs(dy) > 3:
            self._moved = True
        self.ball.geometry(f"+{self._ox + dx}+{self._oy + dy}")
        # 结果横条跟随悬浮球移动
        if self.result_bar is not None:
            self._position_result_bar()

    def _on_release(self, event):
        if not self._moved:
            self._on_click()

    def _on_click(self):
        if self.state == "IDLE":
            self._start_calibration()
        elif self.state in ("GET_ORIGIN", "GET_X", "GET_Y"):
            self.cancel()  # 标定中途点球 = 取消
        elif self.state == "COLLECT":
            self._finish()

    # ------------------------------------------------------------------ #
    # 状态机
    # ------------------------------------------------------------------ #
    def _start_calibration(self):
        self._close_result_bar()
        self.state = "GET_ORIGIN"
        self._show_overlay()
        self._set_ball("取消")
        self._set_hint("点击【原点 (0,0)】")

    def _on_capture_click(self, event):
        x, y = event.x_root, event.y_root
        # 安全网：若点击落在悬浮球区域内，按「点击悬浮球」处理，
        # 这样即使捕获层意外盖在球上方，结束/取消也能正常工作。
        bx, by = self.ball.winfo_x(), self.ball.winfo_y()
        bw, bh = self.ball.winfo_width(), self.ball.winfo_height()
        if bx <= x <= bx + bw and by <= y <= by + bh:
            self._on_click()
            return
        if self.state == "GET_ORIGIN":
            self.O = np.array([x, y], dtype=float)
            self.state = "GET_X"
            self._set_ball("取消")
            self._set_hint("已记录原点。点击【X 轴单位点 (1,0)】")
        elif self.state == "GET_X":
            self.Px = np.array([x, y], dtype=float)
            self.ex = self.Px - self.O
            self.state = "GET_Y"
            self._set_hint("已记录 X 轴。点击【Y 轴单位点 (0,1)】")
        elif self.state == "GET_Y":
            self.Py = np.array([x, y], dtype=float)
            self.ey = self.Py - self.O
            self.state = "COLLECT"
            self.points_screen = []
            self._set_ball("结束")
            self._set_hint("点击数据点；全部完成后点悬浮球「结束」")
        elif self.state == "COLLECT":
            self.points_screen.append((float(x), float(y)))
            self._set_hint(
                f"已采集 {len(self.points_screen)} 个点；完成后点悬浮球「结束」"
            )

    def cancel(self):
        self.state = "IDLE"
        self._hide_overlay()
        self._set_ball("开始")
        self._set_hint(self._idle_hint_text())

    # ------------------------------------------------------------------ #
    # 结束并拟合
    # ------------------------------------------------------------------ #
    def _finish(self):
        self._hide_overlay()
        self.state = "IDLE"
        self._set_ball("开始")

        if len(self.points_screen) < 1:
            messagebox.showerror("提示", "尚未采集任何数据点。")
            self._set_hint(self._idle_hint_text())
            return

        try:
            local = [
                fm.screen_to_local(p, self.O, self.ex, self.ey)
                for p in self.points_screen
            ]
            if self.mode == "piecewise":
                expr_body = fm.build_piecewise_expr(local)
                summary = "折线段模式（连续分段线性）"
                extra = {}
            else:
                coeffs = fm.fit_polynomial(local)
                expr_body = fm.poly_to_str(coeffs)
                summary = f"多项式模式（{len(coeffs) - 1} 次，n-1）"
                extra = {
                    "coeffs_ascending": [float(c) for c in coeffs],
                    "degree": len(coeffs) - 1,
                }
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("计算错误", str(exc))
            self._set_hint(self._idle_hint_text())
            return

        expr = "y = " + expr_body
        self._present_result(local, expr, summary, extra)

    def _present_result(self, local, expr, summary, extra):
        n = len(local)

        # ---- 保存数据（JSON，非绘图）----
        data = {
            "mode": self.mode,
            "summary": summary,
            "origin_screen": self.O.tolist(),
            "x_unit_screen": self.Px.tolist(),
            "y_unit_screen": self.Py.tolist(),
            "points_screen": [list(p) for p in self.points_screen],
            "points_local": [list(map(float, p)) for p in local],
            "expr": expr,
        }
        data.update(extra)
        json_path = os.path.join(PROJECT_DIR, "fit_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # ---- 控制台输出 ----
        print("=" * 60)
        print(f"采集点数 n = {n}，{summary}")
        print("函数（局部坐标系）:")
        print("  " + expr)
        if self.mode == "poly":
            print("系数（升幂 a0 + a1 x + ...）:")
            print("  " + ", ".join(f"{c:.6g}" for c in extra["coeffs_ascending"]))
        print("采集点（局部坐标）:")
        for p in local:
            print(f"  ({p[0]:.4f}, {p[1]:.4f})")
        print(f"数据已保存: {json_path}")
        print("=" * 60)

        # ---- 从悬浮球右侧伸出横条显示函数字符串 ----
        self._show_result_bar(expr)
        self._set_hint("已生成函数，横条显示在球右侧（点击复制 · 右键关闭）")

    # ------------------------------------------------------------------ #
    # 结果横条（从悬浮球右侧伸出）
    # ------------------------------------------------------------------ #
    def _show_result_bar(self, expr: str):
        self._close_result_bar()
        multiline = fm.multiline_expr(expr)
        bar = tk.Toplevel(self.root)
        bar.overrideredirect(True)
        bar.attributes("-topmost", True)
        bar.configure(bg="#111418")
        frame = tk.Frame(bar, bg="#111418")
        frame.pack(padx=2, pady=4)
        tk.Label(
            frame, text=multiline, bg="#111418", fg="#7fd1ff",
            font=("Menlo", 12), justify="left", anchor="w",
            wraplength=360,  # 单项过长时再自动折行
        ).pack(side="top", padx=(8, 6), anchor="w")
        tk.Label(
            frame, text="点击复制 · 右键关闭", bg="#111418", fg="#6b7785",
            font=("Arial", 9), anchor="w",
        ).pack(side="top", padx=(8, 6), pady=(2, 0), anchor="w")
        bar.bind("<Button-1>", lambda e: self._copy_expr(expr))
        bar.bind("<Button-3>", lambda e: self._close_result_bar())
        self.result_bar = bar
        self._position_result_bar()

    def _position_result_bar(self):
        if self.result_bar is None:
            return
        self.result_bar.update_idletasks()
        bw = self.ball.winfo_width()
        bx = self.ball.winfo_x()
        by = self.ball.winfo_y()
        w = self.result_bar.winfo_width()
        h = self.result_bar.winfo_height()
        x = bx + bw + 8
        y = by
        # 右侧空间不足时改放到球的左侧
        sw = self.root.winfo_screenwidth()
        if x + w > sw - 4:
            x = bx - w - 8
        self.result_bar.geometry(f"{w}x{h}+{x}+{y}")

    def _copy_expr(self, expr: str):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(expr)
            self._set_hint("函数已复制到剪贴板")
        except Exception:  # noqa: BLE001
            self._set_hint("复制失败，请手动记录上方函数")

    def _close_result_bar(self):
        if self.result_bar is not None:
            try:
                self.result_bar.destroy()
            except Exception:  # noqa: BLE001
                pass
            self.result_bar = None


if __name__ == "__main__":
    FuncHelper()
