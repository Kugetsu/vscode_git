import tkinter as tk
from tkinter import ttk

import numpy as np

class AngleCalculatorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Angle Calculator")

        self.theta_ab = tk.DoubleVar(value=120)
        self.theta_bc = tk.DoubleVar(value=135)

        self.progress_label = ttk.Label(self.root, text="第三个夹角的大小范围：")
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack()

        self.slider_ab = ttk.Scale(self.root, from_=0, to=180, length=300, variable=self.theta_ab, command=self.update_angle)
        self.slider_ab.pack()

        self.slider_bc = ttk.Scale(self.root, from_=0, to=180, length=300, variable=self.theta_bc, command=self.update_angle)
        self.slider_bc.pack()

        # 创建标签和数值显示
        self.label_ab = ttk.Label(self.root, text="角度AB:")
        self.label_ab.pack()

        self.label_theta_ab = ttk.Label(self.root, textvariable=self.theta_ab)
        self.label_theta_ab.pack()

        self.label_bc = ttk.Label(self.root, text="角度BC:")
        self.label_bc.pack()

        self.label_theta_bc = ttk.Label(self.root, textvariable=self.theta_bc)
        self.label_theta_bc.pack()

    def update_angle(self, event=None):
        theta_ab = self.theta_ab.get()
        theta_bc = self.theta_bc.get()

        angle_range = self.compute_angle_range(theta_ab, theta_bc)
        if angle_range is None:
            self.progress_label.config(text="无法确定第三个夹角的范围")
        else:
            self.progress_label.config(text="第三个夹角的大小范围：[{}°, {}°]".format(angle_range[0], angle_range[1]))

        # 更新进度条的值
        self.progress_bar["value"] = theta_ab

    def compute_angle_range(self, theta_ab, theta_bc):
        # 将角度转换为弧度
        theta_ab_rad = np.radians(theta_ab)
        theta_bc_rad = np.radians(theta_bc)

        # 计算夹角的余弦值
        cos_theta_ab = np.cos(theta_ab_rad)
        cos_theta_bc = np.cos(theta_bc_rad)

        # 检查夹角是否合法
        if abs(cos_theta_ab) > 1 or abs(cos_theta_bc) > 1:
            return None  # 无法确定夹角范围

        # 计算第三个夹角的范围
        a_rad = np.arccos(cos_theta_ab * cos_theta_bc - np.sin(theta_ab_rad) * np.sin(theta_bc_rad))
        b_rad = np.arccos(cos_theta_ab * cos_theta_bc + np.sin(theta_ab_rad) * np.sin(theta_bc_rad))

        # 将弧度转换为角度
        a_deg = np.degrees(a_rad)
        b_deg = np.degrees(b_rad)

        return [a_deg, b_deg]

    def run(self):
        self.root.mainloop()

# 创建应用对象并运行
app = AngleCalculatorApp()
app.run()
