# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:01:17 2023

@author: Dong Yang
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# 定义电子在电场中的运动方程，假设电场只沿 x 轴方向存在
def get_acceleration(x, y):
    Ex = 1/15*1e10 * x# 电场强度的 x 分量
    # Ex = -20503.9672124621 + 1638283217.5806966*x - 707863737472.4362*x*x
    Ey = 0  # 电场强度的 y 分量
    ax = q * Ex / m_e  # 计算电子的加速度，不考虑相对论
    ay = Ey
    return ax, ay

# 模拟电子的运动轨迹
def simulate_trajectory(x0, y0, v0, t):
    dt = 1e-12  # 时间步长
    n_steps = int(t / dt)
    x = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    vx = np.zeros(n_steps + 1)
    vy = np.zeros(n_steps + 1)
    x[0], y[0], vx[0], vy[0] = x0, y0, v0[0], v0[1]
    for i in range(n_steps):
        ax, ay = get_acceleration(x[i], y[i])
        vx[i + 1] = vx[i] + ax * dt  # 使用欧拉法计算速度和位置
        vy[i + 1] = vy[i] + ay * dt
        x[i + 1] = x[i] + vx[i + 1] * dt
        y[i + 1] = y[i] + vy[i + 1] * dt
    return x, y, vx

# 重复模拟多次，获得电子的稳定分布
def simulate_distribution(x0, y0, v0, t, n_simulations):
    xs, ys, vs = [], [], []
    for _ in range(n_simulations):
        x, y, vx = simulate_trajectory(x0, y0, v0, t)
        xs.append(x[-1])  # 记录电子停留的位置
        ys.append(y[-1])
        vs.append(vx[-1])
    return xs, ys, vs

# 绘制电子的稳定分布
def plot_distribution(px, py, pv, t, t_max):
    # plt.hist2d(px[:,0], py[:,0], bins=50, cmap=plt.cm.jet)
    # plt.hist2d(px[:,2], py[:,2], bins=50, cmap=plt.cm.jet)
    num = int(t_max / t)
    for i in range(num):
        # plt.plot(px[1,i]*1e3, (t+i*t)*1e9, '.')
        plt.plot(pv[1,i]/(3*1e8), (t+i*t)*1e9, '.')
    # plt.xlim(0, 30e-6) 
    # plt.ylim(-1e-2, 1e-2)
    plt.xlabel('x (mm)')
    plt.ylabel('time (ns)')
    plt.show()

# 
def run(x0, y0, v0, t, t_max, n_simulations):
    num = int(t_max / t)
    px = np.zeros((n_simulations,num))
    py = np.zeros((n_simulations,num))
    pv = np.zeros((n_simulations,num))
    for i in range(num):
        px[:,i], py[:,i], pv[:,i] = simulate_distribution(x0, y0, v0, t, n_simulations)
        t = t + t_max/num
    return px, py, pv

# 设置参数并运行模拟
q = 1.6e-19 # 电荷量
m_e = 9.1e-31 # 电子静止质量
E_0 = 1*q # 初始能量，换算成焦耳单位

x0, y0 = 1e-15, 1e-15  # 初始位置
ve = np.sqrt(2*E_0/m_e)
alph = np.random.uniform(0,np.pi/2)
theta = np.random.uniform(0,np.pi/2)
ve_111 = ve* math.cos(alph)* math.sin(theta)

v0 = np.array([ve_111,0])
t = 1e-11  # 模拟时间
t_max = 7.5e-10
n_simulations = 100  # 模拟次数
xs, ys = simulate_distribution(x0, y0, v0, t, n_simulations)
px, py, pv = run(x0, y0, v0, t, t_max, n_simulations)
# 绘制电子的稳定分布
plot_distribution(px, py, pv, t, t_max)