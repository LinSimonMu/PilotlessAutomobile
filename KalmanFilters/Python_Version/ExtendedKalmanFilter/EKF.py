import numpy as np
import matplotlib.pyplot as plt

"""
说明：
----状态量
    -----X:     x方向位移
    -----Vx:    x方向速度
    -----Y:     y方向位移
    -----Vy:    y方向速度

----测量量
    -----r:     位置半径
    -----alpha  位置角度
"""

# 扩展的卡尔曼滤波器
K_vx_damped_coefficient = 0.01  # x方向上的阻尼系数
K_vy_damped_coefficient = 0.05  # y方向上的阻尼系数
g = 9.8    # 重力加速度
t = 15  # 仿真时间
Ts = 0.1  # 采样周期
total_steps = int(15 / Ts)   # 总的仿真步数

d_ax = 3  # x方向上的加速度噪声
d_ay = 3  # y方向上的加速度噪声

d_r = 8   # 测量半径r的噪声
d_alpha = 0.01  # 测量角度alpha的噪声

Rk = np.diag([d_r**2, d_alpha**2])  # 测量量协方差

Qk = np.diag([0, (d_ax/10)**2, 0, (d_ay/10)**2])  # 状态量噪声

real_state = np.array([[0], [50], [500], [0]])  # 状态量矩阵
for step in range(1, total_steps):
    # 前一次的状态量
    X_pre = real_state[:, step-1][0]
    Vx_pre = real_state[:, step-1][1]
    Y_pre = real_state[:, step-1][2]
    Vy_pre = real_state[:, step-1][3]
    # 现状态量，由状态模型生成
    X = X_pre + Vx_pre*Ts
    Vx = Vx_pre + (-K_vx_damped_coefficient*Vx_pre**2 +
                   d_ax*np.random.normal(0, 1, 1)[0])*Ts
    Y = Y_pre + Vy_pre*Ts
    Vy = Vy_pre + (K_vy_damped_coefficient*Vy_pre**2 -
                   g + d_ay*np.random.normal(0, 1, 1)[0])*Ts
    real_state = np.hstack((real_state, np.array([[X], [Vx], [Y], [Vy]])))

# 构造测量数据
measurement = np.zeros((2, total_steps))
for step in range(0, total_steps):
    X = real_state[:, step][0]
    Vx = real_state[:, step][1]
    Y = real_state[:, step][2]
    Vy = real_state[:, step][3]

    r = np.sqrt(X**2+Y**2)+d_r*np.random.normal(0, 1, 1)[0]
    alpha = np.arctan(X/Y) + d_alpha*np.random.normal(0, 1, 1)[0]

    measurement[:, step:step+1] = np.array([[r], [alpha]])

# 卡尔曼滤波器滤波后的结果
kalman_result = np.array([[0], [40], [400], [0]])
P_initial = 10*np.eye(4)
for step in range(1, total_steps):
    X_pre = kalman_result[:, step-1][0]
    Vx_pre = kalman_result[:, step-1][1]
    Y_pre = kalman_result[:, step-1][2]
    Vy_pre = kalman_result[:, step-1][3]

    X = X_pre + Vx_pre*Ts
    Vx = Vx_pre + (-K_vx_damped_coefficient*Vx_pre**2)*Ts
    Y = Y_pre + Vy_pre*Ts
    Vy = Vy_pre + (K_vy_damped_coefficient*Vy_pre**2 - g)*Ts

    state = np.array([[X], [Vx], [Y], [Vy]])
    # 状态矩阵
    F = np.array([[1, Ts, 0, 0],
                  [0, 1-2*K_vx_damped_coefficient*Vx*Ts, 0, 0],
                  [0, 0, 1, Ts],
                  [0, 0, 0, 1+2*K_vy_damped_coefficient*Vy*Ts]])

    # 观测矩阵
    R = np.sqrt(X**2+Y**2)
    C = X**2/Y**2
    H = np.array([[X/R, 0, Y/R, 0],
                  [(1/Y)/(1+C), 0, (-X/Y**2)/(1+C), 0]])

    P = np.dot(np.dot(F, P_initial), F.T) + Qk

    S = np.linalg.inv(np.dot(np.dot(H, P), H.T) + Rk)

    K = np.dot(np.dot(P, H.T), S)

    Zk = measurement[:, step:step+1]
    delta = Zk - np.dot(H, state)
    state = state + np.dot(K, delta)
    P = P - np.dot(np.dot(K, H), P)
    P_initial = P

    kalman_result = np.hstack((kalman_result, state))


plt.figure()
plt.plot(real_state[0, :], real_state[2, :], color="b", label="real state")
plt.scatter(measurement[0, :]*np.sin(measurement[1, :]), measurement[0, :]
            * np.cos(measurement[1, :]), color="r", label="measurement")
plt.plot(kalman_result[0, :], kalman_result[2, :],
         color="g", label="kalman_result")
plt.legend()
plt.show()
