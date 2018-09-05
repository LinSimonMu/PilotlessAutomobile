import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from data_utils import *


def angle_control(angle):
    """
    将角度控制在[-pi,pi]
    """
    if angle > np.pi:
        angle = angle-2*np.pi
    elif angle < -np.pi:
        angle = angle+2*np.pi
    else:
        angle = angle

    return angle


def F_matrix(state, delta_t):
    """
    通过雅可比矩阵计算状态矩阵
    """
    v = state[2][0]
    theta = state[3][0]
    w = state[4][0]
    if np.abs(w) >= 0.0001:
        # w != 0时
        J_13 = 1/w*(-np.sin(theta)+np.sin(delta_t*w+theta))
        J_14 = v/w*(-np.cos(theta)+np.cos(delta_t*w+theta))
        J_15 = delta_t*v/w*np.cos(delta_t*w+theta)-v / \
            w**2*(-np.sin(theta)+np.sin(delta_t*w+theta))

        J_23 = 1/w*(np.cos(theta)-np.cos(delta_t*w+theta))
        J_24 = v/w*(-np.sin(theta)+np.sin(delta_t*w+theta))
        J_25 = delta_t*v/w*np.sin(delta_t*w+theta) - \
            v/w**2*(np.cos(theta)-np.cos(delta_t*w+theta))
    elif np.abs(w) < 0.0001:
        # w == 0时
        J_13 = delta_t*np.cos(theta)
        J_14 = -delta_t*v*np.sin(theta)
        J_15 = 0

        J_23 = delta_t*np.sin(theta)
        J_24 = delta_t*v*np.cos(theta)
        J_25 = 0

    return np.array([[1, 0, J_13, J_14, J_15],
                     [0, 1, J_23, J_24, J_25],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, delta_t],
                     [0, 0, 0, 0, 1]])


def transition_function(state_pre, delta_t):
    """
    状态转移函数
    前一次的状态：state_pre：x_pre,y_pre,v,theta_pre,w
    """
    x_pre = state_pre[0][0]
    y_pre = state_pre[1][0]
    v = state_pre[2][0]
    theta_pre = state_pre[3][0]
    w = state_pre[4][0]
    if np.abs(w) >= 0.0001:
        # w != 0时
        x = v/w*np.sin(w*delta_t+theta_pre)-v/w*np.sin(theta_pre)+x_pre
        y = -v/w*np.cos(w*delta_t+theta_pre) + v/w*np.cos(theta_pre)+y_pre

    elif np.abs(w) < 0.0001:
        x = v*np.cos(theta_pre)*delta_t+x_pre
        y = v*np.sin(theta_pre)*delta_t+y_pre

    theta = angle_control(w*delta_t+theta_pre)
    return np.array([[x], [y], [v], [theta], [w]])


def Radar_map_function(state):
    x = state[0][0]
    y = state[1][0]
    v = state[2][0]
    theta = state[3][0]

    rho = np.sqrt(x*x+y*y)
    fai = np.arctan2(y, x)

    rho_d = (v*np.cos(theta)*x + v*np.sin(theta)*y) / rho
    if rho < 0.0001:
        # ρ==0时,ρ' = 0
        rho_d = 0

    return np.array([[rho], [fai], [rho_d]])


def H_matrix_of_Radar(state):
    """
    通过雅可比矩阵计算观测矩阵
    """
    x = state[0][0]
    y = state[1][0]
    v = state[2][0]
    theta = state[3][0]

    R2 = x**2+y**2
    H_11 = x/np.sqrt(R2)
    H_12 = y/np.sqrt(R2)

    H_21 = -y/R2
    H_22 = x/R2

    H_31 = v*np.cos(theta)/np.sqrt(R2)-x * \
        (v*x*np.cos(theta)+v*y*np.sin(theta))/(R2**(3/2))
    H_32 = v*np.sin(theta)/np.sqrt(R2)-y * \
        (v*x*np.cos(theta)+v*y*np.sin(theta))/(R2**(3/2))
    H_33 = (x*np.cos(theta)+y*np.sin(theta))/np.sqrt(R2)
    H_34 = v*(y*np.cos(theta)-x*np.sin(theta)) / np.sqrt(R2)

    return np.array([[H_11, H_12, 0, 0, 0],
                     [H_21, H_22, 0, 0, 0],
                     [H_31, H_32, H_33, H_34, 0]])


def Q_matrix(theta, delta_t):
    """
    处理噪声的协方差矩阵
    """
    std_noise_a = 2.0
    std_noise_yaw_d = 0.3
    #sigma_a = std_noise_a**2
    #sigma_w_d = std_noise_yaw_dd**2
    G_11 = 1/2*delta_t**2*np.cos(theta)
    G_21 = 1/2*delta_t**2*np.sin(theta)
    G_31 = delta_t
    G_42 = 1/2*delta_t**2
    G_52 = delta_t

    G = np.array([[G_11, 0],
                  [G_21, 0],
                  [G_31, 0],
                  [0, G_42],
                  [0, G_52]])

    Q_v = np.array([[std_noise_a**2, 0],
                    [0, std_noise_yaw_d**2]])

    Q = np.dot(np.dot(G, Q_v), G.T)

    return Q


def rmse(estimates, actual):
    result = np.sqrt(np.mean((estimates-actual)**2))
    return result


# 初始化协方差矩阵
P = np.eye(5)
# 激光雷达测量矩阵
H_lidar = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0]])

R_lidar = np.array([[0.0225, 0.0],
                    [0.0, 0.0225]])

R_radar = np.array([[0.09, 0.0, 0.0],
                    [0.0, 0.0009, 0.0],
                    [0.0, 0.0, 0.09]])


# 具体状态初始化
state = np.zeros([5, 1])  # x,y,v,θ,w
init_measurement = dataset[0]
current_time = 0.0
if init_measurement[0] == 0.0:
    print("Initialize with LIDAR measurement!")
    current_time = init_measurement[3]
    state[0][0] = init_measurement[1]
    state[1][0] = init_measurement[2]

if init_measurement[0] == 1.0:
    print("Initialize with RADAR measurement!")
    current_time = init_measurement[4]
    init_rho = init_measurement[1]
    init_angle = angle_control(init_measurement[2])
    state[0][0] = init_rho * np.cos(init_angle)
    state[1][0] = init_rho * np.sin(init_angle)


measurement_steps = len(dataset)
dt = 0.05
I = np.eye(5)

ground_truth_value_x = []
ground_truth_value_y = []

measurement_value_x = []
measurement_value_y = []

kalman_value_x = []
kalman_value_y = []

for step in range(1, measurement_steps):
    t_measurement = dataset[step]
    # LIDAR数据时
    if t_measurement[0] == 0.0:
        m_x = t_measurement[1]
        m_y = t_measurement[2]
        z = np.array([[m_x], [m_y]])

        dt = (t_measurement[3]-current_time)/1000000.0
        current_time = t_measurement[3]

        true_x = t_measurement[4]
        true_y = t_measurement[5]
        true_v_x = t_measurement[6]
        true_v_y = t_measurement[7]

    else:
        m_rho = t_measurement[1]
        m_angle = t_measurement[2]
        m_dot_rho = t_measurement[3]
        m_x = m_rho * np.cos(m_angle)
        m_y = m_rho * np.sin(m_angle)
        z = np.array([[m_rho], [m_angle], [m_dot_rho]])

        dt = (t_measurement[4]-current_time)/1000000.0
        current_time = t_measurement[4]

        true_x = t_measurement[5]
        true_y = t_measurement[6]
        true_v_x = t_measurement[7]
        true_v_y = t_measurement[8]

    ground_truth_value_x.append(true_x)
    ground_truth_value_y.append(true_y)

    measurement_value_x.append(m_x)
    measurement_value_y.append(m_y)

    state = transition_function(state, dt)
    F = F_matrix(state, dt)

    theta = state[3][0]
    Q = Q_matrix(theta, dt)

    P = np.dot(np.dot(F, P), F.T)+Q

    if t_measurement[0] == 0.0:
        # Lidar数据
        S = np.dot(np.dot(H_lidar, P), H_lidar.T)+R_lidar
        K = np.dot(np.dot(P, H_lidar.T), np.linalg.inv(S))  # 卡尔曼增益
        y = z - np.dot(H_lidar, state)
        state = state+np.dot(K, y)
        state[3][0] = angle_control(state[3][0])
        P = np.dot((I-np.dot(K, H_lidar)), P)

    else:
        H_radar = H_matrix_of_Radar(state)
        S = np.dot(np.dot(H_radar, P), H_radar.T)+R_radar
        K = np.dot(np.dot(P, H_radar.T), np.linalg.inv(S))  # 卡尔曼增益
        map = Radar_map_function(state)
        y = z - map
        y[1][0] = angle_control(y[1][0])
        state = state+np.dot(K, y)
        state[3][0] = angle_control(state[3][0])
        P = np.dot((I-np.dot(K, H_radar)), P)

    kalman_value_x.append(state[0][0])
    kalman_value_y.append(state[1][0])


plt.figure()
plt.plot(ground_truth_value_x, ground_truth_value_y,
         color="r", label="ground truth")
plt.plot(measurement_value_x, measurement_value_y,
         color="g", label="measurement")
plt.plot(kalman_value_x, kalman_value_y, color="b", label="kalman value")
plt.legend()
plt.show()
