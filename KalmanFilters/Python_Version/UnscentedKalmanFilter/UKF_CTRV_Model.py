import numpy as np
import matplotlib.pyplot as plt
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

def state_angle_control(state):
    if state[3][0] > np.pi:
        state[3][0] = state[3][0]-2*np.pi
    elif state[3][0] < -np.pi:
        state[3][0] = state[3][0]+2*np.pi
    else:
        state[3][0] = state[3][0]

    return state


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


def F_function_for_Sigma_Points(Sigma_Points, delta_t):
    """
    Sigma_Points:5x15
    """
    for i in range(0, 11):
        if i == 0:
            state_Sigma_predict = transition_function(
                Sigma_Points[0:5, i:i+1], delta_t)
        else:
            state_Sigma_predict = np.hstack((state_Sigma_predict, transition_function(
                Sigma_Points[0:5, i:i+1], delta_t)))
    return state_Sigma_predict


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

# 激光雷达测量矩阵
R_lidar = np.array([[0.0225, 0.0],
                    [0.0, 0.0225]])

R_radar = np.array([[0.09, 0.0, 0.0],
                    [0.0, 0.0009, 0.0],
                    [0.0, 0.0, 0.09]])

std_noise_a = 2.0
std_noise_yaw_d = 0.3


P = np.eye(5)   # 初始化协方差矩阵
state = np.zeros([5, 1])  # 具体状态初始化x,y,v,θ,w,u_a,u_dw
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

ground_truth_value_x = []
ground_truth_value_y = []

measurement_value_x = []
measurement_value_y = []

kalman_value_x = []
kalman_value_y = []

# UKF算法
L = 5  # 状态维数

alpha = 1e-2
ki = 0
beta = 2
lambd = alpha**2*(L+ki)-L

# 计算去权重
for i in range(0, 2*L+1):
    if i == 0:
        Wm = lambd/(L+lambd)
        Wc = Wm+(1-alpha**2+beta)
    else:
        Wm = np.hstack((Wm, 1/(2*(L+lambd))))  # Wm 均值权重
        Wc = np.hstack((Wc, 1/(2*(L+lambd))))  # Wc 方差权重
Wm = Wm.reshape([1, 2*L+1])
Wc = Wc.reshape([1, 2*L+1])

state_estimate = state

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

    # %% 第一步：生成一组Sigma点集
    cho = np.linalg.cholesky(P*(L+lambd))
    for j in range(0, L):
        if j == 0:
            state_gamaP1 = state_estimate+cho[:, j:j+1]
            state_gamaP2 = state_estimate-cho[:, j:j+1]
            state_gamaP1 = state_angle_control(state_gamaP1)
            state_gamaP2 = state_angle_control(state_gamaP2)
        else:
            temp_P1 = state_estimate+cho[:, j:j+1]
            temp_P2 = state_estimate-cho[:, j:j+1]
            temp_P1 = state_angle_control(temp_P1)
            temp_P2 = state_angle_control(temp_P2)
            state_gamaP1 = np.hstack((state_gamaP1, temp_P1))
            state_gamaP2 = np.hstack((state_gamaP2, temp_P2))
    state_Sigma = np.hstack(
        (state_estimate, state_gamaP1, state_gamaP2))  # Sigma点集

    # %% 第二步：对Sigma点集进行一步预测
    state_Sigma_predict = F_function_for_Sigma_Points(state_Sigma, dt)

    # %% 第三步：计算第二步的均值和协方差
    state_predict_estimate = np.zeros([5, 1])
    for j in range(0, 2*L+1):
        state_predict_estimate = state_predict_estimate + \
            Wm[0][j]*state_Sigma_predict[:, j:j+1]

    state_predict_estimate = state_angle_control(state_predict_estimate)

    P_predict = np.zeros([5, 5])  # 协方差矩阵预测
    for j in range(0, 2*L+1):
        error = state_Sigma_predict[:, j:j+1]-state_predict_estimate
        error = state_angle_control(error)
        P_predict = P_predict+Wc[0][j]*np.dot(error, error.T)
    theta = state_predict_estimate[3][0]
    Q = Q_matrix(theta, dt)
    P_predict = P_predict + Q

    # %% 第四步：根据预测值，再使用UT变换，得到新的Sigma点集
    cho = np.linalg.cholesky(P_predict*(L+lambd))
    for j in range(0, L):
        if j == 0:
            state_Sigma_P1 = state_predict_estimate+cho[:, j:j+1]
            state_Sigma_P2 = state_predict_estimate-cho[:, j:j+1]
            state_Sigma_P1 = state_angle_control(state_Sigma_P1)
            state_Sigma_P2 = state_angle_control(state_Sigma_P2)
        else:
            temp_P1 = state_predict_estimate+cho[:, j:j+1]
            temp_P2 = state_predict_estimate-cho[:, j:j+1]
            temp_P1 = state_angle_control(temp_P1)
            temp_P2 = state_angle_control(temp_P2)
            state_Sigma_P1 = np.hstack((state_Sigma_P1, temp_P1))
            state_Sigma_P2 = np.hstack((state_Sigma_P2, temp_P2))
    state_aug_Sigma = np.hstack(
        (state_predict_estimate, state_Sigma_P1, state_Sigma_P2))  # Sigma点集

    if t_measurement[0] == 0.0:
        # Lidar数据

        # %% 第五步：观测预测
        for j in range(0, 2*L+1):
            if j == 0:
                Z_Sigma_predict = state_aug_Sigma[0:2, j:j+1]
            else:
                Z_Sigma_predict = np.hstack(
                    (Z_Sigma_predict, state_aug_Sigma[0:2, j:j+1]))

        # %% 第六步：计算观测预测均值和协方差
        Z_estimate = np.zeros((2, 1))
        for j in range(0, 2*L+1):
            Z_estimate = Z_estimate+Wm[0][j]*Z_Sigma_predict[:, j:j+1]

        # 计算协方差Pzz
        Pzz = np.zeros([2, 2])
        for j in range(0, 2*L+1):
            error = Z_Sigma_predict[:, j:j+1] - Z_estimate
            Pzz = Pzz + Wc[0][j]*np.dot(error, error.T)
        Pzz = Pzz + R_lidar

        # 计算协方差Pxz
        Pxz = np.zeros([L, 2])
        for j in range(0, 2*L+1):
            error_state = state_aug_Sigma[:, j:j+1] - state_predict_estimate
            error_state = state_angle_control(error_state)
            error_z = Z_Sigma_predict[:, j:j+1] - Z_estimate
            Pxz = Pxz + Wc[0][j]*np.dot(error_state, error_z.T)

    else:
        # Radar数据
        # %% 第五步：观测预测
        for j in range(0, 2*L+1):
            if j == 0:
                Z_Sigma_predict = Radar_map_function(state_aug_Sigma[:, j:j+1])
            else:
                Z_Sigma_predict = np.hstack(
                    (Z_Sigma_predict, Radar_map_function(state_aug_Sigma[:, j:j+1])))

        # %% 第六步：计算观测预测均值和协方差
        Z_estimate = np.zeros((3, 1))
        for j in range(0, 2*L+1):
            Z_estimate = Z_estimate+Wm[0][j]*Z_Sigma_predict[:, j:j+1]

        # 计算协方差Pzz
        Pzz = np.zeros([3, 3])
        for j in range(0, 2*L+1):
            error = Z_Sigma_predict[:, j:j+1] - Z_estimate
            Pzz = Pzz + Wc[0][j]*np.dot(error, error.T)
        Pzz = Pzz + R_radar

        # 计算协方差Pxz
        Pxz = np.zeros([L, 3])
        for j in range(0, 2*L+1):
            error_state = state_aug_Sigma[:, j:j+1] - state_predict_estimate
            error_state = state_angle_control(error_state)
            error_z = Z_Sigma_predict[:, j:j+1] - Z_estimate
            Pxz = Pxz + Wc[0][j]*np.dot(error_state, error_z.T)

    # %%第七步：计算卡尔曼增益
    K = np.dot(Pxz, np.linalg.inv(Pzz))

    # %%第八步：状态和方差更新
    state_estimate = state_predict_estimate + \
        np.dot(K, z-Z_estimate)
    state_estimate = state_angle_control(state_estimate)
    P = P_predict-np.dot(np.dot(K, Pzz), K.T)

    kalman_value_x.append(state_estimate[0][0])
    kalman_value_y.append(state_estimate[1][0])


plt.figure()
plt.plot(ground_truth_value_x, ground_truth_value_y,
         color="r", label="ground truth")
plt.plot(measurement_value_x, measurement_value_y,
         color="g", label="measurement")
plt.plot(kalman_value_x, kalman_value_y, color="b", label="kalman value")
plt.legend()
plt.show()
