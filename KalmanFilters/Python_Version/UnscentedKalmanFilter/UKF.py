import numpy as np
import matplotlib.pyplot as plt


def F_function(state, dt):
    """
    状态方程
    state：[x,y,v_x,v_y,a_x,a_y]
    dt:采样间隔
    """
    F = np.array([[1, 0, dt, 0, 1/2*dt*dt, 0],
                  [0, 1, 0, dt, 0, 1/2*dt*dt],
                  [0, 0, 1, 0, dt, 0],
                  [0, 0, 0, 1, 0, dt],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    state = np.dot(F, state)
    return state


def H_function(state):
    """
    观测方程
    """
    x = state[0][0]
    y = state[1][0]

    rho = np.sqrt(x*x+y*y)
    fai = np.arctan2(y, x)

    return np.array([[rho], [fai]])


def generate_truth_state_data(initial_state, steps, Q, dt):
    """
    生成真实状态数据
    initial_state:初始状态
    steps:总步数
    Q:过程噪声协方差
    dt:采样间隔
    """
    for i in range(0, steps):
        if i == 0:
            s = initial_state
            truth_state_set = F_function(
                s, dt) + np.dot(Q**0.5, np.random.normal(0, 1, (6, 1)))
        else:
            s = truth_state_set[:, i-1].reshape([6, 1])
            truth_state_set = np.hstack(
                (truth_state_set, F_function(s, dt)+np.dot(Q**0.5, np.random.normal(0, 1, (6, 1)))))

    return truth_state_set


def generate_measurement_data(truth_state_set, steps, R):
    """
    生成测量数据
    R:测量噪声协方差矩阵
    """
    for i in range(0, steps):
        truth_state = truth_state_set[:, i].reshape([6, 1])
        if i == 0:
            measurement_set = H_function(
                truth_state) + np.dot(R**0.5, np.random.normal(0, 1, (2, 1)))
        else:
            measurement_set = np.hstack((measurement_set, H_function(
                truth_state) + np.dot(R**0.5, np.random.normal(0, 1, (2, 1)))))

    return measurement_set



# 参数
dt = 0.4  # 采样间隔

# 状态矩阵
F_matrix = np.array([[1, 0, dt, 0, 1/2*dt*dt, 0],
                     [0, 1, 0, dt, 0, 1/2*dt*dt],
                     [0, 0, 1, 0, dt, 0],
                     [0, 0, 0, 1, 0, dt],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])

Q = np.diag([1, 1, 0.1**2, 0.1**2, 0.01**2, 0.01**2])  # 过程噪声协方差矩阵
R = np.diag([100, 0.001**2])  # 测量噪声协方差矩阵
steps = 50  # 总仿真步数
initial_state = np.array(
    [[1000], [5000], [10], [50], [2], [-4]])  # 初始状态，生成真实状态数据时使用

# 生成数据
truth_state_set = generate_truth_state_data(
    initial_state, steps, Q, dt)  # 生成真实状态数据

measurement_set = generate_measurement_data(
    truth_state_set, steps, R)  # 生成测量数据

# UKF算法
L = np.shape(truth_state_set)[0]  # 状态维数
m = np.shape(measurement_set)[0]  # 观测维数

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

state_ukf = truth_state_set[:, 0:1]  # ukf状态值

P = np.diag([100, 100, 1, 1, 0.1, 0.1])

state_estimate = state_ukf[:, 0:1]

for i in range(1, steps):
    # %% 第一步：生成一组Sigma点集
    cho = np.linalg.cholesky(P*(L+lambd))
    for j in range(0, L):
        if j == 0:
            state_gamaP1 = state_estimate+cho[:, j:j+1]
            state_gamaP2 = state_estimate-cho[:, j:j+1]
        else:
            state_gamaP1 = np.hstack(
                (state_gamaP1, state_estimate+cho[:, j:j+1]))
            state_gamaP2 = np.hstack(
                (state_gamaP2, state_estimate-cho[:, j:j+1]))
    state_Sigma = np.hstack(
        (state_estimate, state_gamaP1, state_gamaP2))  # Sigma点集

    # %% 第二步：对Sigma点集进行一步预测
    state_Sigma_predict = np.dot(F_matrix, state_Sigma)

    # %% 第三步：计算第二步的均值和协方差
    state_predict_estimate = np.zeros([L, 1])
    for j in range(0, 2*L+1):
        state_predict_estimate = state_predict_estimate + \
            Wm[0][j]*state_Sigma_predict[:, j:j+1]

    P_predict = np.zeros([L, L])  # 协方差矩阵预测
    for j in range(0, 2*L+1):
        error = state_Sigma_predict[:, j:j+1]-state_predict_estimate
        P_predict = P_predict+Wc[0][j]*np.dot(error, error.T)
    P_predict = P_predict + Q

    # %% 第四步：根据预测值，再使用UT变换，得到新的Sigma点集
    cho = np.linalg.cholesky(P_predict*(L+lambd))
    for j in range(0, L):
        if j == 0:
            state_Sigma_P1 = state_predict_estimate+cho[:, j:j+1]
            state_Sigma_P2 = state_predict_estimate-cho[:, j:j+1]
        else:
            state_Sigma_P1 = np.hstack(
                (state_Sigma_P1, state_predict_estimate+cho[:, j:j+1]))
            state_Sigma_P2 = np.hstack(
                (state_Sigma_P2, state_predict_estimate-cho[:, j:j+1]))
    state_aug_Sigma = np.hstack(
        (state_predict_estimate, state_Sigma_P1, state_Sigma_P2))  # Sigma点集

    # %% 第五步：观测预测
    for j in range(0, 2*L+1):
        if j == 0:
            Z_Sigma_predict = H_function(state_aug_Sigma[:, j:j+1])
        else:
            Z_Sigma_predict = np.hstack(
                (Z_Sigma_predict, H_function(state_aug_Sigma[:, j:j+1])))

    # %% 第六步：计算观测预测均值和协方差
    Z_estimate = np.zeros((m, 1))
    for j in range(0, 2*L+1):
        Z_estimate = Z_estimate+Wm[0][j]*Z_Sigma_predict[:, j:j+1]

    # 计算协方差Pzz
    Pzz = np.zeros([m, m])
    for j in range(0, 2*L+1):
        error = Z_Sigma_predict[:, j:j+1] - Z_estimate
        Pzz = Pzz + Wc[0][j]*np.dot(error, error.T)
    Pzz = Pzz + R

    # 计算协方差Pxz
    Pxz = np.zeros([L, m])
    for j in range(0, 2*L+1):
        error_state = state_aug_Sigma[:, j:j+1] - state_predict_estimate
        error_z = Z_Sigma_predict[:, j:j+1] - Z_estimate
        Pxz = Pxz + Wc[0][j]*np.dot(error_state, error_z.T)

    # %%第七步：计算卡尔曼增益
    K = np.dot(Pxz, np.linalg.inv(Pzz))

    # %%第八步：状态和方差更新
    state_estimate = state_predict_estimate + \
        np.dot(K, measurement_set[:, i:i+1]-Z_estimate)
    P = P_predict-np.dot(np.dot(K, Pzz), K.T)
    state_ukf = np.hstack((state_ukf, state_estimate))

# %% 绘图
plt.figure()
plt.plot(truth_state_set[0, :], truth_state_set[1, :], label="ground truth")
plt.plot(measurement_set[0, :]*np.cos(measurement_set[1, :]),
         measurement_set[0, :]*np.sin(measurement_set[1, :]), label="measurement")
plt.plot(state_ukf[0, :], state_ukf[1, :], label="ukf")
plt.legend()
plt.show()
