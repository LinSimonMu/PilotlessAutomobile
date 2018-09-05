import numpy as np
import data_utils
import matplotlib.pyplot as plt

# 通过运动模型获取汽车的真实运动数据
t_list, x_list, v_list, a_list = data_utils.generate_data(
    100, 5, 4, 10, 20, 0.1)

# 创建激光雷达的测量数据
standard_deviation = 15

lidar_x_list = data_utils.generate_lidar(x_list, standard_deviation)

lidar_t_list = t_list
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 15))

# 真实距离
ax1.set_title("true distance")
ax1.set_xlabel("time")
ax1.set_ylabel("distance")
ax1.set_xlim([0, 21])
ax1.set_ylim([0, 1000])
ax1.plot(t_list, x_list)

# 真实速度
ax2.set_title("true velocity")
ax2.set_xlabel("time")
ax2.set_ylabel("velocity")
ax2.set_xlim([0, 21])
ax2.set_ylim([0, 50])
ax2.set_xticks(range(21))
ax2.set_yticks(range(0, 50, 5))
ax2.plot(t_list, v_list)

# 真实加速度
ax3.set_title("true acceleration")
ax3.set_xlabel("time")
ax3.set_ylabel("acceleration")
ax3.set_xlim([0, 21])
ax3.set_ylim([0, 5])
ax3.plot(t_list, a_list)

# 激光雷达测量结果
ax4.set_title("Lidar measurment VS truth")
ax4.set_xlabel("time")
ax4.set_ylabel("distance")
ax4.set_xlim([0, 21])
ax4.set_ylim([0, 1000])
ax4.set_xticks(range(21))
ax4.set_yticks(range(0, 1000, 100))
ax4.plot(t_list, x_list, label="truth distance")
ax4.scatter(lidar_t_list, lidar_x_list, label="Lidar distance",
            color="red", marker="o", s=2)
ax4.legend()


# 使用卡尔曼滤波器
initial_distance = 0
initial_velocity = 0

# 状态矩阵初始值
x_initial = np.array([[initial_distance], [initial_velocity]])

# 状态协方差矩阵P_initial
P_initial = np.array([[5, 0], [0, 5]])

# 加速度方差
acceleration_varience = 50

# 激光雷达测量结果方差
lidar_variance = standard_deviation**2

# 观测矩阵
H = np.array([[1, 0]])

# 测量噪音协方差
R = np.array([[lidar_variance]])

# 单位矩阵
I = np.eye(2)


def F_matrix(delta_t):
    """
    状态转移矩阵
    """
    return np.array([[1, delta_t], [0, 1]])


def Q_matrix(delta_t, variance):
    """
    外部噪音协方差矩阵
    """
    t4 = delta_t**4
    t3 = delta_t**3
    t2 = delta_t**2
    return variance*np.array([[1/4*t4, 1/2*t3], [1/2*t3, t2]])


def B_matrix(delta_t):
    """
    控制矩阵
    """
    return np.array([[1/2*delta_t**2], [delta_t]])


x = x_initial  # 状态矩阵
P = P_initial  # 状态协方差矩阵

time_result = []  # 记录卡尔曼滤波器的时间
x_result = []  # 卡尔曼滤波器计算得到的距离值
v_result = []  # 卡尔曼滤波器计算得到的速度值

for i in range(len(lidar_x_list)-1):
    delta_t = lidar_t_list[i+1] - lidar_t_list[i]
    F = F_matrix(delta_t)
    Q = Q_matrix(delta_t, acceleration_varience)

    # 按运动模型计算的状态量及其协方差
    x_prime = np.dot(F, x)
    P_prime = np.dot(np.dot(F, P), F.T)+Q

    # 加入测量的状态量，同时考虑运动模型与测量值，求取最佳估计
    # 测量向量与状态向量的差值
    y = np.array([[lidar_x_list[i+1]]])-np.dot(H, x_prime)

    S = np.dot(np.dot(H, P_prime), H.T)+R
    K = np.dot(np.dot(P_prime, H.T), np.linalg.inv(S)) # 卡尔曼增益

    # 加入测量状态后，得到最加估计的状态量及其协方差
    x = x_prime+np.dot(K, y)
    P = np.dot(I-np.dot(K, H), P_prime)

    x_result.append(x[0][0])
    v_result.append(x[1][0])
    time_result.append(lidar_t_list[i+1])


# 将真实距离、激光雷达测量的距离及卡尔曼滤波器的结果可视化
ax5.set_title("Lidar measurment VS truth VS Kalman (Distance)")
ax5.set_xlabel("time")
ax5.set_ylabel("distance")
ax5.set_xlim([0, 21])
ax5.set_ylim([0, 1000])
ax5.set_xticks(range(21))
ax5.set_yticks(range(0, 1000, 100))
ax5.plot(t_list, x_list, label="truth distance", color="blue", linewidth=1)
ax5.scatter(lidar_t_list, lidar_x_list, label="Lidar distance",
            color="red", marker="o", s=2)
ax5.scatter(time_result, x_result, label="kalman",
            color="green", marker="o", s=2)
ax5.legend()

# 真实速度、卡尔曼滤波器的结果可视化
ax6.set_title("truth VS Kalman (Velocity)")
ax6.set_xlabel("time")
ax6.set_ylabel("velocity")
ax6.set_xlim([0, 21])
ax6.set_ylim([0, 50])
ax6.set_xticks(range(0, 21, 2))
ax6.set_yticks(range(0, 50, 5))
ax6.plot(t_list, v_list, label="truth velocity", color="blue", linewidth=1)
ax6.scatter(time_result, v_result, label="kalman",
            color="red", marker="o", s=2)
ax6.legend()

plt.show()
