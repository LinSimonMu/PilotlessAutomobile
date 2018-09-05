import numpy as np


def generate_data(x0, v0, a, t1, t2, dt):
    """
    x0:初始位移             ------m
    v0:初始速度             ------m/s
    a:加速度                ------m/s^2
    t1:加速行驶时间         -------s
    t2:匀速行驶时间         -------s
    dt:间隔时间             -------s
    """

    a_current = a
    v_current = v0
    t_current = 0

    # 记录汽车的真实状态
    a_list = []
    v_list = []
    t_list = []

    # 汽车运行的两个阶段
    # 第一阶段加速行驶
    while t_current <= t1:
        # 记录汽车的真实运动动状态
        a_list.append(a_current)
        v_list.append(v_current)
        t_list.append(t_current)
        # 汽车的运动模型
        v_current += a*dt
        t_current += dt

    # 第二阶段，匀速行驶
    a_current = 0
    while t_current >= t1 and t_current < t2:
        a_list.append(a_current)
        v_list.append(v_current)
        t_list.append(t_current)

        t_current += dt

    # 计算汽车行驶的真实距离
    x = x0
    x_list = [x0]
    for i in range(len(t_list)-1):
        tdelta = t_list[i+1]-t_list[i]
        x = x + v_list[i]*tdelta + 0.5 * a_list[i]*tdelta**2
        x_list.append(x)

    return t_list, x_list, v_list, a_list


def generate_lidar(x_list, standard_deviation):
    """
    生成雷达获得的数据，考虑误差，呈现高斯分布
    """
    return x_list + np.random.normal(0, standard_deviation, len(x_list))

