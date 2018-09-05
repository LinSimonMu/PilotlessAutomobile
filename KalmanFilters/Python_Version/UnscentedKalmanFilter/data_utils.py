import numpy as np
"""
数据格式
    序号(类别)  测量              真实
LIDAR:  0.0    x,y,    时间,   x,y,v_x,v_y
RADAR:  1.0    ρ,φ,ρ', 时间,   x,y,v_x,v_y
"""

dataset = []

with open('./data_synthetic.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.strip()
        numbers = line.split()
        result = []
        for i, item in enumerate(numbers):
            item.strip()
            if i == 0:
                if item == 'L':
                    result.append(0.0)
                else:
                    result.append(1.0)
            else:
                result.append(float(item))
        dataset.append(result)
    f.close()
