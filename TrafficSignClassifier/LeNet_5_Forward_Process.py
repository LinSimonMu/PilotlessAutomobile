# -*- coding:utf-8 -*-

# 本模块LeNet-5前向传播过程

# LeNet-5 模型架构（修改了最后一层节点数目为43，原来为10）
# Input     -->        layer1:conv1       -->         layer2:pool1           -->           layer3:conv2
# mx32x32x3   +   f:5x5x3x6(p=0,s=1)--> 28x28x6   +   f:2x2x6(p=0,s=2)--> 14x14x6    +  f:5x5x6x16(p=0,s=1)-->
# -->           layer4:pool2        -->         layer5:FullConnectLayer1(special:conv3) -->   layer6:FullConnectLayer2
# 10x10x16   +   f:2x2x16(p=0,s=2)  -->  5x5x16  +    f:5x5x16x120(p=0,s=1)  --> 1x1x120  +  nodes:84  --> 84   +
# --> layer7:FullConnectLayer3
# -->  nodes:43


import tensorflow as tf
import Data_Import

# 输入图像参数
IMAGE_CHANNELS = Data_Import.image_channels
IMAGE_LABELS = Data_Import.image_classes

OUTPUT_NODE = 43

# 第一层卷积层的尺寸和深度
CONV1_SIZE = 5
CONV1_DEEP = 6

# 第二层卷积层的尺寸和深度
CONV2_SIZE = 5
CONV2_DEEP = 16

# 第一全连接层(特殊的第三卷积层尺寸和深度)
CONV3_SIZE = 5
CONV3_DEEP = 120

# 第二全连接层节点数
FC2_NODES = 84

# 第三全连接层节点数
FC3_NODES = 43


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights


# 定义神经网络的前向传播过程
# 训练中加入dropout正则化，只在训练中使用，测试中不使用dropout
def inference(x_input, train, regularizer):
    # 第一层卷积层1
    with tf.variable_scope("layer1-conv1", reuse=tf.AUTO_REUSE):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(x_input, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层池化层1
    with tf.variable_scope("layer2-pool1", reuse=tf.AUTO_REUSE):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 第三层卷积层2
    with tf.variable_scope("layer3-conv2", reuse=tf.AUTO_REUSE):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层池化层2
    with tf.variable_scope("layer4-pool2", reuse=tf.AUTO_REUSE):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 第五层（全连接层）特殊卷积层3
    with tf.variable_scope("layer5-special-conv3", reuse=tf.AUTO_REUSE):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))

        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding="VALID")
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    # 矩阵向量化(拉直)，准备与全连接层相接
    fc1_shape = relu3.get_shape().as_list()
    # print("pool_shape[0]=", pool_shape[0])
    FC1_NODES = fc1_shape[1] * fc1_shape[2] * fc1_shape[3]
    # fc1_shape[0] 为batch数目
    fc1_output = tf.reshape(relu3, [fc1_shape[0], FC1_NODES])

    # 第六层全连接层2
    with tf.variable_scope("layer6-fc2", reuse=tf.AUTO_REUSE):
        fc2_weights = tf.get_variable("weight", [FC1_NODES, FC2_NODES],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("bias", [FC2_NODES], initializer=tf.constant_initializer(0.1))
        # 全连接层l2正则化
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))

        fc2_output = tf.nn.relu(tf.matmul(fc1_output, fc2_weights) + fc2_biases)

        # 在训练时加入dropout正则化,测试时无需dropout正则化
        if train:
            fc2_output = tf.nn.dropout(fc2_output, 0.5)

    # 第七层全连接层3
    with tf.variable_scope("layer7-fc3", reuse=tf.AUTO_REUSE):
        fc3_weights = tf.get_variable("weight", [FC2_NODES, FC3_NODES],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc3_biases = tf.get_variable("bias", [FC3_NODES], initializer=tf.constant_initializer(0.1))

        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc3_weights))

        logit = tf.matmul(fc2_output, fc3_weights) + fc3_biases

    return logit
