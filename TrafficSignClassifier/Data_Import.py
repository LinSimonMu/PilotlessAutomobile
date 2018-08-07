# -*- coding:utf-8 -*-
"""导入相关数据并将其转换为TFRecord格式数据并进行随机batch处理"""

import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

training_file = "./Data/train.p"
testing_file = "./Data/test.p"

with open(training_file, mode="rb") as f:
    train = pickle.load(f)
with open(testing_file, mode="rb") as f:
    test = pickle.load(f)

x_train, y_train = train["features"], train["labels"]
x_test, y_test = test["features"], test["labels"]

image_train_nums = x_train.shape[0]  # 训练总样本数
image_height = x_train.shape[1]  # 图像高度
image_width = x_train.shape[2]  # 图像宽度
image_channels = x_train.shape[3]  # 图像通道数
image_classes = len(set(y_train))  # 图像总类别数

image_test_nums = x_test.shape[0]  # 测试总样本数

"""解码前的图片存为字符串，图像所对应的类别存为整数列表"""


# 生成整数属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


"""将训练数据保存为TFRecorder"""
train_images = x_train
train_images_labels = y_train
train_images_height = image_height
train_images_width = image_width
train_images_channels = image_channels
train_images_nums = image_train_nums

# 输出TFRecord文件的地址
filename = "./Data/train_images.tfrecords"
# 创建writer写入TFRecord文件
writer = tf.python_io.TFRecordWriter(filename)
for index in range(train_images_nums):
    # 将图像矩阵转换为字符串
    train_image_raw = train_images[index].tostring()
    # 将训练样例转换为Example Protocol Buffer,并将所有信息写入数据结构
    train_example = tf.train.Example(features=tf.train.Features(feature={
        "train_image_raw": _bytes_feature(train_image_raw),
        "train_image_label": _int64_feature(train_images_labels[index]),
        "train_image_height": _int64_feature(train_images_height),
        "train_image_width": _int64_feature(train_images_width),
        "train_image_channels": _int64_feature(train_images_channels),
    }))

    # 将一个Example写入TFRecord文件
    writer.write(train_example.SerializeToString())
writer.close()

"""将测试数据保存为TFRecorder"""
test_images = x_test
test_images_labels = y_test
test_images_height = image_height
test_images_width = image_width
test_images_channels = image_channels
test_images_nums = image_test_nums

# 输出TFRecord文件的地址
filename = "./Data/test_images.tfrecords"
# 创建writer写入TFRecord文件
writer = tf.python_io.TFRecordWriter(filename)
for index in range(test_images_nums):
    # 将图像矩阵转换为字符串
    test_image_raw = test_images[index].tostring()
    # 将训练样例转换为Example Protocol Buffer,并将所有信息写入数据结构
    test_example = tf.train.Example(features=tf.train.Features(feature={
        "test_image_raw": _bytes_feature(test_image_raw),
        "test_image_label": _int64_feature(test_images_labels[index]),
        "test_image_height": _int64_feature(test_images_height),
        "test_image_width": _int64_feature(test_images_width),
        "test_image_channels": _int64_feature(test_images_channels),
    }))

    # 将一个Example写入TFRecord文件
    writer.write(test_example.SerializeToString())
writer.close()
