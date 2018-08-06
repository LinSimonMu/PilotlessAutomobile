# -*- coding:utf-8 -*-

# 采用LeNet-5作为模型进行训练数据
import os
import numpy as np
import tensorflow as tf
import Data_Import
import LeNet_5_Forward_Process as LeF
import Image_Process as ImP

# 配置神经网络训练参数
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"


def train():
    """输入数据处理框架"""
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer(["./Data/train_images.tfrecords"], shuffle=False)
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "train_image_raw": tf.FixedLenFeature([], tf.string),
        "train_image_label": tf.FixedLenFeature([], tf.int64),
        "train_image_height": tf.FixedLenFeature([], tf.int64),
        "train_image_width": tf.FixedLenFeature([], tf.int64),
        "train_image_channels": tf.FixedLenFeature([], tf.int64),
    })

    image = features["train_image_raw"]
    label = tf.cast(features["train_image_label"], tf.int32)
    height = tf.cast(features["train_image_height"], tf.int32)
    width = tf.cast(features["train_image_width"], tf.int32)
    channels = tf.cast(features["train_image_channels"], tf.int32)

    # 原始图像解析出像素矩阵，根据图像尺寸还原图像
    decoded_image = tf.decode_raw(image, tf.uint8)
    decoded_image = tf.reshape(decoded_image, [height, width, channels])

    # 定义输入神经网络输入层图片的大小
    image_height = Data_Import.train_images_height
    image_width = Data_Import.train_images_width

    distorted_image = ImP.preprocess_for_train(decoded_image, image_height, image_width, None)

    # 将处理后的图像和标签整理成神经网络训练时需要的batch
    min_after_dequeue = 10000
    batch_size = 128
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size=batch_size,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

    x_train_batch = tf.placeholder(tf.float32, [BATCH_SIZE, image_height, image_width, LeF.IMAGE_CHANNELS],
                                   name="x-input")
    y_train_batch_ = tf.placeholder(tf.int32, [BATCH_SIZE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 前向传播
    y_train_batch = LeF.inference(x_train_batch, True, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程
    with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_train_batch,
                                                                   labels=y_train_batch_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 设置学习率衰减
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               Data_Import.image_train_nums // BATCH_SIZE, LEARNING_RATE_DECAY)

    # 执行梯度下降，并更新权值
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 神经网络准备工作：变量初始化、线程启动
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(TRAINING_STEPS):
            image_batch_feed, label_batch_feed = sess.run([image_batch, label_batch]) 
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x_train_batch: image_batch_feed, y_train_batch_: label_batch_feed})

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        # 停止所有线程
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
