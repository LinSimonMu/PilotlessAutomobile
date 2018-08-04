import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def distort_color(image, color_ordering=0):
    """
    :param image:
    :param color_ordering:
    :return:
    给定一张图片，随机调整图像的色彩。调整亮度、对比度、饱和度和色相的顺序会影响最后得到的结果，所以可以定义多种不同的顺序
    """
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    elif color_ordering == 3:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, bbox):
    """

    :param image:
    :param height:
    :param width:
    :param bbox:
    :return:
    给定一张解码后的图像，目标图像的尺寸以及图像上的标注框，此函数可对给出的图像进行预处理。
    函数的输入图像是图像识别问题中原始的训练图像，而输出则是神经网络模型的输入层。
    注意：这里只处理模型的训练输数据，对于预测数据，一般不需要使用随机变换的步骤
    """
    # 没有提供标注框时，则认为整个图像就是需要关注部分
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    # 转换图像的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox,
                                                                      min_object_covered=0.1)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图像调整为神经网络输入层的大小
    distorted_image = tf.image.resize_images(distorted_image, (height, width), method=1)
    # 随机左右翻转
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(3))
    return distorted_image