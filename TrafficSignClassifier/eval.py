# -*- coding:utf-8 -*-
import time
import numpy as np
import tensorflow as tf

import LeNet_5_Forward_Process as LeF
import LeNet_5_Train_Model as LeT
import Data_Import

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10


def evaluate():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
                           [Data_Import.test_images_nums, Data_Import.image_height, Data_Import.image_width, Data_Import.image_channels],
                           name="x-input")
        y_ = tf.placeholder(tf.int64, [Data_Import.test_images_nums], name="y-input")

        # 验证集
        validation_feed = {x: Data_Import.x_test, y_: Data_Import.y_test}

        y = LeF.inference(x, False, None)

        # 计算准备率
        correct_prediction = tf.equal(tf.argmax(y,1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(LeT.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # 找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(LeT.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    golbal_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validation_feed)
                    print("After %s training step(s),validation accuracy = %g." % (golbal_step, accuracy_score))

                else:
                    print("No checkpoint file found")
                    return

                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    evaluate()


if __name__ == "__main__":
    tf.app.run()
