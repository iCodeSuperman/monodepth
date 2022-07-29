# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Monodepth data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


class MonodepthDataloader(object):
    """monodepth dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode):
        # https://its401.com/article/bofu_sun/89138003 代码注释参考
        self.data_path = data_path  # 数据路径
        self.params = params  # 类的参数图片尺寸等
        self.dataset = dataset  # 数据集
        self.mode = mode  # 模式，训练或者测试

        self.left_image_batch = None  # 先定义左右图片的batchsize，先不设定具体值
        self.right_image_batch = None

        # 生成文件名队列
        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        # 获取文件中的每行的内容
        _, line = line_reader.read(input_queue)
        # 取出图片名
        split_line = tf.string_split([line]).values
        # Tensor("strided_slice:0", shape=(), dtype=string, device= / device: CPU:0)
        # Tensor("strided_slice_1:0", shape=(), dtype=string, device= / device: CPU:0)

        # 如果测试非立体图那么我们只加载一张左图片
        # we load only one image for test, except if we trained a stereo model
        if mode == 'test' and not self.params.do_stereo:
            # 添加左图片路径
            left_image_path = tf.string_join([self.data_path, split_line[0]])
            # 打开左图片
            left_image_o = self.read_image(left_image_path)
        else:
            # 如果不是测试非立体图那么加载左右两张图片
            left_image_path = tf.string_join([self.data_path, split_line[0]])
            right_image_path = tf.string_join([self.data_path, split_line[1]])
            left_image_o = self.read_image(left_image_path)
            right_image_o = self.read_image(right_image_path)

        if mode == 'train':
            # 任意翻转图片
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            # 以0.5的概率左右图同时左右翻转，否则不翻转
            left_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
            right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o), lambda: right_image_o)

            # randomly augment images
            # 任意填充图片
            # 生成0到1之间的一个数字
            do_augment = tf.random_uniform([], 0, 1)
            # 以0.5 的概率左右图执行augment_image_pair函数，否则不变
            left_image, right_image = tf.cond(do_augment > 0.5,
                                              lambda: self.augment_image_pair(left_image, right_image),
                                              lambda: (left_image, right_image))
            # 设置左右图尺寸为none*none*3
            left_image.set_shape([None, None, 3])
            right_image.set_shape([None, None, 3])
            # Tensor("cond_4/Merge:0", shape=(256, 512, 3), dtype=float32, device= / device: CPU:0)
            # Tensor("cond_4/Merge_1:0", shape=(256, 512, 3), dtype=float32, device= / device: CPU:0)

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            # 对训练集乱序放置，设置容器
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            # 如果是测试立体图，将图片与翻转后的图片进行进行拼接，获得高维度图片，一个维度是原图，另一个维度是翻转后的图片。
            self.left_image_batch, self.right_image_batch = tf.train.shuffle_batch([left_image, right_image],
                                                                                   params.batch_size, capacity,
                                                                                   min_after_dequeue,
                                                                                   params.num_threads)
            # image_batch 尺寸
            # Tensor("shuffle_batch:0", shape=(8, 256, 512, 3), dtype=float32, device= / device: CPU:0)
            # Tensor("shuffle_batch:1", shape=(8, 256, 512, 3), dtype=float32, device= / device: CPU:0)



        elif mode == 'test':
            self.left_image_batch = tf.stack([left_image_o, tf.image.flip_left_right(left_image_o)], 0)
            self.left_image_batch.set_shape([2, None, None, 3])

            if self.params.do_stereo:
                self.right_image_batch = tf.stack([right_image_o, tf.image.flip_left_right(right_image_o)], 0)
                self.right_image_batch.set_shape([2, None, None, 3])

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path)))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image = image[:crop_height, :, :]

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image
