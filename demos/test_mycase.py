import unittest
import tensorflow as tf
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        filenames_file = "/Users/icodeboy/PycharmProjects/monodepth/utils/filenames/kitti_train_files.txt"
        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        # 获取文件中的每行的内容
        _, line = line_reader.read(input_queue)
        print(line)
        # 取出图片名
        split_line = tf.string_split([line]).values

if __name__ == '__main__':
    unittest.main()
