import tensorflow as tf
import numpy as np

# 构建数组
arr = np.arange(1, 17, 1).reshape((2, 2, 4))
print(arr)
print("==============================")

# 调用cumsum函数
arr2 = tf.cumsum(arr, axis=0)
print(arr2)
