import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],[2]])
product=tf.matmul(matrix1,matrix2)

# # method1
# sess=tf.Session()
# result=sess.run(product)
# print(result)

# method2
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)
