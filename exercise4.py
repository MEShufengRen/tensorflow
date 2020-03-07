import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

state=tf.Variable(0,name='counter')
# print(state.name)
one=tf.constant(1)
new_value=tf.add(state, one)
update=tf.assign(state,new_value)

init=tf.initialize_all_variables() # must have if define variables


with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
