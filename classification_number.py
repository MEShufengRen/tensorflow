import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name='layer%d' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('W'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='weights')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b') #不推荐为0
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            output=Wx_plus_b
        else:
            output=activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/output',output)
        return output

def compute_accuracy(prediction,v_xs,v_ys):
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,784],name='x_input')
    ys=tf.placeholder(tf.float32,[None,10],name='y_input')

# add hidden layer
prediction=add_layer(xs,784,10,n_layer=1,activation_function=tf.nn.softmax)

# the error between predition and real data

with tf.name_scope('cross_entropy'):
    corss_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                            reduction_indices=[1]))

    tf.summary.scalar('loss',corss_entropy)

with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(corss_entropy)
    # train_step=tf.train.AdamOptimizer(0.2).minimize(loss)

sess = tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('logs/',sess.graph)
sess.run(tf.initialize_all_variables())

# training
for i in range(1000):
    batch_xs, batch_ys=mnist.train.next_batch(100) #随机梯度下降
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        print(compute_accuracy(prediction, mnist.test.images, mnist.test.labels))
        result=sess.run(merged,feed_dict={xs:batch_xs,ys:batch_ys})
        writer.add_summary(result,i)



