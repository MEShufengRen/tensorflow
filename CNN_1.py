import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data


mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuracy(v_xs,v_ys):
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

def weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    # stride [1,x_movement,y_movement,1]
    # Muast have strides[0]=stride[4]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #???

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,784],name='x_input')
    ys=tf.placeholder(tf.float32,[None,10],name='y_input')
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,1]) #第一个-1是代表图片数量n，设置-1后面后自动识别，1为通道数
# print(x_image.shape) # [n_samples, 28,28,1]

# conv1 layer
W_conv1 = weight_variable([5,5,1,32])
# patch 5x5, in size 1（黑白图因此输入通道数为1）, out size 32(输出通道数)
b_conv1= bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #output size 28x28x32 因为padding=same
h_pool1 = max_pool_2x2(h_conv1) #output 14x14x32

# conv2 layer
W_conv2 = weight_variable([5,5,32,64])
# patch 5x5, in size 32, out size 64(输出通道数)
b_conv2= bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2) #output 7x7x64

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

# func1 layer
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

# func2 layer
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
# the error between predition and real data

with tf.name_scope('cross_entropy'):
    # cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
    #                                         reduction_indices=[1]))
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,ys))

    tf.summary.scalar('loss',cross_entropy)

with tf.name_scope('train'):
    # train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('logs/CNN',sess.graph)
sess.run(tf.initialize_all_variables())
saver=tf.train.Saver()
# training
for i in range(1000):
    batch_xs, batch_ys=mnist.train.next_batch(100) #随机梯度下降
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50==0:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
        result=sess.run(merged,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1})
        writer.add_summary(result,i)

save_path=saver.save(sess,"my_net/save_net.ckpt")

# restore variable
# redefine same shape and same type for your variables


