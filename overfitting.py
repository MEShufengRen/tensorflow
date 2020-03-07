import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name='layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('W'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='weights')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b') #不推荐为0
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)
        if activation_function is None:
            output=Wx_plus_b
        else:
            output=activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/output',output)
        return output

def compute_accuracy(v_xs,v_ys):
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# load data
digits=load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3)


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,64],name='x_input')
    ys=tf.placeholder(tf.float32,[None,10],name='y_input')
    keep_prob=tf.placeholder(tf.float32)

# add hidden layer
l1=add_layer(xs,64,50,n_layer='l1',activation_function=tf.nn.tanh)
prediction=add_layer(l1,50,10,n_layer='l2',activation_function=tf.nn.softmax)

# the error between predition and real data

with tf.name_scope('cross_entropy'):
    corss_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                            reduction_indices=[1]))

    tf.summary.scalar('loss',corss_entropy)

with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.6).minimize(corss_entropy)
    # train_step=tf.train.AdamOptimizer(0.2).minimize(loss)

sess = tf.Session()
merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter('logs/overfitting/train',sess.graph)
test_writer=tf.summary.FileWriter('logs/overfitting/test',sess.graph)
sess.run(tf.initialize_all_variables())

# training
for i in range(1000):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train, keep_prob:0.8})
    if i%50==0:
        print(compute_accuracy(X_test, y_test))
        train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result=sess.run(merged,feed_dict={xs:X_test, ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)

