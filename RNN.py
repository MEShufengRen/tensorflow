import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

# hyperparameters
lr=0.001
training_iters=1000
batch_size=128

n_inputs=28 # MNIST data input(img shape: 28*28)
n_steps=28 # time steps
n_hidden_units=128 # neurons in hidden layer
n_classes=10 # MNIST classes 0-9 digits

# tf Graph input
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

# Define weights
weights={
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

biases={
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

def RNN(X,weights,biases):
    # hidden layer for input to cell
    ############################################
    # X_in=X #这个应该是对的
    # X(128 batch, 28 steps, 28 input)
    # ==>(128*28,28inputs)
    X=tf.reshape(X,[-1,n_inputs])
    # X_in==>(128batch*28steps,128 hidden)
    X_in=tf.matmul(X,weights['in'])+biases['in']
    # X_in==>(128batch,28steps,128 hidden)
    X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_units])
    # cell
    ############################################
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,
                                           forget_bias=1.0,state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, m_state) 主线state和分线state
    init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)

    outputs, states=tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state,time_major=False)

    #hidden layer for output as the final results
    ############################################
    # results=tf.matmul(states[1],weights['out']+biases['out'])

    # or unpack to list [(batch,outputs)..]*steps
    outputs=tf.unstack(tf.transpose(outputs,[1,0,2]))  # states is the last outputs
    results=tf.matmul(outputs[-1],weights['out']+biases['out'])


    return results

def compute_accuracy(v_xs,v_ys):
    y_pre=sess.run(pred,feed_dict={x:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={x:v_xs,y:v_ys})
    return result

pred=RNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)


# correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
# accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
batch_x_test,batch_y_test=mnist.test.next_batch(batch_size)
batch_x_test=batch_x_test.reshape([batch_size,n_steps,n_inputs])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(training_iters):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})
        if i%20==0:
            x_test=mnist.test.images
            # print(x_test.shape[0])
            print(compute_accuracy(batch_x_test, batch_y_test))
