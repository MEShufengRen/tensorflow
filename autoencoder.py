import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/',one_hot=False)

# Parameters
learning_rate=0.01
training_epochs=5
batch_size=256
display_step=1
examples_to_show=10

# Network Parameters
n_input=784

# tf Graph input (only picture)
X=tf.placeholder('float32',[None, n_input])

# hidden layer settings
n_hidden_1=256
n_hidden_2=128
weights={
    'encoder_h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_input])),

}

biases={
    'encoder_b1': tf.Variable(tf.constant(0.1,shape=[n_hidden_1])),
    'encoder_b2': tf.Variable(tf.constant(0.1,shape=[n_hidden_2])),

    'decoder_b1': tf.Variable(tf.constant(0.1,shape=[n_hidden_1])),
    'decoder_b2': tf.Variable(tf.constant(0.1,shape=[n_input])),
}


def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                 biases['encoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                 biases['encoder_b2']))
    return layer_2


def decoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                 biases['decoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                 biases['decoder_b2']))
    return layer_2

# construct model
encoder_op=encoder(X)

decoder_op=decoder(encoder_op)

# prediction
y_pred=decoder_op
# Targets(labels) are the input data
y_true=X

# Define loss and optimizer , minimize the squared error
cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    total_batch=int(mnist.train.num_examples/batch_size)
    # Training Cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys=mnist.train.next_batch(batch_size)
            _, c=sess.run(optimizer, feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch%display_step==0:
            print("Epoch: %04d" % (epoch+1),
                  "cost= .9f" %c)
    print("Optimization Finished")

    encoder_decoder=sess.run(y_pred,feed_dict={X: mnist.test.images[:examples_to_show]})
    f, a =plt.subplot(2, examples_to_show, figsize=(10,2))

