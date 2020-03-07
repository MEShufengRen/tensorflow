import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS=20
BATCH_SIZE=50
INPUT_SIZE=1
OUTPUT_SIZE=1
CELL_SIZE=10
LR=0.006
BATCH_START_TEST=0

def get_batch():
    # global BATCH_START, TIME_STEPS
    xs=np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE,TIME_STEPS))
    seq=np.sin(xs)
    res=np.cos(xs)
    BATCH_START=+TIME_STEPS
    # plt.plot(xs[0,:],res[0,:],'r',xs[0,:],seq[0,:],'b--')
    # plt.show()
    return [seq[:,:,np.newaxis], res[:,:,np.newaxis], xs]

class LSTMRNN:
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps=n_steps
        self.input_size=input_size
        self.output_size=output_size
        self.cell_size=cell_size
        self.batch_size=batch_size

        with tf.name_scope('inputs'):
            self.xs=tf.place_holder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys=tf.place_holder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
            with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.variable_scope('cost'):
            self.compute_cost()
        with tf.variable_scope('train'):
            self.train_op=tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self):

    def add_cell(self):

    def add_output_layer(self):

    def comput_cost(self):

    def _weight_variable(self,shape,name='weights'):
        return tf.Variable(tf.random_normal(shape),name=name)

    def _bias_variable(self,shape,name='biases'):
        return tf.Variable(tf.constant(0.1, shape=shape),name=name)

if __name__=='main':
    model=LSTMRNN(TIME_STEPS, INPUT_SIZE,OUTPUT_SIZE,CELL_SIZE,BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.initial_all_variable())
        for i in range(200):
            seq, res, xs = get_batch()
            if i==0:
                feed_dict={model.xs:seq, model.ys:res}
            else:
                feed_dict={model.xs:seq, model.ys:res, model.cell_init_state:state}
        _, cost, state,pred=sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred]
            ,feed_dict=feed_dict
        )
        # plotting
        plt.plot(xs,res,'r', xs, pred[:TIME_STEPS],'b-')
        plt.ylim((-1.2,1.2))
        plt.draw()
        plt.pause(0.1)

        if i%20==0:
            print('cost' )

