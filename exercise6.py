import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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

# Make up some real data
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

# add hidden layer
l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
# add output layer
prediction=add_layer(l1,10,1,n_layer=2,activation_function=None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys),
                                  reduction_indices=[1]), name='square_mean')
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
# train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    train_step=tf.train.AdamOptimizer(0.2).minimize(loss)


init=tf.initialize_all_variables()

sess = tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('logs/',sess.graph)
sess.run(init)

# plot
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

# training
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
        lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)

plt.pause(0)


