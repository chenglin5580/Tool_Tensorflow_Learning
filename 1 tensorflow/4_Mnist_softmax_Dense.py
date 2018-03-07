"""
Lin Cheng
2018.01.16
Tensorflow MNIST
"""

# import package
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# bulid graph
x_image = tf.placeholder("float", [None, 784])


y_pre = tf.nn.softmax(tf.layers.dense(inputs=x_image, units=10))
y_re = tf.placeholder("float", [None, 10])

# loss
cross_entry = -tf.reduce_sum(y_re*tf.log(y_pre))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cross_entry)

# accuracy
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_re, 1))
accuarcy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Execute Graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x_image: batch_xs, y_re: batch_ys})
    if i % 10 == 0:
        print(sess.run(accuarcy, feed_dict={x_image: mnist.test.images, y_re: mnist.test.labels}))













