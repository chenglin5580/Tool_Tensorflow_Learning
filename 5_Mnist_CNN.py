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



# build graph
x = tf.placeholder("float", [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])


h_conv1 = tf.nn.relu(tf.layers.conv2d(x_image, filters=32, kernel_size=[5, 5], padding='SAME'))
h_pool1 = tf.layers.max_pooling2d(h_conv1, pool_size=[2, 2], strides=[2, 2])

h_conv2 = tf.nn.relu(tf.layers.conv2d(h_pool1, filters=64, kernel_size=[5, 5], padding='SAME'))
h_pool2 = tf.layers.max_pooling2d(h_conv2, pool_size=[2, 2], strides=[2,2])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.layers.dense(h_pool2_flat, 1024)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv=tf.nn.softmax(tf.layers.dense(h_fc1_drop, 10))

y_re = tf.placeholder("float", [None, 10])


cross_entropy = -tf.reduce_sum(y_re*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_re,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Execute Graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(20000):
  batch = mnist.train.next_batch(50)
  sess.run(train_step, feed_dict={x: batch[0], y_re: batch[1], keep_prob: 0.5})
  if i % 100 == 0:
    train_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_re: mnist.test.labels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

print("finial test accuracy %g", sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                               y_re: mnist.test.labels, keep_prob: 1.0}))
