
"""
Lin Cheng
2018.01.16
Tensorflow Introduction

"""

# package input
import tensorflow as tf
import numpy as np

# 使用Numpy 生成假数据
x_data = np.float32(np.random.rand(2, 100)) #随机输入
y_data = np.dot([0.1, 0.2], x_data) + 0.3

# 构造线性模型
b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1, 2], -1, 1))
y = tf.matmul(w, x_data) + b

# 最小方差
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# 初始化
init = tf.initialize_all_variables()


#启动图
sess = tf.Session()
sess.run(init)


# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b), sess.run(loss))
