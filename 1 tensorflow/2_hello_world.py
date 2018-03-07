"""
Lin Cheng
2018.01.16
Tensorflow Hello world

"""

#导入数据库
import tensorflow as tf

# 建立图
hello = tf.constant('hello tensorflow of world')


# 启动图
sess = tf.Session()

# 不需要初始化

# 不需要训练

# 显示
print(sess.run(hello))