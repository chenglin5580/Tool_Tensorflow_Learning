"""
Lin Cheng
2018.01.16
Tensorflow Basic Use
"""


""" 基本概念
# node 进行 operation

# edge 传递 tensor

# graph

# 在python中， tensor is numpy

"""

""" 流程
#  创建图  op 的执行步骤 被描述成一个图
tf.constant()
tf.Variable()
tf.matmul()
tf.assign(state, new_value)
tf.add()
#  执行阶段  使用会话执行执行图中的 op
sess = tf.Session()
if there is varibale
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
fetch:
    sess.run()
feed:
    input1 = tf.placeholder(tf.types.float32)
    print sess.run([output], feed_dict={input1:[7.], input2:[2.]})    

"""




