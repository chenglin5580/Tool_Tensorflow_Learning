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
