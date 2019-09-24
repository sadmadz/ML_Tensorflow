from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist = tf.keras.datasets.mnist.load_data()
# y labels are oh-encoded


n_train = mnist.train.num_examples
# 55,000
n_validation = mnist.validation.num_examples
# 5000n_test = mnist.test.num_examples
# 10,000


n_input = 784
# input layer (28x28 pixels)
n_hidden1 = 512 # 1st hidden layer
n_hidden2 = 256 # 2nd hidden layer
n_hidden3 = 128 # 3rd hidden layer
n_output = 10
# output layer (0-9 digits)

learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5


X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)