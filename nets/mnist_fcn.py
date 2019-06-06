'''
Adapted from question in https://stats.stackexchange.com/questions/376312/mnist-digit-recognition-what-is-the-best-we-can-get-with-a-fully-connected-nn-o

Trying Xavier initialization and ReLU, even though the stackexchange says not to, just so that it is as similar to the CNN as possible
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from util import summary as summ

num_classes = 10
# num_neurons = [784, 200, 80, 10]
num_neurons = [2000, 800, 100, 10]


def mnist_fcn(x, opt, labels_id, dropout_rate):

  reuse = False
  parameters = []
  activations = []
  inputs = [x]

  # Flatten 
  with tf.variable_scope('flatten', reuse=reuse) as scope:
    xvec = tf.contrib.layers.flatten(x)
    dim = xvec.get_shape()[1]
    print('flattened input shape:', xvec)

  # fc0
  with tf.variable_scope('fc0', reuse=reuse) as scope:
    weights = tf.get_variable('weights', shape=[dim, num_neurons[0]], initializer=xavier_initializer())
    biases = tf.get_variable('biases', shape=[num_neurons[0]], initializer=xavier_initializer())
    fc_ = tf.nn.relu(tf.add(tf.matmul(xvec, weights), biases), name=scope.name)
    fc0 = tf.nn.dropout(fc_, dropout_rate)

    activations.append(fc0)
    parameters.append(weights)

  # fc1
  with tf.variable_scope('fc1', reuse=reuse) as scope:
    weights = tf.get_variable('weights', shape=[num_neurons[0], num_neurons[1]], initializer=xavier_initializer())
    biases = tf.get_variable('biases', shape=[num_neurons[1]], initializer=xavier_initializer())
    fc_ = tf.nn.relu(tf.add(tf.matmul(fc0, weights), biases), name=scope.name)
    fc1 = tf.nn.dropout(fc_, dropout_rate)

    activations.append(fc1)
    parameters.append(weights)

  # fc2
  with tf.variable_scope('fc2', reuse=reuse) as scope:
    weights = tf.get_variable('weights', shape=[num_neurons[1], num_neurons[2]], initializer=xavier_initializer())
    biases = tf.get_variable('biases', shape=[num_neurons[2]], initializer=xavier_initializer())
    fc_ = tf.nn.relu(tf.add(tf.matmul(fc1, weights), biases), name=scope.name)
    fc2 = tf.nn.dropout(fc_, dropout_rate)

    activations.append(fc2)
    parameters.append(weights)

  # fc3
  with tf.variable_scope('fc3', reuse=reuse) as scope:
    weights = tf.get_variable('weights', shape=[num_neurons[2], num_neurons[3]], initializer=xavier_initializer())
    biases = tf.get_variable('biases', shape=[num_neurons[3]], initializer=xavier_initializer())
    output = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    activations.append(output)
    parameters.append(weights)

  return output, parameters, activations, inputs




