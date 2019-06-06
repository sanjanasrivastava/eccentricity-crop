'''
Adapted from https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from util import summary as summ

num_classes = 10 
num_neurons = [32, 64, 1024]
# TODO add batch size from opt


# x = tf.placeholder('float', [None, 784])	# TODO adjust for different background sizes
# y = tf.placeholder('float')


def mnist_cnn(x, opt, labels_id, dropout_rate):
   
    reuse = False
    stride = opt.dnn.stride
    parameters = []
    activations = []
    inputs = []

    # conv1
    with tf.variable_scope('conv1', reuse=reuse) as scope:
        # kernel = tf.get_variable(initializer=tf.random_normal([5, 5, 1, num_neurons[0]]), name='weights')
        inputs.append(x)
        kernel = tf.get_variable('weights', shape=[5, 5, opt.dnn.num_input_channels, num_neurons[0]], initializer=xavier_initializer())
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        # biases = tf.get_variable(initializer=tf.random_normal([num_neurons[0]]), name='biases')
        biases = tf.get_variable('biases', shape=[num_neurons[0]], initializer=xavier_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv1, opt)
        activations += [conv1]
        parameters.append(kernel)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # TODO not going to do LRN for now, will if it becomes necessary (e.g. divergence or smth)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse) as scope:
        # kernel = tf.get_variable(initializer=tf.random_normal([5, 5, num_neurons[0], num_neurons[1]]), name='weights')
        kernel = tf.get_variable('weights', shape=[5, 5, num_neurons[0], num_neurons[1]], initializer=xavier_initializer())
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        # biases = tf.get_variable(initializer=tf.random_normal([num_neurons[1]]), name='biases')
        biases = tf.get_variable('biases', shape=[num_neurons[1]], initializer=xavier_initializer())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv2, opt)
        activations += [conv2]
        parameters.append(kernel)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    
    # fc3
    with tf.variable_scope('fc3', reuse=reuse) as scope:
        dim = int(np.prod(pool2.get_shape()[1:]))
        pool_vec = tf.reshape(pool2, [-1, dim])
        # weights = tf.get_variable(initializer=tf.random_normal([dim, num_neurons[2]]), name='weights')
        weights = tf.get_variable('weights', shape=[dim, num_neurons[2]], initializer=xavier_initializer())
        # biases = tf.get_variable(initializer=tf.random_normal([num_neurons[2]]), name='biases')
        biases = tf.get_variable('biases', shape=[num_neurons[2]], initializer=xavier_initializer())
        fc_ = tf.nn.relu(tf.matmul(pool_vec, weights) + biases, name=scope.name)
        fc3 = tf.nn.dropout(fc_, dropout_rate)

        activations += [fc3]
        parameters += [weights]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(fc3, opt)

    # Softmax not applied because the crossentropy function takes unscaled logits
    with tf.variable_scope('softmax_linear', reuse=reuse) as scope:
        # weights = tf.get_variable(initializer=tf.random_normal([num_neurons[2], num_classes]), name='weights')
        weights = tf.get_variable('weights', shape=[num_neurons[2], num_classes], initializer=xavier_initializer())
        # biases = tf.get_variable(initializer=tf.random_normal([num_classes]), name='biases')
        biases = tf.get_variable('biases', shape=[num_classes], initializer=xavier_initializer())
        output = tf.add(tf.matmul(fc3, weights), biases, name=scope.name)

        activations += [output]
        parameters += [weights]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(fc3, opt)

    return output, parameters, activations, inputs
        


