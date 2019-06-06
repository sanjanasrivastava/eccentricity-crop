import tensorflow as tf


from nets.alexnet import Alexnet as net_Alexnet
from nets.mnist_cnn import mnist_cnn as net_mnist_cnn
from nets.mnist_fcn import mnist_fcn as net_mnist_fcn
from util import summary as summ


def Alexnet(x, dropout_rate, opt, labels_id):
    return net_Alexnet(x, opt, labels_id, dropout_rate)

def mnist_cnn(x, dropout_rate, opt, labels_id):
    return net_mnist_cnn(x, opt, labels_id, dropout_rate)

def mnist_fcn(x, dropout_rate, opt, labels_id):
    return net_mnist_fcn(x, opt, labels_id, dropout_rate)
