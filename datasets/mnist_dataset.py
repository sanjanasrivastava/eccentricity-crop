import glob
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import pickle
import numpy as np
from random import randint

from datasets import dataset


class MNIST(dataset.Dataset):

    def __init__(self, opt):
        super(MNIST, self).__init__(opt)

        self.num_threads = 8
        self.output_buffer_size = 1024

        self.list_labels = range(0, 10)
        self.num_images_training = 60000
        self.num_images_test = 10000
        self.num_classes = 10 

#         self.num_images_epoch = self.opt.dataset.proportion_training_set*self.num_images_training
        self.num_images_epoch = opt.hyper.num_train_ex * self.num_classes
        self.num_images_val = self.num_images_training - self.num_images_epoch

        self.create_tfrecords()

    # Helper functions:
    def __unpickle(self, file_name):
        with open(file_name, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data

    # Virtual functions:
    def get_data_trainval(self):

        mnist_data = mnist.read_data_sets('../eccentricity-data/mnist', reshape=False, validation_size=0)	# 95% train, 5% validation as was used for cifar

        ptrain_addrs = list(mnist_data.train.images)
        ptrain_labels = list(mnist_data.train.labels)		# TODO these are np.uint8. Do they need to be int?
        pval_addrs = list(mnist_data.validation.images)
        pval_labels = list(mnist_data.validation.labels)

        counts = [0 for __ in range(len(self.list_labels))]
        train_addrs = []
        train_labels = []
        val_addrs = []
        val_labels = []

        for i in range(self.num_images_training):
            L = ptrain_labels[i]     # get the label L for the ith image
            if counts[L] < self.opt.hyper.num_train_ex:      # if we haven't yet reached <num_train_ex> examples for class L...
                train_addrs.append(ptrain_addrs[i])     # ...add the ith image and label to our final training set
                train_labels.append(ptrain_labels[i])
                counts[L] += 1      # increment counter
            else:   # if we have reached <num_train_ex> examples for class L...
                val_addrs.append(ptrain_addrs[i])    # ...add the ith image and label to our final validation set
                val_labels.append(ptrain_labels[i])

        return train_addrs, train_labels, val_addrs, val_labels

    def get_data_test(self):

        mnist_data = mnist.read_data_sets('../eccentricity-data/mnist', reshape=False, validation_size=int(self.num_images_val))
        test_addrs = list(mnist_data.test.images)
        test_labels = list(mnist_data.test.labels)

        return test_addrs, test_labels

    def preprocess_image(self, augmentation, standarization, image):
#         return image	# TODO trying to break
        if augmentation:
            # Randomly crop a [height, width] section of the image.
            #distorted_image = tf.random_crop(image, [self.opt.hyper.crop_size, self.opt.hyper.crop_size, 3])

            distorted_image = image
            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation.
            # NOTE: since per_image_standardization zeros the mean and makes
            # the stddev unit, this likely has no effect see tensorflow#1458.
            distorted_image = tf.image.random_brightness(distorted_image,
                                                         max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=0.2, upper=1.8)
        else:
            distorted_image = image

#         l = r = tf.random_uniform([self.opt.hyper.image_size, self.opt.hyper.background_size, 1], maxval=255)
#         background_image = tf.concat([l, distorted_image, r], 1)
#         t = b = tf.random_uniform([self.opt.hyper.background_size, self.opt.hyper.image_size + 2 * self.opt.hyper.background_size, 1], maxval=255)
#         background_image = tf.concat([t, background_image, b], 0)
# 
#         if self.opt.hyper.full_size:
#             # background_image = tf.image.resize(background_image, [140, 140], method=tf.image.ResizeMethod.BILINEAR)	# TODO bilinear is quadratic
#             background_image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(background_image, axis=0), [140, 140]), axis=0)
        background_image = distorted_image

        if standarization:
            # Subtract off the mean and divide by the variance of the pixels.
            float_image = tf.image.per_image_standardization(background_image)
            float_image.set_shape([self.opt.hyper.image_size, self.opt.hyper.image_size, 3])
        else:
            float_image = background_image 

        print('shape from preprocess:', float_image.shape)

        return float_image


