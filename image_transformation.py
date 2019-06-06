import numpy as np
import tensorflow as tf


MAX_INPUT_SIZE = 140


def resize_object_fixed_background(imc, opt):
  '''
  Resizes image up to requisite size, then adds given background size.
  Assumes constant background_size
  Assumes full_size image (140-by-140)
  '''
 
  background_size = tf.constant([int((opt.hyper.background_size * MAX_INPUT_SIZE) / (opt.hyper.image_size + 2 * opt.hyper.background_size))])
  image_size = MAX_INPUT_SIZE - (2 * background_size)

  imc = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(imc, axis=0), tf.concat([image_size, image_size], axis=0)), axis=0)
  imc = _add_background_matrices(imc, background_size, image_size)

  return imc, [opt.hyper.batch_size, MAX_INPUT_SIZE, MAX_INPUT_SIZE, 1]


def resize_object_random_background(imc, opt):
  '''
  Resizes image up to a requisite size, then adds random background size.
  Assumes full_size image (140-by-140)
  '''
  
  background_size = tf.random_uniform([1], maxval=(MAX_INPUT_SIZE - opt.hyper.image_size) // 2, dtype=tf.int32)
  image_size = MAX_INPUT_SIZE - (2 * background_size)
  imc = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(imc, axis=0), tf.concat([image_size, image_size], axis=0)), axis=0)

  imc = _add_background_matrices(imc, background_size, image_size)
  
  return imc, [opt.hyper.batch_size, MAX_INPUT_SIZE, MAX_INPUT_SIZE, 1]


def resize_object_fixed_background_invertedpyramid(imc, opt):
  '''
  Resizes image up to requisite size, then adds given background size, then applies invpyr transformation.
  Assumes constant background_size
  Assumes full_size image (140-by-140)
  '''
  
  imc, __ = resize_object_fixed_background(imc, opt)
  imc = _make_inverted_pyramid(imc, opt)
  return imc, [opt.hyper.batch_size, opt.hyper.image_size, opt.hyper.image_size, 5]


def resize_object_random_background_invertedpyramid(imc, opt):
  '''
  Resizes image up to a requisite size, then adds random background size, then applies invpyr transformation.
  Assumes full_size image
  '''

  imc, __ = resize_object_random_background(imc, opt)
  imc = _make_inverted_pyramid(imc, opt)
  return imc, [opt.hyper.batch_size, opt.hyper.image_size, opt.hyper.image_size, 5]




# Util
def _add_background_matrices(imc, background_size, image_size):

  l = tf.random_uniform(tf.concat([image_size, background_size, tf.constant([1])], axis=0), maxval=255)
  r = tf.random_uniform(tf.concat([image_size, background_size, tf.constant([1])], axis=0), maxval=255)
  t = tf.random_uniform(tf.concat([background_size, (background_size * 2) + image_size, tf.constant([1])], axis=0), maxval=255)
  b = tf.random_uniform(tf.concat([background_size, (background_size * 2) + image_size, tf.constant([1])], axis=0), maxval=255)

  imc = tf.concat([l, imc, r], 1)
  imc = tf.concat([t, imc, b], 0)

  return imc


def _make_inverted_pyramid(imc, opt, pyramid_depth=5):
  '''
  Converts image to inverted pyramid format.
  '''
  coords = np.linspace(0, 1, num=(pyramid_depth * 2) + 1)
  boxes = [[coords[i], coords[i], coords[-i - 1], coords[-i - 1]] for i in range(pyramid_depth)]
  imc_ip = tf.image.crop_and_resize(tf.expand_dims(imc, axis=0), boxes, 
                                    [0 for __ in range(len(boxes))], [opt.hyper.image_size, opt.hyper.image_size], method='bilinear')
  imc_ip = tf.transpose(imc_ip, perm=[3, 1, 2, 0])
  imc_ip = tf.squeeze(imc_ip, axis=0)

  return imc_ip
