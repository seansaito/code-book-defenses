from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('conv_kernel'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv2d(_input, out_features, kernel_size,
           strides=[1, 1, 1, 1], padding='SAME', name='kernel', data_format='NCHW'):
    in_features = int(_input.get_shape()[-3])
    kernel = weight_variable_msra(
        [kernel_size, kernel_size, in_features, out_features],
        name=name)
    variable_summaries(kernel)
    output = tf.nn.conv2d(_input, kernel, strides, padding, data_format=data_format)
    return output


def avg_pool(_input, k, data_format='NCHW'):
    ksize = [1, 1, k, k]
    strides = [1, 1, k, k]
    padding = 'VALID'
    output = tf.nn.avg_pool(_input, ksize, strides, padding, data_format=data_format)
    return output


def batch_norm(_input, is_training, name, data_format='NCHW'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=is_training,
            updates_collections=None, data_format=data_format, fused=True)
        return output


def dropout(_input, keep_prob, is_training):
    if keep_prob < 1:
        output = tf.cond(
            is_training,
            lambda: tf.nn.dropout(_input, keep_prob),
            lambda: _input
        )
    else:
        output = _input
    return output


def weight_variable_msra(shape, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())


def weight_variable_xavier(shape, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape, name='bias'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)
