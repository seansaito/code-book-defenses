from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from .layers import weight_variable_msra, weight_variable_xavier, bias_variable, batch_norm, conv2d, \
    dropout, avg_pool


class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class DenseNetBlockLayer(Layer):

    def __init__(self, out_features, kernel_size, is_training, keep_prob, bc_mode,
                 strides=[1, 1, 1, 1], padding='SAME', block_idx=None):
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.bc_mode = bc_mode
        self.strides = strides
        self.padding = padding
        self.block_idx = block_idx

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        # BN
        output = batch_norm(_input, self.is_training,
                            name='block_composite_bn_{}'.format(self.block_idx))
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = conv2d(output, out_features=out_features, kernel_size=kernel_size,
                        name='block_composite_{}'.format(self.block_idx))
        # dropout(in case of training and in case it is no 1.0)
        output = dropout(output, self.keep_prob, self.is_training)
        return output

    def bottleneck(self, _input, out_features):
        output = batch_norm(_input, self.is_training,
                            name='block_bottleneck_bn_{}'.format(self.block_idx))
        output = tf.nn.relu(output)
        inter_features = out_features * 4
        output = conv2d(
            output, out_features=inter_features, kernel_size=1,
            padding='VALID', name='block_bottleneck_{}'.format(self.block_idx))
        output = dropout(output, self.keep_prob, self.is_training)
        return output

    # NCHW implementation
    def fprop(self, _input):
        if not self.bc_mode:
            output = self.composite_function(_input, out_features=self.out_features, kernel_size=3)
        else:
            bottleneck_out = self.bottleneck(_input, out_features=self.out_features)
            output = self.composite_function(bottleneck_out, out_features=self.out_features,
                                             kernel_size=3)
        # Concatenate _input with output - residual connections
        out = tf.concat(axis=1, values=(_input, output))
        return out


class DenseNetTransitionLayer(Layer):

    def __init__(self, reduction, is_training, keep_prob, block_idx):
        self.reduction = reduction
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.block_idx = block_idx

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        # BN
        output = batch_norm(_input, self.is_training,
                            name='transition_composite_bn_{}'.format(self.block_idx))
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = conv2d(output, out_features=out_features, kernel_size=kernel_size,
                        name='transition_conv2d_{}'.format(self.block_idx))
        # dropout(in case of training and in case it is no 1.0)
        output = dropout(output, self.keep_prob, self.is_training)
        return output

    # NCHW implementation
    def fprop(self, _input):
        out_features = int(int(_input.get_shape()[-3]) * self.reduction)
        output = self.composite_function(_input, out_features=out_features, kernel_size=1)
        # Avg pooling
        output = avg_pool(output, k=2)
        return output


class DenseNetFlattenLayer(Layer):

    def __init__(self, is_training):
        self.is_training = is_training

    # NCHW implementation
    def fprop(self, _input):
        # BN
        output = batch_norm(_input, self.is_training, name='flatten_bn')
        # elu
        output = tf.nn.elu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-3])
        output = tf.reshape(output, [-1, features_total], name='flattened_features')

        return output


class Conv2DLayer(Layer):

    def __init__(self, out_features, kernel_size, strides=[1, 1, 1, 1], padding='SAME', name=None):
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name = name

    # NCHW implementation
    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = (input_channels, self.kernel_size, self.kernel_size, self.out_features)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        self.kernel = weight_variable_msra([input_channels, self.kernel_size, self.kernel_size,
                                            self.out_features], name=self.name)
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, _input):
        in_features = int(_input.get_shape()[-3])
        self.kernel = weight_variable_msra([self.kernel_size, self.kernel_size, in_features,
                                            self.out_features], name=self.name)
        return tf.nn.conv2d(_input, self.kernel, self.strides, self.padding, name=self.name,
                            data_format='NCHW')


class AvgPoolLayer(Layer):

    def __init__(self, k):
        self.ksize = [1, k, k, 1]
        self.strides = [1, k, k, 1]
        self.padding = 'VALID'

    def fprop(self, _input):
        return tf.nn.avg_pool(_input, self.ksize, self.strides, self.padding)


class BatchNormLayer(Layer):

    def __init__(self, is_training):
        self.is_training = is_training

    def fprop(self, _input):
        return tf.contrib.layers.batch_norm(_input, scale=True, is_training=self.is_training,
                                            updates_collection=None)


class DropoutLayer(Layer):

    def __init__(self, keep_prob, is_training):
        self.keep_prob = keep_prob
        self.is_training = is_training

    def fprop(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output


class Linear(Layer):

    def __init__(self, num_hid, layer_name=None):
        self.num_hid = num_hid
        self.name = layer_name

    def fprop(self, x):
        input_dim = int(x.get_shape()[-1])
        self.W = weight_variable_xavier([input_dim, self.num_hid], name='{}_W'.format(self.name))
        self.bias = bias_variable([self.num_hid], name='{}_b'.format(self.name))
        return tf.add(tf.matmul(x, self.W), self.bias, name='{}_logits'.format(self.name))


class ReLU(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.relu(x, name='relu_activation')


class ELU(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.elu(x, name='elu_activation')


class Tanh(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.tanh(x, name='tanh_activation')


class Sigmoid(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.sigmoid(x, name='sigmoid_activation')


class Reshape(Layer):

    def __init__(self, shape):
        self.shape = shape

    def fprop(self, x):
        return tf.reshape(x, self.shape)


class Concat(Layer):

    def __init__(self, axis):
        self.axis = axis

    def fprop(self, x):
        """
        x should be a tuple of tensors
        """
        return tf.concat(axis=self.axis, values=x)


class Softmax(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x, name='softmax_activation')
