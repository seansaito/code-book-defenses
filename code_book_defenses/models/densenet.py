from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import logging
import os
import shutil
import time
from datetime import timedelta

import numpy as np
import progressbar
import tensorflow as tf
from sklearn.metrics import accuracy_score

from code_book_defenses.cleverhans.model import Model, NoSuchLayerError
from code_book_defenses.config import codebook_to_loss_fn
from code_book_defenses.constants import TARGET_TYPES
from .cleverhans_layers import Conv2DLayer, Linear, Tanh, Softmax, DenseNetBlockLayer, \
    DenseNetTransitionLayer, DenseNetFlattenLayer
from .code_book_generator import CodeBookGenerator

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

logger = logging.getLogger(__name__)


class DenseNetCleverhans(Model):

    def __init__(self, data_provider, growth_rate, depth, total_blocks, keep_prob, target_type,
                 weight_decay,
                 nesterov_momentum, model_type, dataset, should_save_logs, should_save_model,
                 experiment,
                 renew_logs=False, reduction=1.0, bc_mode=False, should_save_centroids=True,
                 latent_dim=128, gpus='0',
                 **kwargs):
        super(DenseNetCleverhans, self).__init__()

        # Set GPUs to use
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        # For storing the cleverhans layers
        self.layers = []
        self.layer_names = []
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.target_type = target_type
        self.loss_fn_name = codebook_to_loss_fn[self.target_type]
        self.depth = depth
        self.latent_dim = latent_dim
        logger.info('Centroids type: {}'.format(self.target_type))
        self.centroids = CodeBookGenerator().generate_codebooks(**{'target_type': target_type,
                                                                   'n_classes': self.n_classes,
                                                                   'latent_dim': self.latent_dim})
        logger.info('Shape of centroids is: {}'.format(self.centroids.shape))
        self.experiment = experiment
        self.growth_rate = growth_rate
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            logger.info("Build %s model with %d blocks, "
                        "%d composite layers each." % (
                            model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            logger.info("Build %s model with %d blocks, "
                        "%d bottleneck layers and %d composite layers each." % (
                            model_type, self.total_blocks, self.layers_per_block,
                            self.layers_per_block))
        logger.info("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0
        if should_save_centroids:
            self.save_centroids()
        self.graph = tf.Graph()
        self._define_inputs()
        self._build_graph()
        self._initialize_session(self.graph)
        self._count_trainable_params()

    def fprop(self, x, set_ref=False):
        states = []
        logger.info('Propagating Forward')
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        self.states = dict(zip(self.get_layer_names(), states))
        return self.states

    def get_layer(self, x, layer):
        output = self.fprop(x)
        try:
            requested = output[layer]
        except KeyError:
            raise NoSuchLayerError()
        return requested

    def _initialize_layer_names(self):
        if isinstance(self.layers[-1], Softmax):
            self.layers[-1].name = 'probs'
            self.layers[-2].name = 'logits'
        else:
            self.layers[-1].name = 'logits'

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self.layer_names.append(name)

    def _initialize_session(self, graph):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=graph)
        self.merged = tf.summary.merge_all()

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        logger.info("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            if not os.path.exists('saves/{}'.format(self.target_type)):
                os.makedirs('saves/{}'.format(self.target_type))
            save_path = 'saves/%s/%s' % (self.target_type, self.model_identifier + '-' +
                                         datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
            if os.path.exists(save_path):
                save_path = save_path
            else:
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    def set_save_path(self, save_path):
        self._save_path = save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            if not os.path.exists('logs/{}'.format(self.target_type)):
                os.makedirs('logs/{}'.format(self.target_type))
            logs_path = 'logs/%s/%s' % (self.target_type, self.model_identifier + '-' +
                                        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset={}_{}_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name, self.experiment,
            self.target_type)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def save_centroids(self):
        np.save(self.save_path + '_centroids.npy', self.centroids)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        logger.info("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            logger.info("mean loss: %f\n mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        if self.target_type == TARGET_TYPES.ONEHOT_CE:
            self.centroids_ph = tf.placeholder(
                tf.float32, shape=[None, self.n_classes], name='centroids'
            )
        else:
            self.centroids_ph = tf.placeholder(
                tf.float32,
                shape=[None].extend(self.centroids.shape),
                name='centroids')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

        # Placeholders for storing training and validation accuracies
        self.train_acc_ph = tf.placeholder(tf.float32, shape=(), name='train_acc')
        self.train_acc_summary = tf.summary.scalar('train_accuracy', self.train_acc_ph)

    """
    Methods for building the model
    """

    def add_internal_layer(self, growth_rate, block_idx):
        self.layers.append(DenseNetBlockLayer(out_features=growth_rate, kernel_size=3,
                                              is_training=self.is_training,
                                              keep_prob=self.keep_prob, bc_mode=self.bc_mode,
                                              block_idx=block_idx))

    def add_block(self, growth_rate, layers_per_block, block_idx):
        for idx, layer in enumerate(range(layers_per_block)):
            self.add_internal_layer(growth_rate=growth_rate,
                                    block_idx='{}_{}'.format(block_idx, idx))

    def transition_layer(self, block_idx):
        self.layers.append(
            DenseNetTransitionLayer(reduction=self.reduction, is_training=self.is_training,
                                    keep_prob=self.keep_prob, block_idx=block_idx))

    def transition_layer_to_classes(self):
        self.layers.append(DenseNetFlattenLayer(is_training=self.is_training))
        if self.target_type == TARGET_TYPES.RANDOM:
            self.layers.append(Linear(num_hid=self.latent_dim, layer_name='last'))
        else:
            self.layers.append(Linear(num_hid=self.n_classes, layer_name='last'))

        if self.target_type == TARGET_TYPES.RANDOM:
            logger.info('Using tanh')
            self.layers.append(Tanh())
        else:
            logger.info('Using softmax')
            self.layers.append(Softmax())

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block

        with tf.variable_scope('initial_conv', reuse=tf.AUTO_REUSE):
            self.layers.append(Conv2DLayer(out_features=self.first_output_features,
                                           kernel_size=3, name='initial_conv'))

        with tf.variable_scope('blocks', reuse=tf.AUTO_REUSE):
            for block_idx in range(self.total_blocks):
                print(block_idx)
                with tf.name_scope('Block_{}'.format(block_idx)):
                    self.add_block(growth_rate=growth_rate, layers_per_block=layers_per_block,
                                   block_idx=block_idx)

                if block_idx != self.total_blocks - 1:
                    with tf.name_scope('Transition_layer_{}'.format(block_idx)):
                        self.transition_layer(block_idx=block_idx)

        with tf.variable_scope('transition_to_classes', reuse=tf.AUTO_REUSE):
            with tf.name_scope('Transition_to_classes'):
                self.transition_layer_to_classes()

        logger.info('We have {} layers after building the graph'.format(len(self.layers)))
        logger.info('Initializing the layer names')
        self._initialize_layer_names()

        logger.info('Layer names: {}'.format(self.layer_names))

        # If the last layer is Softmax, then we take the second last layer.
        # Otherwise, we take the last layer
        self.proj = self.get_logits(self.images)

        if self.target_type == TARGET_TYPES.ONEHOT_CE or self.target_type == TARGET_TYPES.ONEHOT_MSE:
            last_layer = self.proj
        else:
            last_layer = self.get_layer(self.images, 'last')
        tf.summary.histogram('pre-final-activation', last_layer)
        self.logits = self.proj

        # Losses
        if self.loss_fn_name == 'ce' or self.target_type == TARGET_TYPES.ONEHOT_CE:
            logger.info('Using cross entropy')
            self.probs = self.states['probs']
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.centroids_ph, logits=self.proj))
        elif self.target_type == TARGET_TYPES.ONEHOT_MSE:
            logger.info('Using MSE and softmax')
            self.probs = self.states['probs']
            self.loss = tf.losses.mean_squared_error(self.centroids_ph, self.probs)
        else:
            logger.info('Using MSE')
            self.loss = tf.losses.mean_squared_error(self.centroids_ph, self.proj)

        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='L2-loss')

        tf.summary.scalar(codebook_to_loss_fn[self.target_type], self.loss)
        tf.summary.scalar('l2_loss', l2_loss)

        # optimizer and train step
        op = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.optimizer = op.minimize(
            self.loss + l2_loss * self.weight_decay)

        # Plot the gradients too
        gradients = op.compute_gradients(self.loss)

        for g in gradients:
            if 'BatchNorm' not in g[1].name and 'bn' not in g[1].name and 'gamma' not in g[1].name:
                tf.summary.histogram("%s-grad" % g[1].name, g[0])

    def get_accuracy(self, proj, centroids, groundtruths):
        if self.target_type == TARGET_TYPES.ONEHOT_CE or self.target_type == TARGET_TYPES.ONEHOT_MSE:
            predictions = np.argmax(proj, axis=1)
        else:
            predictions = []
            for row in proj:
                pred = np.argmin(np.mean(np.power(row - centroids, 2), axis=1))
                predictions.append(pred)

        return accuracy_score(predictions, groundtruths)

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        reduce_lr_epoch_3 = train_params['reduce_lr_epoch_3']
        total_start_time = time.time()

        for epoch in range(1, n_epochs + 1):
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            print('Model: {}'.format(self.model_identifier))
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2 or epoch == reduce_lr_epoch_3:
                learning_rate = learning_rate / 10
                logger.info("Decrease learning rate, new lr = %f" % learning_rate)

            logger.info("Training...")
            loss, train_acc = self.train_one_epoch(
                self.data_provider.train, batch_size, learning_rate, epoch)

            if self.should_save_logs:
                self.log_loss_accuracy(loss, train_acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                logger.info("Validation...")
                loss, test_acc = self.test(
                    self.data_provider.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, test_acc, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            logger.info("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

        total_training_time = time.time() - total_start_time
        logger.info("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate, epoch):
        num_examples = data.num_examples
        errors = []
        train_combined = []
        train_labels = []
        bar = progressbar.ProgressBar(max_value=num_examples // batch_size)

        for _ in bar(range(num_examples // batch_size)):
            batch = data.next_batch(batch_size)
            images, onehot_labels = batch
            labels = np.argmax(onehot_labels, axis=1)
            train_labels.extend(labels)

            if self.target_type == TARGET_TYPES.ONEHOT_CE:
                centroids_batch = onehot_labels
            else:
                centroids_batch = np.take(self.centroids, labels.ravel(), axis=0)

            # loss
            feed_dict = {
                self.images: images,
                self.centroids_ph: centroids_batch,
                self.learning_rate: learning_rate,
                self.is_training: True
            }

            error, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            errors.append(error)

            if self.target_type == TARGET_TYPES.ONEHOT_CE or self.target_type == TARGET_TYPES.ONEHOT_MSE:
                layer = self.sess.run(self.probs, feed_dict=feed_dict)
            else:
                layer = self.sess.run(self.proj, feed_dict=feed_dict)

            train_combined.append(layer)

        mean_loss = np.mean(errors)
        train_proj = np.concatenate((train_combined))
        accuracy = self.get_accuracy(train_proj, self.centroids, train_labels)
        feed_dict[self.train_acc_ph] = accuracy
        train_summaries = self.sess.run(self.merged, feed_dict=feed_dict)
        self.summary_writer.add_summary(train_summaries, epoch)

        return mean_loss, accuracy

    def test(self, data, batch_size):
        num_examples = data.num_examples
        errors = []
        test_combined = []
        test_labels = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            labels = np.argmax(labels, axis=1)
            test_labels.extend(labels)
            centroids_batch = np.take(self.centroids, labels.ravel(), axis=0)

            feed_dict = {
                self.images: images,
                self.centroids_ph: centroids_batch,
                self.is_training: False
            }

            error, layer = self.sess.run([self.loss, self.proj], feed_dict=feed_dict)

            if self.target_type == TARGET_TYPES.ONEHOT_CE or self.target_type == TARGET_TYPES.ONEHOT_MSE:
                layer = self.sess.run(self.probs, feed_dict=feed_dict)

            errors.append(error)
            test_combined.append(layer)

        test_proj = np.concatenate((test_combined))

        mean_loss = np.mean(errors)
        accuracy = self.get_accuracy(test_proj, self.centroids, test_labels)

        return mean_loss, accuracy
