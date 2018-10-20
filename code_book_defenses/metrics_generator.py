from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from code_book_defenses.constants import PATHS, TARGET_TYPES

logger = logging.getLogger(__name__)


def _accuracy_aux(preds, labels, targeted, target_labels=None):
    adv_accuracy = accuracy_score(preds, labels)
    if targeted:
        assert target_labels is not None
        success_rate = accuracy_score(preds, target_labels)
        return {'accuracy': adv_accuracy, 'success_rate': success_rate}
    return {'accuracy': adv_accuracy}


def measure_accuracy(target_type, checkpoint_dir, dataset, attack, targeted,
                     data_provider, batch_size, gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    logger.info('Generating metrics with:')
    logger.info('\tTarget Type: {}'.format(target_type))
    logger.info('\tDataset: {}'.format(dataset))
    logger.info('\tAttack: {}'.format(attack))
    logger.info('\tTargeted: {}'.format(targeted))

    model_dir = os.path.join(PATHS.MODEL_SAVE_PATH, target_type)
    model_dir = os.path.join(model_dir, checkpoint_dir)
    model_meta_path = os.path.join(model_dir, 'model.chkpt.meta')
    model_chkpt_path = os.path.join(model_dir, 'model.chkpt')
    centroids_path = os.path.join(model_dir, 'model.chkpt_centroids.npy')

    tf.reset_default_graph()
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    sess = tf.Session(config=configproto)

    # Load model
    logger.info('Restoring model from {}'.format(model_meta_path))
    new_saver = tf.train.import_meta_graph(model_meta_path)
    new_saver.restore(sess, model_chkpt_path)
    graph = tf.get_default_graph()

    # Load centroids
    logger.info('Restoring centroids from {}'.format(centroids_path))
    centroids = np.load(centroids_path)
    logger.info('Shape of centroids: {}'.format(centroids.shape))

    # Instantiate result store
    result_store = {}

    with graph.as_default():
        # Get tensors
        x_ph = graph.get_tensor_by_name('input_images:0')
        if target_type == TARGET_TYPES.RANDOM:
            suffix = 'tanh_activation'
        else:
            suffix = 'Softmax'

        try:
            pred_ops = graph.get_tensor_by_name('{}:0'.format(suffix))
        except:
            pred_ops = graph.get_tensor_by_name('{}:0'.format('softmax_activation'))

        # Get test accuracy on clean data
        data = data_provider.test
        num_examples = data.num_examples
        labels = []
        preds = []
        for i in range(num_examples // batch_size):
            images, labels_batch = data.next_batch(batch_size)
            labels_batch = np.argmax(labels_batch, axis=1)
            labels.extend(labels_batch)
            layer = sess.run(pred_ops, feed_dict={x_ph: images})

            if target_type == TARGET_TYPES.ONEHOT_CE:
                pred_batch = np.argmax(layer, axis=1)
            else:
                pred_batch = []
                for row in layer:
                    pred = np.argmin(np.mean(np.power(row - centroids, 2), axis=1))
                    pred_batch.append(pred)

            preds.extend(list(pred_batch))

        clean_test_accuracy = _accuracy_aux(preds, labels, targeted=False)
        logger.info('Test accuracy on clean data: {}'.format(clean_test_accuracy))
        result_store['clean'] = clean_test_accuracy

        # Find all files relevant to attack
        attack_files_dir = os.path.join(PATHS.ADVERSARIAL_EXAMPLES, dataset, attack)
        attack_files = os.listdir(attack_files_dir)
        if targeted:
            adv_image_files = list(
                filter(lambda x: 'untargeted' not in x and 'label' not in x, attack_files))
        else:
            adv_image_files = list(
                filter(lambda x: 'untargeted' in x and 'label' not in x, attack_files))

        for image_file in adv_image_files:
            attack_identifier = image_file[:-4]

            adv_images = np.load(os.path.join(attack_files_dir, image_file))
            labels = np.load(os.path.join(attack_files_dir, image_file[:-4] + '_labels.npy'))
            if targeted:
                target_labels = np.load(os.path.join(attack_files_dir,
                                                     image_file[:-4] + '_targeted_labels.npy'))
                if len(target_labels.shape) == 2:
                    target_labels = np.argmax(target_labels, axis=1)
            else:
                target_labels = None

            if len(labels.shape) == 2:
                labels = np.argmax(labels, axis=1)

            num_examples = adv_images.shape[0]
            preds = []
            for batch_idx in range(num_examples // batch_size):
                adv_batch = adv_images[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                layer = sess.run(pred_ops, feed_dict={x_ph: adv_batch})
                if target_type == TARGET_TYPES.ONEHOT_CE:
                    pred_batch = np.argmax(layer, axis=1)
                else:
                    pred_batch = []
                    for row in layer:
                        pred = np.argmin(np.mean(np.power(row - centroids, 2), axis=1))
                        pred_batch.append(pred)
                preds.extend(list(pred_batch))

            result = _accuracy_aux(preds=preds, labels=labels, targeted=targeted,
                                   target_labels=target_labels)
            logger.info('Attack result for {}: {}'.format(attack_identifier, result))
            result_store[attack_identifier] = result

    return result_store
