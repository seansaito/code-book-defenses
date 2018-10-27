from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import copy
import json
import logging
import os
import sys
import time

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from code_book_defenses.config import codebook_to_loss_fn, attack_name_to_params, \
    attack_name_to_class, \
    targeted_attacks, untargeted_attacks, NUMBER_OF_ATTACK_EXAMPLES, \
    attack_name_to_configurable_param, \
    attack_name_prefix, attack_to_prefix_template
from code_book_defenses.constants import PATHS, ATTACKS, LOSSES, TARGET_TYPES, DATASETS
from code_book_defenses.data_providers.utils import get_data_provider_by_name
from code_book_defenses.models.densenet import DenseNetCleverhans
from code_book_defenses.utils import get_train_params_by_name

logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

logger = logging.getLogger(__name__)


def attack_and_save(model, attack, attack_name, test_images, test_labels, dataset,
                    params, target_type, model_loss_fn, targeted,
                    target_labels=None, batch_size=100, experiment='default'):
    """
    Args:
        model: An instance of DenseNetCleverhans
        attack: An instance of cleverhans.Attack
        attack_name: (str) Name of the attack
        test_images: (np.ndarray) The images used for generating adversarial examples
        test_labels: (np.ndarray) The labels for the respective test_images
        params: (dict) parameters of the attack
        target_type: (str) Type of the targets
        model_loss_fn: (str) The loss function of the model
        targeted: (bool) Whether the attack is targeted or not
        batch_size: (int) batch size
        experiment: (str) name of the experiment (can be an UUID)

    Returns:
        if targeted, a tuple of floats representing (adversarial accuracy, success rate)
        else, a float representing adversarial accuracy
    """
    if targeted:
        assert target_labels is not None

    if not os.path.exists(PATHS.ADVERSARIAL_EXAMPLES):
        os.makedirs(PATHS.ADVERSARIAL_EXAMPLES)

    dataset_dir = os.path.join(PATHS.ADVERSARIAL_EXAMPLES, dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    attack_save_dir = os.path.join(dataset_dir, attack_name)
    if not os.path.exists(attack_save_dir):
        os.makedirs(attack_save_dir)

    targeted_prefix = "targeted" if targeted else "untargeted"

    attack_images_file_name_prefix = \
        get_attack_images_filename_prefix(attack_name, params, target_type,
                                          targeted_prefix, experiment)

    attack_images_labels_file_name = attack_images_file_name_prefix + '_labels'
    attack_images_targeted_labels_file_name = attack_images_file_name_prefix + '_targeted_labels'

    attack_images_file_name = os.path.join(attack_save_dir, attack_images_file_name_prefix)
    attack_images_labels_file_name = os.path.join(attack_save_dir, attack_images_labels_file_name)
    attack_images_targeted_labels_file_name = os.path.join(attack_save_dir,
                                                           attack_images_targeted_labels_file_name)

    all_file_names = os.listdir(attack_save_dir)
    if any(map(lambda x: x.startswith(attack_images_file_name_prefix), all_file_names)):
        logger.info('Adv examples already exist, skipping')
        return None

    logger.info('============================================================')
    if targeted:
        logger.info("TARGETED ATTACK")
    else:
        logger.info("NON-TARGETED ATTACK")
    logger.info('Generating adversarial examples using {} attack'.format(attack_name))
    logger.info('Targeting model with target type {} and loss function {}'.format(target_type,
                                                                                  model_loss_fn))
    logger.info('Generating {} adversarial examples'.format(test_images.shape[0]))
    logger.info('Parameters of attack:')
    print(params)

    adv_combined = []
    adv_x = attack.generate(model.images, **params)

    logger.info('Shape of centroids: {}'.format(model.centroids.shape))

    logger.info('Created generator tensor, now generating adversarial examples')
    for batch_idx in range((test_images.shape[0] // batch_size)):
        img_batch = test_images[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        labels_batch = test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        if targeted:
            target_labels_batch = target_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            if target_type == TARGET_TYPES.RANDOM:
                target_labels_batch = list(map(int, np.argmax(target_labels_batch, axis=1)))
                print(target_labels_batch)
                target_labels_batch = np.take(model.centroids, target_labels_batch, axis=0)
                labels_batch = list(map(int, np.argmax(labels_batch, axis=1)))
                labels_batch = np.take(model.centroids, labels_batch, axis=0)

            feed_dict = {model.images: img_batch, model.centroids_ph: labels_batch,
                         params["y_target"]: target_labels_batch}
        else:
            feed_dict = {model.images: img_batch, model.centroids_ph: labels_batch}
        adv_batch = model.sess.run(adv_x, feed_dict=feed_dict)
        adv_combined.append(adv_batch)

    adv_images = np.concatenate((adv_combined))

    logger.info('Created adversarial examples with shape: {}'.format(adv_images.shape))
    logger.info('Running prediction op on adv examples')
    adv_preds = []
    for batch_idx in range((adv_images.shape[0] // batch_size)):
        adv_batch = adv_images[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        if target_type == TARGET_TYPES.ONEHOT_CE or target_type == TARGET_TYPES.ONEHOT_MSE:
            preds = model.sess.run(model.probs, {model.images: adv_batch})
            preds = np.argmax(preds, axis=1)
        else:
            tmp = model.sess.run(model.logits, {model.images: adv_batch})
            preds = []
            for row in tmp:
                pred = np.argmin(np.mean(np.power(row - model.centroids, 2), axis=1))
                preds.append(pred)

        adv_preds.extend(list(preds))

    labels = np.argmax(test_labels, axis=1)

    adv_accuracy = accuracy_score(adv_preds, labels)
    logger.info('Test accuracy on adversarial data: {:.4f}'.format(adv_accuracy))
    logger.info('Saving adversarial images to {}'.format(attack_images_file_name))
    logger.info('Saving adversarial images labels to {}'.format(attack_images_labels_file_name))

    np.save(attack_images_file_name, adv_images)
    np.save(attack_images_labels_file_name, test_labels)

    if targeted:
        adv_labels = np.argmax(target_labels, axis=1)
        logger.info("Success rate of attacks: {:.4f}".format(accuracy_score(adv_preds, adv_labels)))
        logger.info('Saving adversarial image targets to {}'.format(
            attack_images_targeted_labels_file_name))
        np.save(attack_images_targeted_labels_file_name, adv_labels)
        return (adv_accuracy, accuracy_score(adv_preds, adv_labels))
    else:
        return adv_accuracy


def get_attack_images_filename_prefix(attack_name, params, target_type,
                                      targeted_prefix, experiment):
    prefix = attack_name_prefix.format(**{'targeted_prefix': targeted_prefix,
                                          'attack_name': attack_name,
                                          'target_type': target_type,
                                          'experiment': experiment})
    attack_configs_name = attack_to_prefix_template[attack_name].format(**params)
    attack_images_file_name_prefix = prefix + attack_configs_name

    return attack_images_file_name_prefix


def attack_main(model, configs, data_provider, quick_test, model_params, experiment='default'):
    copy_params = copy.copy(model_params)
    copy_params['experiment'] = experiment
    copy_params['should_save_centroids'] = False

    logger.info('Generating adversarial examples')
    target_type = configs['target_type']

    if target_type == TARGET_TYPES.RANDOM or target_type == TARGET_TYPES.ONEHOT_MSE:
        attack_loss_fn = LOSSES.MSE
    else:
        attack_loss_fn = LOSSES.CE

    if quick_test:
        num_samples = configs['batch_size']
    else:
        num_samples = NUMBER_OF_ATTACK_EXAMPLES

    test_images, test_labels = data_provider.test.images, data_provider.test.labels
    if configs['dataset'] == 'ImageNet' or test_images.shape[0] > num_samples:
        test_images = test_images[:num_samples]
        test_labels = test_labels[:num_samples]

    labels_as_ints = list(np.argmax(test_labels, axis=1))
    targeted_labels = []
    class_idxes = range(data_provider.n_classes)
    logger.info("Constructing target labels")
    for label in labels_as_ints:
        possibe_targets = list(set(class_idxes) - {label})
        target = np.random.choice(possibe_targets, 1)[0]
        targeted_labels.append(target)

    targeted_labels = keras.utils.to_categorical(targeted_labels,
                                                 num_classes=data_provider.n_classes)

    logger.info(
        '[NON-TARGETED] Attacking with the following attacks: {}'.format(untargeted_attacks))
    result_dir = {}
    for attack in untargeted_attacks:
        acc_list = attack_execute(attack=attack,
                                  attack_loss_fn=attack_loss_fn,
                                  configs=configs,
                                  copy_params=copy_params,
                                  data_provider=data_provider,
                                  model=model,
                                  target_type=target_type,
                                  test_images=test_images,
                                  test_labels=test_labels,
                                  targeted=False,
                                  targeted_labels=None,
                                  experiment=experiment)
        result_dir[attack] = acc_list

    logger.info('[TARGETED] Attacking with the following attacks: {}'.format(targeted_attacks))
    targeted_results_dir = {}
    for attack in targeted_attacks:
        acc_list = attack_execute(attack=attack,
                                  attack_loss_fn=attack_loss_fn,
                                  configs=configs,
                                  copy_params=copy_params,
                                  data_provider=data_provider,
                                  model=model,
                                  target_type=target_type,
                                  test_images=test_images,
                                  test_labels=test_labels,
                                  targeted=True,
                                  targeted_labels=targeted_labels,
                                  experiment=experiment)
        targeted_results_dir[attack] = acc_list

    logger.info('Summary of un-targeted attacks:')
    for (k, v) in result_dir.items():
        logger.info('{}: {}'.format(k, v))

    logger.info('Summary of targeted attacks:')
    for (k, v) in targeted_results_dir.items():
        logger.info('{}: {}'.format(k, v))


def attack_execute(attack, attack_loss_fn, configs, copy_params, data_provider, model, target_type,
                   test_images, test_labels, targeted, targeted_labels=None, experiment='default'):
    if targeted:
        assert targeted_labels is not None

    # Load a new model with a new graph
    tf.reset_default_graph()
    new_model = DenseNetCleverhans(data_provider=data_provider, **copy_params)
    new_model.centroids = model.centroids
    new_model.saver.restore(new_model.sess, model.save_path)
    logger.info('Shape of test images: {}'.format(test_images.shape))
    attack_instance = attack_name_to_class[attack](new_model, back='tf', sess=new_model.sess,
                                                   loss_type=attack_loss_fn)
    attack_params = attack_name_to_params[attack]
    grid_search_key = attack_name_to_configurable_param[attack]

    if 'batch_size' in attack_params:
        attack_params['batch_size'] = configs['batch_size']

    if not targeted:
        attack_params['batch_size'] = 5

    batch_size = configs['batch_size']
    tmp_params = copy.copy(attack_params)

    if targeted:
        # Extra setup for targeted attacks
        header_text = 'TARGETED'
        if configs['target_type'] == TARGET_TYPES.ONEHOT_CE or configs[
            'target_type'] == TARGET_TYPES.ONEHOT_MSE:
            tmp_params["y_target"] = tf.placeholder(tf.float32,
                                                    shape=[None, data_provider.n_classes],
                                                    name="{}_target_y".format(attack))
        else:
            tmp_params["y_target"] = tf.placeholder(tf.float32, shape=[None, configs['latent_dim']],
                                                    name='{}_target_y'.format(attack))

        if attack == ATTACKS.MADRY:
            tmp_params['batch_size'] = 5
            batch_size = 5

        if attack in [ATTACKS.FGSM, ATTACKS.BIM, ATTACKS.MIM]:
            attack_params['eps'] = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        header_text = 'NON-TARGETED'

    logger.info(
        '[{}] Grid searching on param {} with values {}'.format(header_text, grid_search_key,
                                                                attack_params[grid_search_key]))
    results_list = []
    for value in list(attack_params[grid_search_key]):
        tmp_params[grid_search_key] = value
        acc = attack_and_save(model=new_model,
                              attack=attack_instance,
                              attack_name=attack,
                              test_images=test_images,
                              test_labels=test_labels,
                              target_labels=targeted_labels,
                              params=tmp_params,
                              target_type=target_type,
                              model_loss_fn=codebook_to_loss_fn[target_type],
                              targeted=targeted,
                              dataset=configs['dataset'],
                              batch_size=batch_size,
                              experiment=experiment)
        results_list.append(acc)

    return results_list


if __name__ == '__main__':
    # Configure the logger
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--gpus', type=str, required=True,
                        help='GPUs to run on separated by commas (e.g. --gpu=0,1,2)')

    args = parser.parse_args()
    args = vars(args)
    model_path = args['model']
    config_path = args['config']
    experiment = args['experiment']
    gpus = args['gpus']

    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    if os.path.exists(config_path):
        with open(config_path, 'r') as fp:
            configs = json.load(fp)
    else:
        print('Check path of config file!')
        sys.exit(1)

    if os.path.exists(model_path):
        centroids_path = os.path.join(model_path, 'model.chkpt_centroids.npy')
        chkpt_path = os.path.join(model_path, 'model.chkpt')
    else:
        print('Check path of model dir!')
        sys.exit(1)

    if not configs['keep_prob']:
        if configs['dataset'] in [DATASETS.C10, DATASETS.MNIST, DATASETS.FMNIST]:
            configs['keep_prob'] = 0.8
        else:
            configs['keep_prob'] = 1.0
    if configs['model_type'] == 'DenseNet':
        configs['bc_mode'] = False
        configs['reduction'] = 1.0
    elif configs['model_type'] == 'DenseNet-BC':
        configs['bc_mode'] = True
    else:
        logger.info('Invalid model type')
        sys.exit(1)

    configs['experiment'] = experiment
    configs['gpus'] = gpus
    model_params = configs

    train_params = get_train_params_by_name(configs['dataset'])
    train_params['batch_size'] = configs['batch_size']
    train_params['latent_dim'] = configs['latent_dim']
    train_params['experiment'] = configs['experiment']
    train_params['target_type'] = configs['target_type']

    try:
        quick_test = configs['quick_test']
    except:
        quick_test = False

    data_provider = get_data_provider_by_name(configs['dataset'], train_params)

    logger.info('Loading existing graph')
    tf.reset_default_graph()
    model = DenseNetCleverhans(data_provider=data_provider, **model_params)
    logger.info('Loading centroids')
    centroids = np.load(centroids_path)
    model.centroids = centroids
    model.saver.restore(model.sess, chkpt_path)
    model.set_save_path(chkpt_path)
    logger.info('Model restored')

    logger.info('Resuming attack')
    attack_main(model=model, configs=configs, data_provider=data_provider,
                model_params=model_params,
                quick_test=quick_test, experiment=experiment)
