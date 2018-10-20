from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import os
import sys
import time

import code_book_defenses.attack_controller as attack_controller
from code_book_defenses.data_providers.utils import get_data_provider_by_name
from code_book_defenses.models.densenet import DenseNetCleverhans
from code_book_defenses.utils import get_train_params_by_name

if __name__ == '__main__':
    # Configure the logger
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--gpus', type=str, required=True,
                        help='GPUs to run on separated by commas (e.g. --gpu=0,1,2)')
    parser.add_argument('--experiment', type=str, required=True, default='default',
                        help='Experiment name')

    args = parser.parse_args()
    args = vars(args)
    config_path = args['config']
    gpus = args['gpus']
    experiment = args['experiment']

    if os.path.exists(config_path):
        with open(config_path, 'r') as fp:
            configs = json.load(fp)
    else:
        print('Check path of config file!')
        sys.exit(1)

    configs['experiment'] = experiment
    configs['gpus'] = gpus

    if not configs['keep_prob']:
        if configs['dataset'] in ['C10', 'MNIST', 'FMNIST"']:
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

    model_params = configs

    # some default params dataset/architecture related
    train_params = get_train_params_by_name(configs['dataset'])
    train_params['batch_size'] = configs['batch_size']
    train_params['latent_dim'] = configs['latent_dim']
    train_params['experiment'] = configs['experiment']
    train_params['target_type'] = configs['target_type']
    try:
        quick_test = configs['quick_test']
    except:
        quick_test = False

    if quick_test:
        logger.info('Running quick test with 3 epochs')
        train_params['n_epochs'] = 3
    else:
        train_params['n_epochs'] = configs['n_epochs']

    logger.info("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    logger.info("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    logger.info("Prepare training data...")
    data_provider = get_data_provider_by_name(configs['dataset'], train_params)

    logger.info("Initialize the model..")
    model = DenseNetCleverhans(data_provider=data_provider, **model_params)

    if configs['train']:
        logger.info("Data provider train images: {}".format(data_provider.train.num_examples))
        model.train_all_epochs(train_params)
    if configs['test']:
        if not configs['train']:
            model.load_model()
        logger.info("Data provider test images: {}".format(data_provider.test.num_examples))
        logger.info("Testing...")
        loss, accuracy = model.test(data_provider.test, batch_size=200)
        if type(accuracy) is list:
            accuracy = accuracy[0]
        logger.info("mean loss: %f\n mean accuracy: %f" % (
            loss, accuracy))

    # Generate attacks and test
    attack_controller.attack_main(model=model, configs=configs, data_provider=data_provider,
                                  quick_test=quick_test,
                                  model_params=model_params, experiment=experiment)

    end = time.time()
    logger.info('Training and testing took {:.4f} seconds'.format(end - start))
