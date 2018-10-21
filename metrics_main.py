from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import os
import sys

from code_book_defenses.config import targeted_attacks, untargeted_attacks, checkpoint_mapping
from code_book_defenses.config import train_params_cifar, target_types, train_params_mnist, \
    train_params_fmnist
from code_book_defenses.constants import PATHS, DATASETS
from code_book_defenses.data_providers.cifar import Cifar10DataProvider
from code_book_defenses.data_providers.fmnist import FMNISTDataProvider
from code_book_defenses.data_providers.mnist import MNISTDataProvider
from code_book_defenses.metrics_generator import measure_accuracy

if __name__ == '__main__':
    # Configure the logger
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset (C10, MNIST, FMNIST)')
    parser.add_argument('--gpus', type=str, required=True,
                        help='GPUs to run on separated by commas (e.g. --gpu=0,1,2)')

    args = parser.parse_args()
    args = vars(args)
    dataset = args['dataset']
    gpus = args['gpus']

    logger.info('Gathering metrics')
    untargeted_attack_results = {}
    targeted_attack_results = {}
    for attack in untargeted_attacks:
        untargeted_attack_results[attack] = {}

    for attack in targeted_attacks:
        targeted_attack_results[attack] = {}

    target_type_to_checkpoint_dict = checkpoint_mapping[dataset]
    if dataset == DATASETS.C10:
        data_provider = Cifar10DataProvider(**train_params_cifar)
    elif dataset == DATASETS.MNIST:
        data_provider = MNISTDataProvider(**train_params_mnist)
    elif dataset == DATASETS.FMNIST:
        data_provider = FMNISTDataProvider(**train_params_fmnist)
    else:
        logger.error('Invalid dataset')
        sys.exit(1)

    for attack in untargeted_attacks:
        for target_type in target_types:
            checkpoint_dir = target_type_to_checkpoint_dict[target_type]
            temp_res = measure_accuracy(
                target_type=target_type,
                checkpoint_dir=checkpoint_dir,
                dataset=dataset,
                attack=attack,
                targeted=False,
                data_provider=data_provider,
                batch_size=10,
                gpus=gpus
            )
            untargeted_attack_results[attack][target_type] = temp_res

    for attack in targeted_attacks:
        for target_type in target_types:
            checkpoint_dir = target_type_to_checkpoint_dict[target_type]
            temp_res = measure_accuracy(
                target_type=target_type,
                checkpoint_dir=checkpoint_dir,
                dataset=dataset,
                attack=attack,
                targeted=True,
                data_provider=data_provider,
                batch_size=10,
                gpus=gpus
            )
            targeted_attack_results[attack][target_type] = temp_res

    if not os.path.exists(PATHS.COMPARISON_RESULTS):
        os.makedirs(PATHS.COMPARISON_RESULTS)

    untargeted_results_path = os.path.join(PATHS.COMPARISON_RESULTS,
                                           '{}_untargeted_results.json'.format(dataset))
    targeted_results_path = os.path.join('{}_targeted_results.json'.format(dataset))

    with open(untargeted_results_path, 'w') as fp:
        json.dump(untargeted_attack_results, fp)

    with open(targeted_results_path, 'w') as fp:
        json.dump(targeted_attack_results, fp)
