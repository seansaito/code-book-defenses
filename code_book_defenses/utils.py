from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from code_book_defenses.config import train_params_cifar, train_params_mnist, train_params_fmnist
from code_book_defenses.constants import DATASETS

def get_train_params_by_name(name):
    if name == DATASETS.C10:
        return train_params_cifar
    if name == DATASETS.MNIST:
        return train_params_mnist
    if name == DATASETS.FMNIST:
        return train_params_fmnist
