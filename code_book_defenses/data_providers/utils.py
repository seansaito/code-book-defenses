from .cifar import Cifar10DataProvider
from .fmnist import FMNISTDataProvider
from .mnist import MNISTDataProvider
from code_book_defenses.constants import DATASETS


def get_data_provider_by_name(name, train_params):
    """Return required data provider class"""
    if name == DATASETS.C10:
        return Cifar10DataProvider(**train_params)
    if name == DATASETS.MNIST:
        return MNISTDataProvider(**train_params)
    if name == DATASETS.FMNIST:
        return FMNISTDataProvider(**train_params)
    else:
        print("Sorry, data provider for `%s` dataset "
              "was not implemented yet" % name)
        exit()
