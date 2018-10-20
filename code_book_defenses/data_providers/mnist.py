# Needed for importing MNIST dataset
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from .base_provider import ImagesDataSet, DataProvider


class MNISTDataSet(ImagesDataSet):

    def __init__(self, images, labels, n_classes, shuffle, normalization='divide_256'):
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch = False
            images, labels = self.shuffle_images_and_labels(images, labels)
        elif shuffle == 'every_epoch':
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.normalization = normalization
        self.images = self.normalize_images(images, self.normalization)
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_images_and_labels(
                self.images, self.labels)
        else:
            images, labels = self.images, self.labels
        self.epoch_images = images
        self.epoch_labels = labels

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice

class MNISTDataProvider(DataProvider):

    def __init__(self, shuffle=None, validation_set=None, validation_split=None,
                 normalization='divide_256', one_hot=True, **kwargs):
        self.one_hot = one_hot
        self._n_classes = 10

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Conform to NCHW format
        X_train = X_train[:,np.newaxis,:,:]
        X_test = X_test[:,np.newaxis,:,:]

        if self.one_hot:
            y_train = to_categorical(y_train, num_classes=10)
            y_test = to_categorical(y_test, num_classes=10)

        if validation_set is not None and validation_split is not None:
            split_idx = int(X_train.shape[0] * (1-validation_split))
            self.train = MNISTDataSet(
                images=X_train[:split_idx], labels=y_train[:split_idx],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization
            )
            self.validation = MNISTDataSet(
                images=X_train[split_idx:], labels=y_train[split_idx:],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization
            )
        else:
            self.train = MNISTDataSet(
                images=X_train, labels=y_train, n_classes=self.n_classes,
                shuffle=shuffle, normalization=normalization
            )

        self.test = MNISTDataSet(images=X_test, labels=y_test, shuffle=None,
                                 n_classes=self.n_classes, normalization=normalization)

        if validation_set and not validation_split:
            self.validation = self.test


    @property
    def data_shape(self):
        return (1, 28, 28)

    @property
    def n_classes(self):
        return self._n_classes
