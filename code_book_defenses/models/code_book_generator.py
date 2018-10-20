from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class CodeBookGenerator():

    def __init__(self):
        self.codebook_generators = {
            'onehot_ce': self.generate_one_hot,
            'onehot_mse': self.generate_one_hot,
            'random': self.generate_random
        }

    def generate_codebooks(self, **kwargs):
        codebook_type = kwargs.get('target_type')
        generator = self.codebook_generators[codebook_type]
        return generator(**kwargs)

    def generate_random(self, **kwargs):
        n_classes = kwargs.get('n_classes')
        latent_dim = kwargs.get('latent_dim')
        randoms = np.random.uniform(low=-1.0, high=1.0, size=(n_classes, latent_dim))
        return randoms

    def generate_one_hot(self, **kwargs):
        n_classes = kwargs.get('n_classes')
        onehots = np.identity(n_classes)
        return onehots
