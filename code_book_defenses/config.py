import numpy as np

import code_book_defenses.cleverhans.attacks as attacks
from code_book_defenses.constants import LOSSES, ATTACKS, TARGET_TYPES

NUMBER_OF_ATTACK_EXAMPLES = 10000

target_types = [TARGET_TYPES.ONEHOT_CE, TARGET_TYPES.ONEHOT_MSE, TARGET_TYPES.RANDOM]

checkpoint_mapping = {
    'C10': {
        'onehot_ce': 'dir_name_of_checkpoint',
        'onehot_mse': 'dir_name_of_checkpoint',
        'random': 'dir_name_of_checkpoint'
    },
    'MNIST': {
        'onehot_ce': 'dir_name_of_checkpoint',
        'onehot_mse': 'dir_name_of_checkpoint',
        'random': 'dir_name_of_checkpoint'
    },
    'FMNIST': {
        'onehot_ce': 'dir_name_of_checkpoint',
        'onehot_mse': 'dir_name_of_checkpoint',
        'random': 'dir_name_of_checkpoint'
    }
}

attack_name_prefix = '{targeted_prefix}_{attack_name}_model-{target_type}_experiment-{experiment}'

attack_to_prefix_template ={
    ATTACKS.LBFGS: '_binary_search_steps-{binary_search_steps}_max_iterations-{max_iterations}_initial_const-{initial_const}',
    ATTACKS.FGSM: '_eps-{eps}',
    ATTACKS.BIM: '_nb_iter-{nb_iter}_eps-{eps}_eps_iter-{eps_iter}',
    ATTACKS.MIM: '_nb_iter-{nb_iter}_eps-{eps}_eps_iter-{eps_iter}',
    ATTACKS.DEEP_FOOL: '_nb_candidate-{nb_candidate}_max_iter-{max_iter}_overshoot-{overshoot}',
    ATTACKS.MADRY: '_nb_iter-{nb_iter}_eps-{eps}_eps_iter-{eps_iter}',
    ATTACKS.CARLINI_WAGNER: '_binary_search_steps-{binary_search_steps}_learning_rate-{learning_rate}_initial_const-{initial_const}_max_iterations-{max_iterations}'
}

codebook_to_loss_fn = {
    'onehot_ce': LOSSES.CE,
    'onehot_mse': LOSSES.MSE,
    'random': LOSSES.MSE
}

target_type_to_proper_name = {
    TARGET_TYPES.ONEHOT_CE: 'ONEHOT:SOFTMAX:CE',
    TARGET_TYPES.ONEHOT_MSE: 'ONEHOT:SOFTMAX:MSE',
    TARGET_TYPES.RANDOM: 'RANDOM:TANH:MSE'
}

attack_name_to_class = {
    ATTACKS.CARLINI_WAGNER: attacks.CarliniWagnerL2,
    ATTACKS.FGSM: attacks.FastGradientMethod,
    ATTACKS.MIM: attacks.MomentumIterativeMethod,
    ATTACKS.BIM: attacks.BasicIterativeMethod,
    ATTACKS.DEEP_FOOL: attacks.DeepFool,
    ATTACKS.LBFGS: attacks.LBFGS,
    ATTACKS.MADRY: attacks.MadryEtAl,
}

untargeted_attacks = [
    ATTACKS.CARLINI_WAGNER,
    ATTACKS.FGSM,
    ATTACKS.MIM,
    ATTACKS.BIM,
    ATTACKS.DEEP_FOOL, # An untargeted attack
    ATTACKS.MADRY,
]

targeted_attacks = [
    ATTACKS.CARLINI_WAGNER,
    ATTACKS.FGSM,
    ATTACKS.MIM,
    ATTACKS.BIM,
    ATTACKS.LBFGS,  # A targeted attack
    ATTACKS.MADRY,
]

attack_name_to_params = {
    ATTACKS.CARLINI_WAGNER: {
        'binary_search_steps': 5, # 5 is better
        'max_iterations': 1000, # 1000 is best
        'learning_rate': 0.01,
        'batch_size': 50,
        'initial_const': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    ATTACKS.FGSM: {
        'eps': [0.01, 0.05, 0.1, 0.15, 0.2]
    },
    ATTACKS.BIM: {
        'eps': [0.01, 0.05, 0.1, 0.15, 0.2],
        'eps_iter': 0.05,
        'nb_iter': 10 # should be 10
    },
    ATTACKS.MIM: {
        'eps': [0.01, 0.05, 0.1, 0.15, 0.2],
        'eps_iter': 0.06,
        'nb_iter': 10 # should be 10
    },
    ATTACKS.DEEP_FOOL: {
        'nb_candidate': 10,
        'overshoot': 0.02,
        'max_iter': [10, 20, 30, 40, 50] # should be 50
    },
    ATTACKS.LBFGS: {
        'batch_size': 50,
        'binary_search_steps': 5, # 5 is better
        'max_iterations': 1000, # 1000 is best
        'initial_const': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    ATTACKS.MADRY: {
        'eps': [0.02, 0.04, 0.06, 0.08, 0.1],
        'eps_iter': 0.01,
        'nb_iter': 40 # should be 40
    },
}

attack_name_to_configurable_param = {
    ATTACKS.CARLINI_WAGNER: 'initial_const',
    ATTACKS.FGSM: 'eps',
    ATTACKS.BIM: 'eps',
    ATTACKS.MIM: 'eps',
    ATTACKS.DEEP_FOOL: 'max_iter',
    ATTACKS.LBFGS: 'initial_const',
    ATTACKS.MADRY: 'eps',
}

train_params_cifar = {
    'batch_size': 50,
    'latent_dim': 128,
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,
    'reduce_lr_epoch_2': 225,
    'reduce_lr_epoch_3': 275,
    'validation_set': True,
    'validation_split': None,
    'shuffle': 'every_epoch',
    'normalization': 'by_chanels',
}

train_params_mnist = {
    'batch_size': 128,
    'validation_set': True,
    'validation_split': 0.2,
    'normalization': 'divide_256',
    'shuffle': 'every_epoch',
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,
    'reduce_lr_epoch_2': 225,
    'reduce_lr_epoch_3': 275,
}

train_params_fmnist = {
    'batch_size': 128,
    'validation_set': True,
    'validation_split': 0.2,
    'normalization': 'divide_256',
    'shuffle': 'every_epoch',
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,
    'reduce_lr_epoch_2': 225,
    'reduce_lr_epoch_3': 275,
}
