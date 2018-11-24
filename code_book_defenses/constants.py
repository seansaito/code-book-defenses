class LOSSES:
    CE = 'cross_entropy'
    MSE = 'mean_squared_error'

class DATASETS:
    C10 = 'C10'
    MNIST = 'MNIST'
    FMNIST = 'FMNIST'

class ACTIVATIONS:
    SOFTMAX = 'Softmax'

class ATTACKS:
    CARLINI_WAGNER = 'carlini_wagner'
    FGSM = 'fgsm'
    MIM = 'mim'
    BIM = 'bim'
    DEEP_FOOL = 'deep_fool'
    LBFGS = 'lbfgs'
    MADRY = 'madry_et_al'
    SAITO_ROY = 'saito_roy'

class PATHS:
    ADVERSARIAL_EXAMPLES = 'adversarial_examples'
    MODEL_SAVE_PATH = 'saves'
    COMPARISON_RESULTS = 'results'

class TARGET_TYPES:
    ONEHOT_CE = 'onehot_ce'
    ONEHOT_MSE = 'onehot_mse'
    RANDOM = 'random'
