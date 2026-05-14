"""Hyperparameters for feature selection experiments"""

LEARNING_RATE = {
    'mice': 1e-3,
    'mnist': 1e-3,
    'fashion': 1e-3,
    'isolet': 1e-3,
    'coil': 1e-3,
    'activity': 1e-3,
    'california_housing': 1e-3,
    'madelon': 1e-3,
    'har70': 1e-3,
    'multispectral': 1e-3,
}

BATCH = {
    'mice': 64,
    'mnist': 64,
    'fashion': 64,
    'isolet': 64,
    'coil': 64,
    'activity': 64,
    'california_housing': 64,
    'madelon': 64,
    'har70': 64,
    'multispectral': 64,
}

EPOCHS = {
    'mice': 400,
    'mnist': 100,
    'fashion': 200,
    'isolet': 400,
    'coil': 1000,
    'activity': 200,
    'california_housing': 200,
    'madelon': 500,
    'har70': 6,
    'multispectral': 20,
}

EPOCHS_FIT = {
    'mice': 200,
    'mnist': 50,
    'fashion': 100,
    'isolet': 200,
    'coil': 500,
    'activity': 100,
    'california_housing': 100,
    'madelon': 250,
    'har70': 3,
    'multispectral': 10,
}

DEEP_LAYERS = {
    'mnist': '261',
    'fashion': '261', 
    'mice': '25', 
    'coil': '133', 
    'isolet': '205', 
    'activity': '187', 
    'california_housing': '3',
    'madelon': '166',
    'har70': '35',
    'multispectral': '5',
}

SIGMA = {
    'mice': 1.5,
    'mnist': 1.5,
    'fashion': 1.5,
    'isolet': 1.5,
    'coil': 1.5,
    'activity': 1.5,
    'california_housing': 1.5,
    'madelon': 1.5,
    'har70': 1.5,
    'multispectral': 1.5,
}