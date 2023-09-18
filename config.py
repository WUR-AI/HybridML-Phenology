
import os

import torch

"""
    Some project structure configurations
"""


# Project root dir
CONFIG_DIR = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to folder where data is stored
PATH_DATA_DIR = os.path.join(CONFIG_DIR, 'data')
os.makedirs(PATH_DATA_DIR, exist_ok=True)

# Path to folder where figures are stored
PATH_FIGURES_DIR = os.path.join(CONFIG_DIR, 'results_figures')
os.makedirs(PATH_FIGURES_DIR, exist_ok=True)

# Path to folder for storing model evaluation statistics
PATH_MODEL_EVAL_DIR = os.path.join(CONFIG_DIR, 'results_eval')
os.makedirs(PATH_MODEL_EVAL_DIR, exist_ok=True)

# Path to folder where model parameters are stored
PATH_PARAMS_DIR = os.path.join(CONFIG_DIR, 'model_parameters')
os.makedirs(PATH_PARAMS_DIR, exist_ok=True)

# Path to folder for storing saved datasets
PATH_DATASET_DIR = os.path.join(CONFIG_DIR, 'datasets', 'saved')
os.makedirs(PATH_DATASET_DIR, exist_ok=True)


"""
    Other
"""

# One seed to rule them all
SEED = 758  # Chosen by a generating a random number between 1-1000

"""
    PyTorch
"""

# TORCH_DTYPE = torch.double
TORCH_DTYPE = torch.float32

TORCH_SEED = SEED
torch.manual_seed(TORCH_SEED)