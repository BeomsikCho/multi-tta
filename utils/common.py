import random
import numpy as np
import torch
import os

def setup_deterministic(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)


