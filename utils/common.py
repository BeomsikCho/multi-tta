import random
import numpy as np
import torch
import os

import argparse
import yaml
from collections import defaultdict

def setup_deterministic(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_cfgs():
    parser = argparse.ArgumentParser(description="the methods to train")
    parser.add_argument('--config', type=str)
    parser.add_argument('--mode', type=str, choices=[None, 'train', 'eval', 'all'], default=None, help="Operation mode of the model.")
    args = parser.parse_args()

    with open(args.config) as file:
        cfgs = yaml.load(file, Loader = yaml.FullLoader)
        cfgs = defaultdict(lambda: None, cfgs)
        cfgs['config'] = args.config
        cfgs['mode'] = args.mode
    return cfgs