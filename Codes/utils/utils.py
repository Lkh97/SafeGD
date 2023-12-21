import os
import io
import re
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict

import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
