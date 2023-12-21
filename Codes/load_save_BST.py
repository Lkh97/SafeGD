
import os
import io
from csv import writer
import re
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import debugpy
import torch.nn.functional as F
#from typing import Dict
import torch.nn as nn
import timeit
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils import load_generations_toDf

import transformers
from transformers import (
    BlenderbotSmallForConditionalGeneration,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
    BlenderbotSmallTokenizer, 
    BlenderbotModel,
    top_k_top_p_filtering
)
#from typer import Exit

from data_utils.dataloader import Seq2SeqDataCollator, get_dataloader
from data_utils.bad import BAD, get_contexted, get_contexted_for_generation, bb_get_contexted
from utils.utils import set_seed


mname = 'facebook/blenderbot_small-90M'
model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
epoch_path = '/home/leila/LOT_Neurips_2023/Data_models/final_models/BST_90M.pth'
torch.save({"model_state_dict": model.state_dict()}, epoch_path) 