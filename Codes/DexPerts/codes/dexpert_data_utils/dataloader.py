import os
import io
import re
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    
class Seq2SeqDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch): 
        #-> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["data_ids"] for x in batch])
        attention_mask = torch.stack([x["data_msk"] for x in batch])
        decoder_input_ids = torch.stack([x["target_ids"] for x in batch])
        labels = torch.stack([x["label"] for x in batch])

        # print("BEFORE: decoder_input_ids", decoder_input_ids.shape)
        # print("BEFORE: labels", labels.shape)
        # print("BEFORE: input_ids", input_ids.shape)
        # print("BEFORE: attention_mask", attention_mask.shape)

        decoder_input_ids = trim_batch(decoder_input_ids, self.pad_token_id)
        labels = trim_batch(labels, self.pad_token_id)
        input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)
        labels = self.ignore_pad_token_for_loss(labels, self.pad_token_id)

        # print("AFTER:  decoder_input_ids", decoder_input_ids.shape)
        # print("AFTER:  labels", labels.shape)
        # print("AFTER:  input_ids", input_ids.shape)
        # print("AFTER:  attention_mask", attention_mask.shape)
        # input()     

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_input_ids":decoder_input_ids,
        }
        return batch

    def ignore_pad_token_for_loss(self, labels, pad_token_id):
        label_mask = labels.eq(pad_token_id)
        labels[label_mask.bool()] = -100
        return labels



def get_dataloader(dataset, batch_size, collate_fn):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        collate_fn=collate_fn,
                                        shuffle=False)
    return loader

