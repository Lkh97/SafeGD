from __future__ import absolute_import
import argparse
from pathlib import Path
from typing import Union, List
import os
from transformers import BartForConditionalGeneration, BartTokenizer
from IPython import embed
from training import *
from utils import preprocess, detokenize, seed_everything
import nltk.tokenize.casual
import torch
import torch.nn.functional as F
import sys
import gen_utils
import generation_logits_process
import pandas as pd
import functools
import operator
from tqdm import tqdm
import re
import html
import string
import numpy as np
import debugpy

debugpy.listen(5678)
print("socket waitiiiiiiiiiing for client")
debugpy.wait_for_client()   

# Find needle in the haystack
def find_in_seq(haystack, needle):
    cands = [None]
    for i in range(len(haystack)):
        if torch.equal(haystack[i:i+len(needle)], needle):
            cands.append(i +len(needle))
    return cands[-1]

def find_in_seq_list(haystack, needle):
    cands = [None]
    for i in range(len(haystack)):
        if haystack[i:i+len(needle)] == needle:
            cands.append(i + len(needle))
    return cands[-1]

# Jensen divergence
def js_div(a,b, reduction):
    return 0.5 * F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b,dim=-1), reduction=reduction) + \
         0.5 * F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a,dim=-1), reduction=reduction) 

"""
Main Masker Class
- Initialized with seed, base model, antiexpert, expert, and tokenizer
- mask() method will apply MaRCO masking procedure to find where antiexpert/expert disagree and mask these locations
- Given a list of text in inputs, mask() returns the masked versions of these texts, where bad tokens are replaced with <mask> token (this is BART's mask token)
"""
class Masker():
    def __init__(
        self, 
        seed = 0, 
        base_path = "facebook/bart-base", 
        antiexpert_path  = "facebook/bart-base",
        expert_path = "facebook/bart-base",
        tokenizer = "facebook/bart-base"
        ):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if not torch.cuda.is_available():
            print("No GPUs found!")
        else:
            print("Found", str(torch.cuda.device_count()), "GPUS!")

        self.seed = seed
        seed_everything(self.seed)

        # Initalize self.tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)

        # Initialize models
        self.model = BartForConditionalGeneration.from_pretrained(base_path, forced_bos_token_id = self.tokenizer.bos_token_id).to(self.device)
        self.antiexpert = BartForConditionalGeneration.from_pretrained(antiexpert_path, forced_bos_token_id = self.tokenizer.bos_token_id).to(self.device)
        self.expert = BartForConditionalGeneration.from_pretrained(expert_path, forced_bos_token_id = self.tokenizer.bos_token_id).to(self.device)
        self.model.eval()
        self.antiexpert.eval()
        self.expert.eval()
    
    """
    Takes in a list of text inputs, and a divergence threshold (thresh)
    Returns a list of the same text inputs, where some of the tokens are now replaced with <mask>
    """
    def mask(self,
        inputs,
        thresh = 1.5, # Divergence threshold to find which tokens to mask
        topk = 0, # Parameter not supported, can specify to mask the topk tokens with highest divergence rather than a random number of tokens above a threshold
        div_ba_thresh = 0.0, # Sets the threshold between base and other models if use_base_model_for_divergence=True. 
        use_base_model_for_divergence = False # If we want to use the base model's logits and compare to the expert/anti-expert. Purely for experimental purpose, this is NOT the implementation MaRCo uses and will likely perform worse. Set to False to match paper implementation.
    ):
        outputs = []

        batch = self.tokenizer(inputs, return_tensors='pt', padding = True).to(self.device)
        cur_labels = ["KL(base || exp)","KL(base || anti)","JS(exp || anti)"]
        for i in tqdm(range(len(inputs)), desc = "Identifying masks"):
            cur_seq = inputs[i]
            casual = nltk.tokenize.casual.casual_tokenize(cur_seq)
            
            # Default MaRCO implementation: use only the expert and anti-expert and find divergence of prob. distributions on each token in the input
            if not use_base_model_for_divergence:
                # ignore start and end idxs
                ignore_idxs = []

                for c_idx, c in enumerate(casual):
                    punc_only = True
                    for k in c:
                        if k not in string.punctuation:
                            punc_only = False
                            break
                    if punc_only:
                        ignore_idxs.append(c_idx)

                sum_divs_ea = []
                for j in range(len(casual)):
                    new_seq = casual.copy()
                    new_seq[j] = self.tokenizer.mask_token
                    new_full_seq = detokenize(new_seq)
                    new_full_seq = re.sub(r"\s*<mask>", "<mask>", new_full_seq)

                    new_tok = self.tokenizer(new_full_seq,return_tensors="pt").input_ids.to(self.device)
                    mask_idx = torch.nonzero(new_tok[0] == self.tokenizer.mask_token_id)

                    expert_logits = self.expert.forward(input_ids = new_tok).logits
                    antiexpert_logits = self.antiexpert.forward(input_ids = new_tok).logits
                    divs_ea = js_div(expert_logits,antiexpert_logits, reduction='none').sum(dim = -1)
                    all_divs = []
                    for cor_idx in mask_idx:
                        all_divs.append(divs_ea[0][cor_idx.item()].item())
                    sum_divs_eaz.append(np.mean(all_divs))

                # delete the ignore idxs
                mean_norm_ea = np.delete(sum_divs_ea, ignore_idxs)
                mean_norm_ea = np.array(mean_norm_ea) / mean_norm_ea.mean()
                above_thresh = np.nonzero(mean_norm_ea >= thresh)[0]

                new_casual=casual.copy()
                for a in above_thresh:
                    num_below = (np.array(ignore_idxs <= a)).sum()
                    new_casual[a + num_below] = self.tokenizer.mask_token

                outputs.append(re.sub(r"\s*<mask>", "<mask>",detokenize(new_casual)))

        if use_base_model_for_divergence:
            outputs = self.tokenizer.batch_decode(outputs,skip_special_tokens=False)
        return outputs

if __name__ == '__main__':
    # Below is a simple example using the Masker method and mask class on a couple of examples with a threshold of 1.5
    # If you want to run just the Masker from the command line, you can modify the below to take in a list of inputs, process them, and feed them into the mask method

    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh", type = float, default = 1.5, help = "Divergence threshold to identify which tokens to mask")
    
    masker = Masker(
        seed = 0, 
        base_path = "facebook/bart-base", 
        antiexpert_path = "hallisky/bart-base-toxic-antiexpert",\
        expert_path = "hallisky/bart-base-nontoxic-expert", \
        tokenizer = "facebook/bart-base"
        )
    args = parser.parse_args()

    inputs =  ["I'm surprised you got it done, seeing as you're all girls!", "You are a human"]
    
    masked_inputs = masker.mask(inputs, thresh=args.thresh)
    print("inputs:", inputs, "\nmasked inputs:", masked_inputs)
    #embed()



