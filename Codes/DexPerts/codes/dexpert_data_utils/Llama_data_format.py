import os
import io
import re
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict

import torch
from datasets import load_dataset

import transformers
from transformers import (
    BlenderbotSmallForConditionalGeneration, 
    BlenderbotSmallTokenizer,
    )
import debugpy
# debugpy.listen(5678)
# print("socket waitiiiiiiiiiing for client")
# debugpy.wait_for_client()  

def get_LLAMA_contexted(data_directory, window_size, data_split):
    #contexted = []
    #targets = []
    #personas = []
    train = []
    targets = []
    gold_labels = []
    prompt=[]
    # data_directory='/home/leila/LOT_Neurips_2023/Data_models/DiaSafety/DiaSafety_dataset'
    # data_split='test.txt'
    # window_size = 3

    with io.open(os.path.join(data_directory,data_split)) as f:
            total = 0
            count = 0
            i = 0
            for line in f:
                context_tmp=[]
                context=[]
                #print(i)
                #print(line)
                #tempList = []
                #tempList = list(filter(None,re.split("\t",line)))
                #print("tempList:", tempList)
                #print("listLen:", len(tempList))
                line = {temp.split(':', 1)[0].strip(): temp.split(':', 1)[1].strip() for temp in list(filter(None,re.split("\t",line.strip())))}
                # for k, v in line.items():
                #      print(v)
                #print(line)
                text = line['text']
                #print(text)
                utterances = text.split('\\n')
                #print(utterances)
                #print(len(utterances))
                #context = utterances[-(window_size+1):-1]
                context_tmp.append(utterances[-(window_size):-1])
                #print(context_tmp)
                #print(context_tmp)
                context = [item for sublist in context_tmp for item in sublist]
                #print(context)
                target = f"{utterances[-1].strip()}</s>"
                #context = '\\n'.join([context, target])
                reformatted_segments=[]
                prompt_reformatted_segments=[]
                # for i in range (0,1,2):
                #     print(i)
                for i in range(0, len(context), 2):
                    human_text = context[i].strip()

                    # Check if there is a corresponding assistant segment before processing
                    if i + 1 < len(context):
                        assistant_text = context[i+1].strip()

                        # Apply the new template
                        reformatted_segments.append(f'<s>[INST]{human_text}[/INST]{assistant_text}</s>')
                        prompt_reformatted_segments.append(f'<s>[INST]{human_text}[/INST]{assistant_text}</s>')
                    else:
                        # Handle the case where there is no corresponding assistant segment
                        reformatted_segments.append(f'<s>[INST]{human_text}[/INST]')
                        prompt_reformatted_segments.append(f'<s>[INST]{human_text}[/INST]</s>')
                    #print(reformatted_segments)
                train.append(''.join(reformatted_segments))
                targets.append(target)
                prompt.append(''.join(prompt_reformatted_segments))

    return train, targets, prompt

# train, target, prompt = get_LLAMA_contexted(data_directory='/home/leila/LOT_Neurips_2023/Data_models/DiaSafety/DiaSafety_dataset', window_size=3, data_split='test.txt')
                
# print(len(train))
# print(len(target))
# for i in range(len(train)):
#     print(f"{train[i]}\n")
#     print(target[i])
#     print(prompt[i])

