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
from transformers import (
    BlenderbotSmallForConditionalGeneration, 
    BlenderbotSmallTokenizer,
    )
import debugpy
# debugpy.listen(5678)
# print("socket waitiiiiiiiiiing for client")
# debugpy.wait_for_client()    

def get_contexted(data_directory, window_size, data_split):
    #contexted = []
    #targets = []
    #personas = []
    train = []
    targets = []
    gold_labels = []
    toxicity = []
    #bad_dataset ={}
    with io.open(os.path.join(data_directory,data_split)) as f:
            total = 0
            count = 0
            i = 0
            for line in f:
                #print(i)
                #print(line)
                #tempList = []
                #tempList = list(filter(None,re.split("\t",line)))
                #print("tempList:", tempList)
                #print("listLen:", len(tempList))
                line = {temp.split(':', 1)[0].strip(): temp.split(':', 1)[1].strip() for temp in list(filter(None,re.split("\t",line.strip())))}
                #for k, v in line.items():
                #      print(k)
                #print(line)
                text = line['text']
                #print(text)
                utterances = text.split('\\n')
                #context = utterances[-(window_size+1):-1]
                context = '\n'.join(utterances[-(window_size+1):-1])
                #print(context)
                target = utterances[-1]
                #print(target)
                labels = line['labels']
                speaker_to_eval = line['speaker_to_eval']
                persona = '\n'.join(str(line['bot_persona']).split('\\n'))
                speaker_to_eval = line['speaker_to_eval']

                # if persona != 'nan':
                #   persona = persona.split(':',1)[1].replace('\\nyour persona:','').strip()
                
                total += 1

                if len(context) >= 1:   # Needs to have at least one context 
                    #contexted.append(''.join(utterances[-(window_size+1):-1]))
                    count += 1
                    train_sample = '\n'+persona+'\n'+ context                                            
                    label = target + '__end__'
                    target_sample = '__start__'+target 
                    gold_labels.append(label.strip())
                    train.append(train_sample.strip())
                    targets.append(target_sample.strip())
                    toxicity.append(labels.strip())
    #flatten = lambda l: [item for sublist in l for item in sublist]
    #contexted = flatten(contexted)             
    #return contexted, targets, personas
                i=i+1
    return train , targets, gold_labels, toxicity #, total, count

def remove_canned(train_tmp,target_tmp,gold_labels_tmp, toxicity_tmp):
    remove_term = 'Hey do you want to talk about something else? How about we talk about'
    remove_list_idx = [i for i in range(len(target_tmp)) if remove_term in target_tmp[i] or remove_term in train_tmp[i]]

    train = [ele for idx, ele in enumerate(train_tmp) if idx not in remove_list_idx]
    target = [ele for idx, ele in enumerate(target_tmp) if idx not in remove_list_idx]
    gold_labels = [ele for idx, ele in enumerate(gold_labels_tmp) if idx not in remove_list_idx]
    toxicity = [ele for idx, ele in enumerate(toxicity_tmp) if idx not in remove_list_idx]
    #train, target, gold_labels = get_contexted(f"{args.data_folder}/{args.split}.txt", args.window_size)
    return train, target, gold_labels, toxicity


def get_contexted(data_directory, window_size, data_split, st, et):
    #contexted = []
    #targets = []
    #personas = []
    train = []
    targets = []
    gold_labels = []
    toxicity = []
    #bad_dataset ={}
    with io.open(os.path.join(data_directory,data_split)) as f:
            total = 0
            count = 0
            i = 0
            for line in f:
                #print(i)
                #print(line)
                #tempList = []
                #tempList = list(filter(None,re.split("\t",line)))
                #print("tempList:", tempList)
                #print("listLen:", len(tempList))
                line = {temp.split(':', 1)[0].strip(): temp.split(':', 1)[1].strip() for temp in list(filter(None,re.split("\t",line.strip())))}
                #for k, v in line.items():
                #      print(k)
                #print(line)
                text = line['text']
                #print(text)
                utterances = text.split('\\n')
                #context = utterances[-(window_size+1):-1]
                context = '\n'.join(utterances[-(window_size+1):-1])
                #print(context)
                target = utterances[-1]
                #print(target)
                labels = line['labels']
                speaker_to_eval = line['speaker_to_eval']
                persona = '\n'.join(str(line['bot_persona']).split('\\n'))
                speaker_to_eval = line['speaker_to_eval']

                # if persona != 'nan':
                #   persona = persona.split(':',1)[1].replace('\\nyour persona:','').strip()
                
                total += 1

                if len(context) >= 1:   # Needs to have at least one context 
                    #contexted.append(''.join(utterances[-(window_size+1):-1]))
                    count += 1
                    train_sample = st + persona+'\n' + context + et                                            
                    label = target + et
                    target_sample = st + target 
                    gold_labels.append(label.strip())
                    train.append(train_sample.strip())
                    targets.append(target_sample.strip())
                    toxicity.append(labels.strip())
    #flatten = lambda l: [item for sublist in l for item in sublist]
    #contexted = flatten(contexted)             
    #return contexted, targets, personas
                i=i+1
    return train , targets, gold_labels, toxicity #, total, count

def remove_canned(train_tmp,target_tmp,gold_labels_tmp, toxicity_tmp):
    remove_term = 'Hey do you want to talk about something else? How about we talk about'
    remove_list_idx = [i for i in range(len(target_tmp)) if remove_term in target_tmp[i] or remove_term in train_tmp[i]]

    train = [ele for idx, ele in enumerate(train_tmp) if idx not in remove_list_idx]
    target = [ele for idx, ele in enumerate(target_tmp) if idx not in remove_list_idx]
    gold_labels = [ele for idx, ele in enumerate(gold_labels_tmp) if idx not in remove_list_idx]
    toxicity = [ele for idx, ele in enumerate(toxicity_tmp) if idx not in remove_list_idx]
    #train, target, gold_labels = get_contexted(f"{args.data_folder}/{args.split}.txt", args.window_size)
    return train, target,gold_labels, toxicity

def get_contexted_for_generation(data_directory, window_size, bs, es):
    train = []
    targets = []
    gold_labels = []
    with io.open(data_directory) as f:
        total = 0
        count = 0
        for line in f:
            line = {temp.split(':', 1)[0].strip(): temp.split(':', 1)[1].strip() for temp in list(filter(None, re.split("\t",line.strip())))}
            text = line['text']
            utterances = text.split('\\n')
            if len(utterances)%2!=0:
                continue 
            #context = utterances[-(window_size+1):-1]
            context = '\n'.join(utterances[-(window_size+1):-1])
            target = utterances[-1]
            labels = line['labels']
            speaker_to_eval = line['speaker_to_eval']
            persona = '\n'.join(str(line['bot_persona']).split('\\n'))
            speaker_to_eval = line['speaker_to_eval']

                # if persona != 'nan':
                #   persona = persona.split(':',1)[1].replace('\\nyour persona:','').strip()

            total += 1

            #if labels == "__notok__":
            #if len(context) >= 1:   # Needs to have at least one context 
                #contexted.append(''.join(utterances[-(window_size+1):-1]))
            count += 1
            train_sample = bs + persona + '\n' + context + es
            label = target + es
            target_sample = bs + target
            gold_labels.append(label.strip())
            train.append(train_sample.strip())
            targets.append(target_sample.strip())
    return train , targets, gold_labels



class BAD(Dataset):
  def __init__(self, data, target, label):
    self.data = data
    self.target = target
    self.label = label

  def __len__(self):
    return len(self.label)

  def __getitem__(self, idx):
    item = {}
    item['data_ids'] = torch.LongTensor(self.data['input_ids'][idx])
    item['data_msk'] = torch.LongTensor(self.data['attention_mask'][idx])
    item['target_ids'] = torch.LongTensor(self.target['input_ids'][idx])
    item['target_msk'] = torch.LongTensor(self.target['attention_mask'][idx])
    item['label'] = torch.LongTensor(self.label[idx])
    return item
  
def bb_get_contexted(data_directory, window_size, data_split):
    #contexted = []
    #targets = []
    #personas = []
    train = []
    targets = []
    gold_labels = []
    toxicity = []
    #bad_dataset ={}
    with io.open(os.path.join(data_directory,data_split)) as f:
            total = 0
            count = 0
            i = 0
            for line in f:
                #print(i)
                #print(line)
                #tempList = []
                #tempList = list(filter(None,re.split("\t",line)))
                #print("tempList:", tempList)
                #print("listLen:", len(tempList))
                line = {temp.split(':', 1)[0].strip(): temp.split(':', 1)[1].strip() for temp in list(filter(None,re.split("\t",line.strip())))}
                #for k, v in line.items():
                #      print(k)
                #print(line)
                text = line['text']
                #print(text)
                utterances = text.split('\\n')
                #context = utterances[-(window_size+1):-1]
                context = '\n'.join(utterances[-(window_size+1):-1])
                #print(context)
                target = utterances[-1]
                #print(target)
                labels = line['labels']
                speaker_to_eval = line['speaker_to_eval']
                persona = '\n'.join(str(line['bot_persona']).split('\\n'))
                speaker_to_eval = line['speaker_to_eval']

                # if persona != 'nan':
                #   persona = persona.split(':',1)[1].replace('\\nyour persona:','').strip()
                
                total += 1

                if len(context) >= 1:   # Needs to have at least one context 
                    #contexted.append(''.join(utterances[-(window_size+1):-1]))
                    count += 1
                    train_sample = '<s>'+persona+'\n'+ context+'</s>'                                            
                    label = target + '</s>'
                    target_sample = '<s>'+target 
                    gold_labels.append(label.strip())
                    train.append(train_sample.strip())
                    targets.append(target_sample.strip())
                    toxicity.append(labels.strip())
    #flatten = lambda l: [item for sublist in l for item in sublist]
    #contexted = flatten(contexted)             
    #return contexted, targets, personas
                i=i+1
    return train , targets, gold_labels, toxicity #, total, count

# train, target, label, toxicity = bb_get_contexted(data_directory='/home/leila/LOT_Neurips_2023/Data_models/DiaSafety/DiaSafety_dataset', window_size=3, data_split='test.txt')
# train1, target1, label1, toxicity1 = get_contexted(data_directory='/home/leila/LOT_Neurips_2023/Data_models/DiaSafety/DiaSafety_dataset', window_size=3, data_split='test.txt')

# print(len(train))
# print(len(train1))
# for i in range(len(train)):
#     print(f"{train[i]}\n")
#     print(target[i])
#     print(label[i])

#     print(f"{train1[i]}\n")
#     print(target1[i])
#     print(label1[i])