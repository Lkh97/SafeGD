import io
import re
import numpy as np
import pandas as pd
import torch
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import os
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BitsAndBytesConfig
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import csv
import random
from typing import Dict
import matplotlib.pyplot as plt
from utils.utils import set_seed
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training
#import tensorflow as tf
import debugpy
from apex import amp

# debugpy.listen(5678)
# print("socket waitiiiiiiiiiing for client")
# debugpy.wait_for_client()    

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# class Args():
#     def __init__(self):
#         self.output_dir = 'Lora_LWNTL'
#         self.model_type = 'gpt2'
#         self.model_name_or_path = "facebook/blenderbot-3B"
#         self.config_name = "facebook/blenderbot-3B"
#         self.tokenizer_name = "facebook/blenderbot-3B"
#         self.cache_dir = 'cached-fine-tuned-blenderbot-3B'
#         self.block_size = 512
#         self.do_train = True
#         self.do_eval = True
#         self.evaluate_during_training = False
#         self.per_gpu_train_batch_size = 2
#         self.per_gpu_eval_batch_size = 2
#         self.gradient_accumulation_steps = 1
#         self.learning_rate = 5e-5
#         self.weight_decay = 0.0
#         self.adam_epsilon = 1e-8
#         self.max_grad_norm = 1.0
#         self.num_train_epochs = 2
#         self.max_steps = -1
#         self.warmup_steps = 100
#         self.logging_steps = 1000
#         self.save_steps = 50000
#         self.save_total_limit = None
#         self.eval_all_checkpoints = False
#         self.no_cuda = False
#         self.overwrite_output_dir = True
#         self.overwrite_cache = True
#         self.should_continue = False
#         self.seed = 42
#         self.local_rank = -1
#         self.fp16 = True #False
#         self.fp16_opt_level = 'O1'
#         # self.clean_device=torch.device("cuda:0")
#         # self.toxic_device=torch.device("cuda:1")
#         self.pth_path="/home/leila/LOT_Neurips_2023/LOT_LoRa"
#         self.data_path="/home/leila/compNet_AAAI/validation/data/bot_adversarial_dialogue_datasets_with_persona"
#         self.filter=False

# args = Args()

# loading dataset
def get_contexted(filename, window_size = 3):
          #contexted = []
          #targets = []
          #personas = []
          train = []
          targets = []
          gold_labels = []
          toxicity = []
          #bad_dataset ={}
          with io.open(os.path.join(args.data_path,filename)) as f:
                  total = 0
                  count = 0

                  for line in f:
                      #tempList = []
                      #tempList = list(filter(None,re.split("\t",line)))
                      #print("tempList:", tempList)
                      #print("listLen:", len(tempList))
                      line = {temp.split(':', 1)[0].strip(): temp.split(':', 1)[1].strip() for temp in list(filter(None,re.split("\t",line.strip())))}
                      #for k, v in line.items():
                      #      print(k)
                      text = line['text']
                      utterances = text.split('\\n')
                      #context = utterances[-(window_size+1):-1]
                      context = '<s>'.join(utterances[-(window_size+1):-1])
                      target = utterances[-1]
                      labels = line['labels']
                      speaker_to_eval = line['speaker_to_eval']
                      persona = '<s>'.join(str(line['bot_persona']).split('\\n'))
                      speaker_to_eval = line['speaker_to_eval']

                      # if persona != 'nan':
                      #   persona = persona.split(':',1)[1].replace('\\nyour persona:','').strip()
                      
                      total += 1

                      if len(context) >= 1 and labels == "__ok__":   # Needs to have at least one context 
                          #contexted.append(''.join(utterances[-(window_size+1):-1]))
                          count += 1
                          train_sample = '<s>'+persona+'<s>'+ context                                            
                          label = target + '</s>'
                          target_sample = '<s>'+target 
                          gold_labels.append(label.strip())
                          train.append(train_sample.strip())
                          targets.append(target_sample.strip())
                          toxicity.append(labels.strip())
          #flatten = lambda l: [item for sublist in l for item in sublist]
          #contexted = flatten(contexted)             
          #return contexted, targets, personas
          return train , targets, gold_labels, toxicity

  ################################################################################

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    #Remove columns that are populated exclusively by pad_token_id
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    
class Seq2SeqDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch): 
        # -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["data_ids"] for x in batch])
        attention_mask = torch.stack([x["data_msk"] for x in batch])
        decoder_attention_mask = torch.stack([x["target_msk"] for x in batch])
        decoder_input_ids = torch.stack([x["target_ids"] for x in batch])
        labels = torch.stack([x["label"] for x in batch])

        # print("BEFORE: decoder_input_ids", decoder_input_ids.shape)
        # print("BEFORE: labels", labels.shape)
        # print("BEFORE: input_ids", input_ids.shape)
        # print("BEFORE: attention_mask", attention_mask.shape)

        decoder_input_ids, decoder_attention_mask = trim_batch(decoder_input_ids, self.pad_token_id, attention_mask = decoder_attention_mask)
        labels = trim_batch(labels, self.pad_token_id)
        input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)
        labels = self.ignore_pad_token_for_loss(labels, self.pad_token_id)

        # print("AFTER:  decoder_input_ids", decoder_input_ids.shape)
        # print("AFTER:  labels", labels.shape)
        # print("AFTER:  input_ids", input_ids.shape)
        # print("AFTER:  attention_mask", attention_mask.shape)
        # input()     

        batch = {
            "data_ids": input_ids,
            "data_msk": attention_mask,
            "label": labels,
            "target_ids": decoder_input_ids,
            "target_msk": decoder_attention_mask
        }
        return batch

    def ignore_pad_token_for_loss(self, labels, pad_token_id):
        label_mask = labels.eq(pad_token_id)
        labels[label_mask.bool()] = -100
        return labels

################################################################################

#train_tmp, target_tmp, gold_labels_tmp, toxicity_tmp = get_contexted('train.txt', window_size = 3)
'''
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
print(train[9])
print(target[9])
print(gold_labels[9])
print(toxicity[9])
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
'''
################################################################################

'''
ok_list_idx = [i for i in range(len(toxicity_tmp)) if toxicity_tmp[i] =='__ok__']
notok_list_idx = [i for i in range(len(toxicity_tmp)) if toxicity_tmp[i] =='__notok__']

ok_train_idx = []
notok_train_idx = []

instance_no = 7000
ok_portion = 0.64 

n = round(ok_portion*instance_no)
m = round((1-ok_portion)*instance_no)

for i in range(n):
ok_train_idx.append(random.choice(ok_list_idx))

for j in range(m):
notok_train_idx.append(random.choice(notok_list_idx))

train = [train_tmp[i] for i in ok_train_idx] + [train_tmp[i] for i in notok_train_idx]
target = [target_tmp[i] for i in ok_train_idx] + [target_tmp[i] for i in notok_train_idx] 
gold_labels = [gold_labels_tmp[i] for i in ok_train_idx] + [gold_labels_tmp[i] for i in notok_train_idx]
toxicity = [toxicity_tmp[i] for i in ok_train_idx] + [toxicity_tmp[i] for i in notok_train_idx]

print(len(train))
print(len(target))
print(len(gold_labels))
print(len(toxicity))
'''

################################################################################
def bb_tokenizer(train, target, gold_labels, tokenizer, model, mname='facebook/blenderbot_small-90M'):

    tokenized_train_context = tokenizer(train, padding='max_length', pad_to_max_length = True, truncation=True, max_length=512,return_tensors="pt",add_special_tokens= True )
    tokenized_train_labels = tokenizer(target, padding='max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= True)
    tokenized_gold_labels_tmp = tokenizer(gold_labels, padding = 'max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= False)
    tokenized_gold_labels = tokenized_gold_labels_tmp['input_ids']
    #tokenized_gold_labels[tokenized_gold_labels==0] = -100
    #print(model.get_input_embeddings()) 

    return(tokenized_train_context, tokenized_train_labels, tokenized_gold_labels)

class BAD(Dataset):
    def __init__(self, data, target, label, toxicity):
        self.data = data
        self.target = target
        self.label = label
        self.toxicity = toxicity

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        item = {}
        item['data_ids'] = torch.tensor(self.data['input_ids'][idx])
        item['data_msk'] = torch.tensor(self.data['attention_mask'][idx])
        item['target_ids'] = torch.tensor(self.target['input_ids'][idx])
        item['target_msk'] = torch.tensor(self.target['attention_mask'][idx])
        item['label'] = torch.tensor(self.label[idx])
        item['toxicity'] = self.toxicity[idx]
        return item

        # label = self.label[idx]
        # target = self.target[idx]
        # data = self.data[idx]
        # sample = {"Context": data, "Target": target, "Label":label}
        # return data, target, label, sample 
    ####### FILTER CANNED SENTENCES FROM TRAIN DATA #####

def main(args):

    device = 'cuda'
    model_name_or_path = "facebook/blenderbot-3B"
    tokenizer_name_or_path = "facebook/blenderbot-3B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    checkpoint_name = "clean_t.pt"
    text_column = "sentence"
    label_column = "text_label"
    max_length = 512
    lr = 5e-6
    num_epochs = 2
    batch_size = 1


    # creating model
    # peft_config = AdaLoraConfig(
    #     init_r=12,
    #     target_r=8,
    #     beta1=0.85,
    #     beta2=0.85,
    #     tinit=200,
    #     tfinal=1000,
    #     deltaT=10,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     task_type=TaskType.SEQ_2_SEQ_LM,
    #     inference_mode=False,
    # )

    # peft_config = LoraConfig(
    #     r=8, 
    #     lora_alpha=32, 
    #     target_modules= ["query", "value"], #["query_key_value"], 
    #     lora_dropout=0.05, 
    #     bias="none", 
    #     task_type="SEQ_2_SEQ_LM"
    # )


    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, 
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32, 
    lora_dropout=0.1)

    model = BlenderbotForConditionalGeneration.from_pretrained(model_name_or_path,quantization_config=bnb_config) #, device_map={"":0})
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name_or_path)
    
    special_token_dict = tokenizer.special_tokens_map
    tokenizer.add_special_tokens(special_token_dict)
    model.resize_token_embeddings(tokenizer.vocab_size)

    #BlenderbotForConditionalGeneration.from_pretrained(model_name_or_path,quantization_config=bnb_config, device_map={"":0})
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # cpus = tf.config.list_physical_devices('CPU')
    # tf.config.set_visible_devices(cpus[0], 'CPU')
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    if args.filter:
        train_tmp, target_tmp, gold_labels_tmp, toxicity_tmp = get_contexted('train.txt', 3)
        #valData, valTarget, valLabels, valToxicity, total, count = get_valid_data('valid.txt', window_size = 3)
        remove_term = 'Hey do you want to talk about something else? How about we talk about'
        remove_list_idx = [i for i in range(len(target_tmp)) if remove_term in target_tmp[i] or remove_term in train_tmp[i]]

        train = [ele for idx, ele in enumerate(train_tmp) if idx not in remove_list_idx]
        target = [ele for idx, ele in enumerate(target_tmp) if idx not in remove_list_idx]
        gold_labels = [ele for idx, ele in enumerate(gold_labels_tmp) if idx not in remove_list_idx]
        toxicity = [ele for idx, ele in enumerate(toxicity_tmp) if idx not in remove_list_idx]

        del train_tmp
        del target_tmp
        del gold_labels_tmp
        del toxicity_tmp

    else:
        train, target, gold_labels, toxicity = get_contexted('train.txt', 3)

    tokenized_train_context, tokenized_train_labels, tokenized_gold_labels = bb_tokenizer(train, target, gold_labels, tokenizer, model, mname='facebook/blenderbot_small-90M')

    train_set = BAD(tokenized_train_context, tokenized_train_labels, tokenized_gold_labels, toxicity)

    dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(dl) * num_epochs),
    )    
    #model.base_model.peft_config.total_step = len(dl) * num_epochs
    
    #progress_bar = tqdm(range(model.base_model.peft_config.total_step)) 
    model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    global_step = 0
    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(tqdm(dl)):
            #print(batch)

            encoder_input_ids = batch['data_ids'].to(device)
            encoder_att_msk = batch['data_msk'].to(device)
            label = batch['label'].to(device)
            decoder_input_ids = batch['target_ids'].to(device)
            decoder_att_msk = batch['target_msk'].to(device)

            output = model(input_ids=encoder_input_ids,
                            attention_mask = encoder_att_msk,
                            decoder_input_ids = decoder_input_ids,
                            decoder_attention_mask = decoder_att_msk,
                            labels = label)
            loss = output.loss
            epoch_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            model.base_model.update_and_allocate(global_step)
            optimizer.zero_grad()
            #progress_bar.update(1)
            global_step += 1
            loss_itr.append(loss.item())

        epoch_path = args.pth_path+'/'+'epoch'+'_'+str(epoch)+'_clean_t'+'.pt'
        model.save_pretrained(epoch_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--pth_path', type=str, default="/home/leila/LOT_Neurips_2023/LOT_LoRa")
    # parser.add_argument('--ppl_path', type=str, default="/home/leila/compNet/CN_filtered_newJS/metrics/train/ppl/trainPPL_CN_filtered_newJS.csv") 
    # parser.add_argument('--seed', type=int, default=42) 
    # parser.add_argument('--validation', type=bool, default=False) 
    # parser.add_argument('--toxic_expert_address', type=str, default="/home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_toxic/toxic_expert_0_filtered_clean_data.pth")
    # parser.add_argument('--clean_expert_address', type=str, default="/home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_clean/BAD_filtered_clean_2.pth")
    parser.add_argument('--data_path', type=str, default="/home/leila/compNet_AAAI/validation/data/bot_adversarial_dialogue_datasets_with_persona")
    parser.add_argument('--filter', type=bool, default=False) 

    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args) 