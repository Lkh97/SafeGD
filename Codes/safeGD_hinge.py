#from comet_ml import Experiment
import io
import re
import numpy as np
import pandas as pd
import torch
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import os
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import csv
import random
from typing import Dict
import matplotlib.pyplot as plt
#from utils.utils import set_seed
import argparse
#import pdb4
# import debugpy
# # Allow other computers to attach to debugpy at this IP address and port.
# debugpy.listen(5678)
# # Pause the program until a remote debugger is attached
# print('############### port ready to listen ################')
# debugpy.wait_for_client()


def get_contexted(filename, window_size = 5):
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
                    context = '\n'.join(utterances[-(window_size+1):-1])
                    target = utterances[-1]
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
                        train_sample = '__start__'+persona+'\n'+ context + '__end__'                                          
                        label = target + '__end__'
                        target_sample = '__start__'+target 
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
def bb_tokenizer(train, target, gold_labels, tokenizer, model, mname):
  #mname = 'facebook/blenderbot_small-90M'
  #model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)

  # special_tokens_dict = {'additional_special_tokens': ['__bprs__','__eprs__']}
  # token_dict = {'sep_token': '__sep__'}

  # tokenizer.add_special_tokens(special_tokens_dict)
  # tokenizer.add_special_tokens(token_dict)

  # model.resize_token_embeddings(len(tokenizer))

  tokenized_train_context = tokenizer(train, padding='max_length', pad_to_max_length = True, truncation=True, max_length=512,return_tensors="pt",add_special_tokens= True )
  tokenized_train_labels = tokenizer(target, padding='max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= True)
  tokenized_gold_labels_tmp = tokenizer(gold_labels, padding = 'max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= False)
  tokenized_gold_labels = tokenized_gold_labels_tmp['input_ids']
  #tokenized_gold_labels[tokenized_gold_labels==0] = -100
  #print(model.get_input_embeddings()) 

  return(tokenized_train_context, tokenized_train_labels, tokenized_gold_labels)

  #tokenized_train_context, tokenized_train_labels, tokenized_gold_labels = bb_tokenizer(train, target, gold_labels, tokenizer, model, mname='facebook/blenderbot_small-90M')

  '''
  print('###################################################################################################')    
  TT = train[0]
  print(TT)
  ET = tokenizer(TT, truncation = True, padding = True)
  print("\n", ET)
  DT = tokenizer.decode(ET["input_ids"])
  print("\n", DT)
  print('###################################################################################################')   
  ###############################create dataset###################################
  '''

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

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

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


########## MODEL SETTING #############################
  main_mname= args.mname 
  mname='facebook/blenderbot_small-90M'
  tokenizer = BlenderbotSmallTokenizer.from_pretrained(main_mname)
  model = BlenderbotSmallForConditionalGeneration.from_pretrained(main_mname)

  if args.GPU3090:
      device0 = device1 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  else:
    device0 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device1 = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

  ##################### LOADER PREP ######################

  tokenized_train_context, tokenized_train_labels, tokenized_gold_labels = bb_tokenizer(train, target, gold_labels, tokenizer, model, main_mname)
  train_set = BAD(tokenized_train_context, tokenized_train_labels, tokenized_gold_labels, toxicity)

  collate_fn = Seq2SeqDataCollator(tokenizer.pad_token_id)
  dl = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)

  ############# VALIDATION DATA AND LOADER PREP #############
  if args.validation == 1:
    valData, valTarget, valLabels, valToxicity = get_contexted('valid.txt', window_size = 3)
    tokenized_valid_context, tokenized_valid_labels, tokenized__valid_gold = bb_tokenizer(valData, valTarget, valLabels, tokenizer, model, main_mname)
    val_set = BAD(tokenized_valid_context, tokenized_valid_labels, tokenized__valid_gold, toxicity)
    val_dl = DataLoader(val_set, batch_size=64, collate_fn=collate_fn)
    header = ['alpha', 'betha', 'gamma', 'epoch', 'total_loss', 'CE_loss', 'epoch_js_clean_dist', 'epoch_js_toxic_dist', 'epoch_implicit_grad' ,'train_ppl', 'validation_loss', 'validation_ppl']
  else:
    header = ['alpha', 'betha', 'gamma', 'epoch', 'total_loss', 'CE_loss', 'epoch_js_clean_dist', 'epoch_js_toxic_dist', 'epoch_implicit_grad', 'train_ppl']

  # #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
  ################################################################################  
  toxic_model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
  clean_model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
  toxic_dict = torch.load(args.toxic_expert_address, map_location=device1)
  toxic_model.load_state_dict(toxic_dict["model_state_dict"])
  clean_dict = torch.load(args.clean_expert_address, map_location=device1)
  clean_model.load_state_dict(clean_dict["model_state_dict"])
  for param in toxic_model.parameters():
      param.requires_grad = False
  for param in clean_model.parameters():
      param.requires_grad = False

  # Cpoint = torch.load(epoch_path, map_location=device0) 
  # model_clean = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
  # model_clean.resize_token_embeddings(len(tokenizer))
  # model_clean.load_state_dict(Cpoint['model_state_dict'])

  # for p1, p2 in zip(toxic_model.parameters(), clean_model.parameters()):
  #     if p1.data.ne(p2.data).sum() >0:
  #         print("toxic_model and clean_model not the same")
  #     else:
  #       print("toxic_model and clean_model the same")
  # print("Done")

  toxic_model = toxic_model.to(device1)
  clean_model = clean_model.to(device1)
  toxic_model.eval()
  clean_model.eval()
  #pdb.set_trace()

  #################################################################################

  #alpha = 1
  #betha = 1 #1e+3
  #gamma = 3 #1e+3

  #lossFile_FT3 = open ('/home/leila/compNet/loss_results/FT3_loss_sym.csv', 'w') 
  #wrt = csv.writer(lossFile_FT3)
  #wrt.writerow(["CE","js_clean","js_toxic","loss","kl_clean_toxic"])
  #'/home/leila/compNet/results/compNet/parameters/model_parameters/bb_
  # path_comet = '/home/leila/compNet/results/comet_experiment.csv'
  # header_comet = ['alpha', 'betha', 'gamma', 'comet_URL']   
  #path = args.pth_path
  #'alpha', 'betha', 'gamma', 'epoch', 'total_loss', 'CE_loss', 'KL_clean_dist', 'KL_toxic_dist',
  # param_dict = {
  # "alpha":[1,1,1,1,1,1,1e+4,1e+4,1e+4,1,1,2,10,100,1000],
  # "betha":[0,1,1,2,1,4,3,3,3,2e-3,2e-2,4e-3,3,3,3],
  # "gamma":[0,2,3,5,6,8,6,8,15,5e-3,6e-2,8e-3,1.5e-2,2e-2,3e-2]}

  # "alpha":[1,1,1,1,1,1e+4,1e+4,1e+4,1,1,2,10,100,1000],
  # "betha":[0,1,2,1,4,3,3,3,2e-3,2e-2,4e-3,3,3,3],
  # "gamma":[0,3,5,6,8,6,8,15,5e-3,6e-2,8e-3,1.5e-2,2e-2,3e-2]}

  lrate = 5e-6
  num_epoch = 5
  network_size = 87508992
  implicit_coef = (network_size*lrate)/4

  # param_dict = {
  # "alpha":[args.alpha],
  # "betha":[args.beta],
  # "gamma":[args.gamma],
  # "etha":[args.etha]}
  
  param_dict = {
  "alpha":[1,1,1,1,1],
  "betha":[1,1,2,3,3],
  "gamma":[0.5,1,10,100,0.2],
  "etha":[1,1,4,2,1]}

  # param_dict = {
  # "alpha":[1,1,1,1,1],
  # "betha":[1,1,2,3,1],
  # "gamma":[1,3,3,6,3],
  # "etha":[1,1,4,2,6]}

  # "alpha":[1,1,1,1,1],
  # "betha":[1,1,1,2,2],
  # "gamma":[1,3,2,3,2],
  # "etha":[1,1,1,2,1]}

  # "alpha":[1],
  # "betha":[2e-2],
  # "gamma":[5e-3]}

  # "alpha":[1,1,1,1,1,1,1],
  # "betha":[1e-4, 2e-4, 3e-4, 2e-4, 2e-4, 100e-4, 10e-4],
  # "gamma":[1e-4, 2e-4, 3e-4, 10e-4, 100e-4, 2e-4, 2e-4]}

  with open(args.ppl_path, 'w', encoding='UTF8') as f1:
    writer = csv.writer(f1)
    writer.writerow(header)

    # with open(path_comet, 'w', encoding='UTF8') as f2:
    #   writer_comet = csv.writer(f2)
    #   writer_comet.writerow(header_comet)

    for i,j in enumerate(param_dict["alpha"]):
        alpha = param_dict["alpha"][i]
        betha = param_dict["betha"][i]
        gamma = param_dict["gamma"][i]
        etha = param_dict["etha"][i]

        print(f"####################  alpha = {alpha} | betha = {betha} | gamma= {gamma} | #####################\n")

        optimizer = AdamW(model.parameters(), lr=lrate)
        training_step_size = num_epoch*(len(dl))
        lr_scheduler = get_scheduler("linear", 
                                    optimizer = optimizer, 
                                    num_warmup_steps = 0, 
                                    num_training_steps = training_step_size
                                    )   
          
        progress_bar = tqdm(range(training_step_size))  
        #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        model.to(device0)
        loss_step = 0
        loss_ce = []
        loss_klT = []
        loss_klC= [] 
        dist_clean_toxic = []
        checklist = []
        epoch_total_loss_vec = []
        epoch_ce_loss_vec = []
        train_ppl_epoch_vec = []
        valid_ppl_epoch_vec = []
        prev_path = ""
        # experiment = Experiment(project_name="compNet", api_key = "YIwmwbx2uNC1nA0DRHcwW56Cc", display_summary_level=0)
        # #exp_url = input("If you want to see the experiments press enter")
        # writer_comet.writerow([alpha, betha, gamma, experiment.url])

      
        # with experiment.train():
        for epoch in range(num_epoch):
          epoch_loss_total = 0
          epoch_ce_loss = 0
          accu_js_clean = 0
          accu_js_toxic = 0
          accu_implicit_grad = 0
          model.train()
          for batch in dl:
                #print(batch)
                p_no=0
                sum_pow = 0

                encoder_input_ids = batch['data_ids'].to(device0)
                encoder_att_msk = batch['data_msk'].to(device0)
                label = batch['label'].to(device0)
                decoder_input_ids = batch['target_ids'].to(device0)
                decoder_att_msk = batch['target_msk'].to(device0)

                bb_output = model(input_ids=encoder_input_ids, 
                              attention_mask = encoder_att_msk,
                              decoder_input_ids = decoder_input_ids,
                              decoder_attention_mask = decoder_att_msk,
                              labels = label)
                bb_output_logits = bb_output.logits

                encoder_input_ids = batch['data_ids'].to(device1)
                encoder_att_msk = batch['data_msk'].to(device1)
                label = batch['label'].to(device1)
                decoder_input_ids = batch['target_ids'].to(device1)
                decoder_att_msk = batch['target_msk'].to(device1)

                bbOutput_toxic = toxic_model(input_ids=encoder_input_ids, 
                              attention_mask = encoder_att_msk,
                              decoder_input_ids = decoder_input_ids,
                              decoder_attention_mask = decoder_att_msk,
                              labels = label)
                bbOutput_toxic_logits = bbOutput_toxic.logits

                bbOutput_clean = clean_model(input_ids=encoder_input_ids, 
                              attention_mask = encoder_att_msk,
                              decoder_input_ids = decoder_input_ids,
                              decoder_attention_mask = decoder_att_msk,
                              labels = label)
                bbOutput_clean_logits = bbOutput_clean.logits
                
                #logsoft2 = nn.LogSoftmax(dim=2)
                #logsoft3 = nn.LogSoftmax(dim=2)
                
                #biased_logits = logsoft2(bbOutput_toxic_logits)
                #clean_logits = logsoft3(bbOutput_clean_logits)
                
                #bbOutput_toxic_logits = bbOutput_toxic_logits.to(device0)
                #bbOutput_clean_logits = bbOutput_clean_logits.to(device0)
                
                #biased_logits = biased_logits.to(device0)
                #clean_logits = clean_logits.to(device0)

                del encoder_input_ids
                del encoder_att_msk 
                del label 
                del decoder_input_ids 
                del decoder_att_msk

                #################################################################################

                # bb_output_logSm = nn.functional.log_softmax(bb_output_logits, dim = 1)          
                # bb_output_sm = nn.functional.softmax(bb_output_logits, dim = 1)
                
                # bbOutput_toxic_logSm_tmp = nn.functional.log_softmax(bbOutput_toxic_logits, dim = 1)
                # bbOutput_toxic_sm_tmp = nn.functional.softmax(bbOutput_toxic_logits, dim = 1)
                # bbOutput_toxic_logSm = bbOutput_toxic_logSm_tmp.to(device0)          
                # bbOutput_toxic_sm = bbOutput_toxic_sm_tmp.to(device0)
                
                # bbOutput_clean_logSm_tmp = nn.functional.log_softmax(bbOutput_clean_logits, dim = 1)
                # bbOutput_clean_sm_tmp = nn.functional.softmax(bbOutput_clean_logits, dim = 1)
                # bbOutput_clean_logSm = bbOutput_clean_logSm_tmp.to(device0)
                # bbOutput_clean_sm = bbOutput_clean_sm_tmp.to(device0)          

                bb_output_dist, bbOutput_toxic_dist, bbOutput_clean_dist = nn.functional.softmax(
                bb_output_logits, dim=1).to(device0), nn.functional.softmax(
                bbOutput_toxic_logits, dim=1).to(device0), nn.functional.softmax(
                bbOutput_clean_logits, dim=1).to(device0)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture_toxic_log = torch.clamp((bbOutput_toxic_dist + bb_output_dist) / 2., 1e-7, 1).log()
                bbOutput_toxic_log = bbOutput_toxic_dist.log()
                p_mixture_toxic = (bbOutput_toxic_dist + bb_output_dist) / 2.
                JSDiv = (nn.functional.kl_div(p_mixture_toxic_log, bbOutput_toxic_dist, reduction='batchmean') +
                              nn.functional.kl_div(bbOutput_toxic_log, p_mixture_toxic, reduction='batchmean')) / 2.

                p_mixture_clean_log = torch.clamp((bbOutput_clean_dist + bb_output_dist) / 2., 1e-7, 1).log()
                bbOutput_clean_log = bbOutput_clean_dist.log()
                p_mixture_clean = (bbOutput_clean_dist + bb_output_dist) / 2.
                JSDiv_clean = (nn.functional.kl_div(p_mixture_clean_log, bbOutput_clean_dist, reduction='batchmean') +
                              nn.functional.kl_div(bbOutput_clean_log, p_mixture_clean, reduction='batchmean')) / 2

                #loss_ = bb_output.loss - ((3*JSDiv_clean-JSDiv)/3*JSDiv_clean) * bb_output.loss
                #loss_ = bb_output.loss + (JSDiv_clean/(JSDiv_clean+JSDiv)) * bb_output.loss       
                # normalized_JSDiv = torch.div(JSDiv,torch.pow(torch.tensor(10),torch.tensor(len(str(int((torch.ceil(JSDiv).data.cpu().numpy()))))).to(device0)))
                # normalized_JSDiv_clean = torch.div(JSDiv_clean,torch.pow(torch.tensor(10),torch.tensor(len(str(int((torch.ceil(JSDiv_clean).data.cpu().numpy()))))).to(device0)))
                zero = torch.zeros(JSDiv_clean.shape, requires_grad=True)
                hinge = torch.pow(torch.max(zero,JSDiv_clean-JSDiv+gamma*torch.norm(bbOutput_toxic_dist-bbOutput_clean_dist,2)),2)
                loss_ =  alpha * bb_output.loss + betha * hinge 
                
                loss_.backward(retain_graph=True)
                for param in model.parameters():
                  p_no = p_no + param.grad.view(-1).numel()
                  sum_pow = sum_pow + torch.sum(torch.pow(param.grad,2)) 
                  norm_2 = torch.sqrt(sum_pow)
                  implicit = norm_2  # torch.div(norm_2,torch.pow(torch.tensor(10),torch.tensor(len(str(int((torch.ceil(norm_2).data.cpu().numpy()))))).to(device0)))

                optimizer.zero_grad()

                loss_ =  alpha * bb_output.loss + betha * hinge + etha * implicit_coef * implicit


                #loss_ = alpha * bb_output.loss - 
                # beta * (JSDiv_clean-JSDiv)

                # experiment.log_metric("loss_total_sym", loss_, step=batch)
                # experiment.log_metric("CE_sym", bb_output.loss, step=batch)
                # experiment.log_metric("kl_toxic", JSDiv, step=batch)
                # experiment.log_metric("kl_clean", JSDiv_clean, step=batch)

                #loss_ce.append(bb_output.loss.item())
                '''
                loss_klT.append(JSDiv.item())
                loss_klC.append(JSDiv_clean.item())
                dist_clean_toxic.append(kl_clean_toxic.item())

                #if loss_step % 100 == 0:
                #wrt.writerow([bb_output.loss.item(), JSDiv_clean.item(), JSDiv.item(), loss.item(), kl_clean_toxic.item()])
                '''
                
                #################################################################################

                epoch_loss_total += loss_
                epoch_ce_loss += bb_output.loss.item()
                accu_js_clean += JSDiv.item()
                accu_js_toxic += JSDiv_clean.item()
                accu_implicit_grad += implicit
                loss_.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                loss_step += 1

          epoch_js_clean = accu_js_clean/len(dl)
          epoch_js_toxic = accu_js_toxic/len(dl)
          epoch_implicit_grad = accu_implicit_grad/len(dl)

          epoch_total_loss_vec.append(epoch_loss_total.item()/len(dl))
          epoch_ce_loss_vec.append(epoch_ce_loss/len(dl))
          train_ppl_epoch_vec.append(np.exp(epoch_ce_loss/len(dl)))

          # Calculate and save train ppl ###############################################
          epoch_path = args.pth_path+'/'+str(alpha)+'_'+str(betha)+'_'+str(gamma)+'_'+str(etha)+'_'+str(epoch)+'_implicitGrad'+'.pth'
          torch.save({"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "Loss_en": epoch_total_loss_vec[-1]}, epoch_path)
          
          ####   Validation  ###########################################################
          # Calculate and save validation ppl
          if args.validation: 
            model.eval()
            model.to(device0)
            val_accum_loss = 0
            for batch_ in val_dl:
                  encoder_input_ids_ = batch_['data_ids'].to(device0)
                  encoder_att_msk_ = batch_['data_msk'].to(device0)
                  label_ = batch_['label'].to(device0)
                  decoder_input_ids_ = batch_['target_ids'].to(device0)
                  decoder_att_msk_ = batch_['target_msk'].to(device0)

                  with torch.no_grad():
                    valid_bb_output = model(input_ids=encoder_input_ids_, 
                                    attention_mask = encoder_att_msk_,
                                    decoder_input_ids = decoder_input_ids_,
                                    decoder_attention_mask = decoder_att_msk_,
                                    labels = label_)
                  val_accum_loss += valid_bb_output.loss.item()
                  #experiment.log_metric("validation_loss", valid_bb_output.loss.item(), step=batch_)
                
            val_loss_CE = val_accum_loss/len(val_dl)
            valid_ppl_epoch_vec.append(np.exp(val_loss_CE))
            # write loss values into a file: #################################################
            data = [alpha, betha, gamma, epoch, epoch_total_loss_vec[-1], epoch_ce_loss_vec[-1], epoch_js_clean, epoch_js_toxic, epoch_implicit_grad,  train_ppl_epoch_vec[-1], val_loss_CE, valid_ppl_epoch_vec[-1]]
            writer.writerow(data)
            #header = ['alpha', 'betha', 'gamma', 'epoch', 'total_loss', 'CE_loss', 'epoch_js_clean_dist', 'epoch_js_toxic_dist', 'train_ppl', 'validation_loss', 'validation_ppl']
          else:
            data = [alpha, betha, gamma, epoch, epoch_total_loss_vec[-1], epoch_ce_loss_vec[-1], epoch_js_clean, epoch_js_toxic, epoch_implicit_grad, train_ppl_epoch_vec[-1]]
            writer.writerow(data)

          # First save all model checkpoints after each epoch to later measure toxicity:
          #############################################################################
          # if epoch ==0:
          #   prev_path = epoch_path
          #   continue
          
          # Cpoint = torch.load(prev_path, map_location=device0) 
          # model1 = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
          # #model3 = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
          # model1.resize_token_embeddings(len(tokenizer))
          # #model3.resize_token_embeddings(len(tokenizer))
          # model1.load_state_dict(Cpoint['model_state_dict'])
          # model1.to(device0)
          # #model3.to(device0)
          # for prev, curr in zip(model1.parameters(), model.parameters()):
          #     if prev.data.ne(curr.data).sum() > 0:
          #         print(f"\nprev and curr not same at epoch {epoch} at prev_path = {prev_path} and epoch_path = {epoch_path}")
          #     else: 
          #       print(f"\nprev and curr the same at epoch {epoch}")
          #     # if curr.data.ne(bb.data).sum() > 0:
          #     #    print(f"\ncurr and bb not same at epoch {epoch}")
          #     # else: 
          #     #   print(f"\ncurr and bb same at epoch {epoch}")
          #     print("b"*100)
          # prev_path = epoch_path
          # print("e"*100)
          # print(f"epoch {epoch} is done")

        #########################    RESET MODEL FOR THE NEXT PARAMETERS ####################
        torch.cuda.empty_cache()
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(main_mname)

  ########################     END of THIS OPTIMIZATION PARAMETERS ###################


if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--pth_path', type=str, default="/home/leila/NAACL2023_Experiments/hinge")
    parser.add_argument('--ppl_path', type=str, default="/home/leila/NAACL2023_Experiments/hinge/ppl_safeGD_hinge.csv") 
    parser.add_argument('--seed', type=int, default=42) 

    parser.add_argument('--alpha', type=int, default=1) 
    parser.add_argument('--beta', type=int, default=1) 
    parser.add_argument('--gamma', type=int, default=1) 
    parser.add_argument('--etha', type=int, default=1) 

    # parser.add_argument('--device0', type=str, default="cuda:0") 
    # parser.add_argument('--device1', type=str, default="cuda:1") 
    parser.add_argument('--GPU3090', type=bool, default=True) 

    parser.add_argument('--validation',action='store_true') 
    parser.add_argument('--toxic_expert_address', type=str, default="/home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_toxic/toxic_expert_0_filtered_clean_data.pth")
    parser.add_argument('--clean_expert_address', type=str, default="/home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_clean/BAD_filtered_clean_2.pth")
    parser.add_argument('--data_path', type=str, default="/home/leila/compNet_AAAI/validation/data/bot_adversarial_dialogue_datasets_with_persona")
    parser.add_argument('--filter', action='store_true') 
    parser.add_argument('--mname', type=str, default="facebook/blenderbot_small-90M")


    parser.add_argument('--pth_path', type=str, default="/home/leila/NAACL2023_Experiments/safeGD_W/checkpoints")
    parser.add_argument('--ppl_path', type=str, default="/home/leila/NAACL2023_Experiments/LWNTL/train/trainPPL_CN_filtered_newJS.csv") 
    parser.add_argument('--data_path', type=str, default="/home/leila/compNet_AAAI/validation/data/bot_adversarial_dialogue_datasets_with_persona")



    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args) 

    # Calculate ppl on toxic and clean model on validation set ############################
    '''
    del encoder_input_ids
    del encoder_att_msk 
    del label 
    del decoder_input_ids 
    del decoder_att_msk
    
    del encoder_input_ids_
    del encoder_att_msk_ 
    del label_ 
    del decoder_input_ids_ 
    del decoder_att_msk_
    
    val_toxic_lossTmp = 0
    val_clean_lossTmp = 0
    '''
    #f2.close()

    # val_toxic_lossTmp = 0
    # val_clean_lossTmp = 0

    # for batch_ in val_dl:
    #   encoder_input_ids_ = batch_['data_ids'].to(device1)
    #   encoder_att_msk_ = batch_['data_msk'].to(device1)
    #   label_ = batch_['label'].to(device1)
    #   decoder_input_ids_ = batch_['target_ids'].to(device1)
    #   decoder_att_msk_ = batch_['target_msk'].to(device1)

    #   with torch.no_grad():
    #     toxic_bb_output = toxic_model(input_ids=encoder_input_ids_, 
    #                     attention_mask = encoder_att_msk_,
    #                     decoder_input_ids = decoder_input_ids_,
    #                     decoder_attention_mask = decoder_att_msk_,
    #                     labels = label_)
    #   batch_loss = toxic_bb_output.loss.item()
    #   val_toxic_lossTmp += batch_loss 

    #   with torch.no_grad():
    #     clean_bb_output = clean_model(input_ids=encoder_input_ids_, 
    #                     attention_mask = encoder_att_msk_,
    #                     decoder_input_ids = decoder_input_ids_,
    #                     decoder_attention_mask = decoder_att_msk_,
    #                     labels = label_)
    #   batch_loss = clean_bb_output.loss.item()
    #   val_clean_lossTmp += batch_loss

    # ppl_clean = np.exp(val_clean_lossTmp/len(val_dl))
    # ppl_toxic = np.exp(val_toxic_lossTmp/len(val_dl))  

    # data = ['-', '-' , '-', '-', '-', '-', '-', '-', '-', '-', ppl_clean, ppl_toxic]
    # writer.writerow(data)
    # f1.close()

  ##############################################################################

  #lossFile_FT3.close()
  ###################### uncomment this part later:#################################
  # with open('/home/leila/compNet/loss_results/loss_Epoch_FT3_sym.csv', 'w') as f:
  #   for i,j in enumerate(loss_vec):
  #     f.write("{}.  {}\n".format(i,j))

  #torch.save(model, '/home/leila/compNet/results/tmp/save_final_sym.pth')
  #torch.save(modetate_dict(), '/home/leila/compNet/results/tmp/state_dict_final_sym.pth')

  # ce = plt.figure(1)
  # xpoints = list(range(len(loss_ce)))
  # ypoints_ce = np.array(loss_ce)
  # plt.plot(xpoints, ypoints_ce, label='CE')
  # plt.legend(loc='best')
  # ce.savefig('/home/leila/compNet/loss_results/CE_sym.png')

  # kl = plt.figure(2)

  # ypoints_klT = np.array(loss_klT)
  # ypoints_klC = np.array(loss_klC)
  # ypoints_Toxic_Clean = np.array(dist_clean_toxic)

  # plt.plot(xpoints, ypoints_klT, label = 'KL_Toxic')
  # plt.plot(xpoints, ypoints_klC, label = 'KL_Clean')
  # plt.plot(xpoints, ypoints_Toxic_Clean, label = 'Distance_Toxic_Clean')
  # plt.legend(loc='best')
  # kl.savefig('/home/leila/compNet/loss_results/KL_sym.png')
  ##################################################################################





