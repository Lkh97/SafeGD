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
#from typing import Dict
import torch.nn as nn
import timeit
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import transformers
from transformers import (
    BlenderbotSmallForConditionalGeneration, 
    BlenderbotSmallTokenizer, 
    top_k_top_p_filtering
)
#from typer import Exit

from data_utils.dataloader import Seq2SeqDataCollator, get_dataloader
from data_utils.bad import BAD, get_contexted, get_contexted_for_generation
#from utils.utils import set_seed

# import debugpy
# debugpy.listen(5678)
# print("socket waitiiiiiiiiiing for client")
# debugpy.wait_for_client()    

def model_perplexity(prompt, generations, mname):
    device='cuda'
    # device1 = 'cpu'
    perplexities = []
    ct = 0
    loss = 0.000
    # for every prompt
    # tokenized_context.to(device)
    # gold_labels.to(device)
    if mname=='facebook/blenderbot-400M-distill':
    #mname='facebook/blenderbot-400M-distill'
        model = BlenderbotForConditionalGeneration.from_pretrained(mname)
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model.to(device)
    elif mname=='facebook/blenderbot-1B-distill': #load in 4 bit quantization
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model = BlenderbotForConditionalGeneration.from_pretrained(mname, device_map='cuda:0', load_in_8bit=True)
        
    #model.to(device)
    model.eval()
    for i, row in tqdm(enumerate(generations), total=len(generations), desc='Evaluating fluency'):
        #print()
        #padding='max_length', pad_to_max_length = True, truncation=True, max_length=512, 
        tokenized_context = tokenizer.encode(prompt[i], truncation=True, return_tensors='pt')
        #print(context.size())
        #print(context)
        #tokenized_context = context["input_ids"]
        #tokenized_context = context.squeeze(0)
        attention_mask = tokenized_context.eq(tokenizer.pad_token_id)
        #print(attention_mask)
        #attention_mask = context["attention_mask"]
        #attention_mask = attention_mask.unsqueeze(0)
        #label = row.unsqueeze(0)
        ### for every generation conditioned on the prompt ###
        gen = row.strip()+tokenizer.eos_token
        target = tokenizer.bos_token+row.strip()
        #padding='max_length', pad_to_max_length = True, truncation=True, max_length=256 ,
        tokenized_gen = tokenizer.encode(gen, truncation=True, return_tensors='pt')
        tokenized_gen[tokenized_gen==tokenizer.pad_token_id] = -100
        #, padding='max_length', pad_to_max_length = True, truncation=True, max_length=256 ,
        tokenized_target = tokenizer.encode(target, truncation=True, return_tensors='pt')
        # x=generation.shape
        # y=context.shape
        # z=attention_mask.shape
        # print(tokenized_context.size())
        # print(tokenized_target.size())
        # print(tokenized_gen.size())
        with torch.no_grad():    
            #attention_mask = attention_mask.cuda()
            output = model(input_ids = tokenized_context.cuda(), decoder_input_ids = tokenized_target.cuda(), labels = tokenized_gen.cuda())
            #print(output)
        loss = output.loss
        #print(loss)
        ppl = np.exp(loss.item())
        if ppl < 1e4:   # for sanity
            perplexities.append(ppl)
    return np.nanmean(perplexities)


def conditional_perplexity(tokenized_context, gold_labels, model):
    #device='cpu'
    # device1 = 'cpu'
    perplexities = []
    ct = 0
    # for every prompt
    # tokenized_context.to(device)
    # gold_labels.to(device)
    # model.to(device)
    model.eval()
    for i, row in tqdm(enumerate(gold_labels), total=len(gold_labels), desc='Evaluating fluency'):
        context = tokenized_context["input_ids"][i]
        context = context.unsqueeze(0)
        attention_mask = tokenized_context["attention_mask"][i]
        attention_mask = attention_mask.unsqueeze(0)
        label = row.unsqueeze(0)
        # label.to(device)
        # attention_mask.to(device)
        # context.to(device)
        # model.to(device)
        ### for every generation conditioned on the prompt ###
        # generation = tokenizer.encode(row, return_tensors='pt').to(device)
        # generation[generation==tokenizer.pad_token_id] = -100
        # x=generation.shape
        # y=context.shape
        # z=attention_mask.shape
        with torch.no_grad():    
            output = model(input_ids = context.cuda(), attention_mask = attention_mask.cuda(), labels = label.cuda())
        loss = output.loss
        ppl = np.exp(loss.item())
        #if ppl < 1e4:   # for sanity
        perplexities.append(ppl)
    return np.nanmean(perplexities)

def shrink_data(train_tmp_, target_tmp_, gold_labels_tmp_, toxicity_tmp_, instance_no=100):
    
    ok_list_idx = [i for i in range(len(toxicity_tmp_)) if toxicity_tmp_[i] =='__ok__']
    notok_list_idx = [i for i in range(len(toxicity_tmp_)) if toxicity_tmp_[i] =='__notok__']

    ok_train_idx = []
    notok_train_idx = []

    instance_no = 100
    ok_portion = 0.64 

    n = round(ok_portion*instance_no)
    m = round((1-ok_portion)*instance_no)

    for i in range(n):
        ok_train_idx.append(random.choice(ok_list_idx))

    for j in range(m):
        notok_train_idx.append(random.choice(notok_list_idx))

    train = [train_tmp_[i] for i in ok_train_idx] + [train_tmp_[i] for i in notok_train_idx]
    target = [target_tmp_[i] for i in ok_train_idx] + [target_tmp_[i] for i in notok_train_idx] 
    gold_labels = [gold_labels_tmp_[i] for i in ok_train_idx] + [gold_labels_tmp_[i] for i in notok_train_idx]
    toxicity = [toxicity_tmp_[i] for i in ok_train_idx] + [toxicity_tmp_[i] for i in notok_train_idx]

    print(len(train))
    print(len(target))
    print(len(gold_labels))
    print(len(toxicity))

    return(train, target, gold_labels, toxicity)

def set_seed(seed):

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

def main(args):
    set_seed(args.seed)

    # header = ['param','loss', 'ppl']  
    # with open('/home/leila/compNet/results/Test_perplexity/loss_ppl.csv' , 'w', encoding='UTF8') as f1:
    #     writer = csv.writer(f1)
    #     writer.writerow(header)

    # ===================
    # Evaluate PPL
    # ===================
    #CHECK ARGS:
    # =================== 
    # print(f"SAVE FOLDER: {args.save_folder}")
    # print(f"SAVE DATA FOLADER: {args.data_folder}")
    # print(f"SAVE SPLIT: {args.split}")
    # print(f"SAVE ppl: {args.ppl}")
    # print(f"SAVE toxicity: {args.toxicity}")
    # print(f"SAVE TOP-K: {args.top_k}")
    # print(f"SAVE TOP-P: {args.top_p}")
    # print(f"SAVE PPL PATH: {args.ppl_path}")
    # print(f"SAVE GENERATION PATH: {args.gen_path}")
    # print(f"SAVE context_gen_path: {args.context_gen_path}")
    # print(f"SAVE CHECKPOINT: {args.checkpoint}")

    #if args.ppl:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #torch.cuda.empty_cache()
    # model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_and_path)
    path = args.checkpoint_path + args.checkpoint 
    #"/home/leila/compNet/CN_filtered_newJS/parameters/1_1_3_3_CN_filtered.pth"
    #print(f"##########################   This is checkpoint path: {path} ############################\n")
    #token_dict = {'sep_token': '__sep__'}
    #tokenizer.add_special_tokens(token_dict)
    #special_tokens_dict = {'additional_special_tokens': ['__bprs__','__eprs__']}
    #tokenizer.add_special_tokens(special_tokens_dict)
    #model.resize_token_embeddings(len(tokenizer))
    mname=args.mname
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.mname)
    print(f"path is: {path}")
    Cpoint = torch.load(path, map_location=device) 
    if isinstance(Cpoint, dict):
        main_model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.mname)
        main_model.load_state_dict(Cpoint['model_state_dict'])
    else:
        main_model = torch.load(path)

    #mname=args.mname
    # = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
    #eval_model.to(device)
    main_model.to(device)

    # train_tmp, target_tmp, gold_labels_tmp, toxicity_tmp, total, count = get_contexted(args.data_folder, 3, args.split)
    # #valData, valTarget, valLabels, valToxicity, total, count = get_valid_data('valid.txt', window_size = 3)
    # remove_term = 'Hey do you want to talk about something else? How about we talk about'
    # remove_list_idx = [i for i in range(len(target_tmp)) if remove_term in target_tmp[i] or remove_term in train_tmp[i]]

    # train = [ele for idx, ele in enumerate(train_tmp) if idx not in remove_list_idx]
    # target = [ele for idx, ele in enumerate(target_tmp) if idx not in remove_list_idx]
    # gold_labels = [ele for idx, ele in enumerate(gold_labels_tmp) if idx not in remove_list_idx]
    # toxicity = [ele for idx, ele in enumerate(toxicity_tmp) if idx not in remove_list_idx]
    # train, target, gold_labels, toxicity, total, count = get_contexted(args.data_folder, args.window_size, args.split)

    train, target, gold_labels, toxicity = get_contexted(args.data_folder, args.window_size, args.split)
    if args.data_shrink:
        train, target, gold_labels, toxicity  = shrink_data(train, target, gold_labels, toxicity, 100)
        #, total, count

    tokenized_train_context = tokenizer(train, padding='max_length', pad_to_max_length = True, truncation=True, max_length=512,return_tensors="pt",add_special_tokens= True )
    tokenized_train_labels = tokenizer(target, padding='max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= True)
    tokenized_gold_labels_tmp = tokenizer(gold_labels, padding = 'max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= False)
    tokenized_gold_labels = tokenized_gold_labels_tmp['input_ids']
    # tokenized_gold_labels[tokenized_gold_labels==0] = -100

    dataset = BAD(tokenized_train_context, tokenized_train_labels, tokenized_gold_labels, toxicity)
    collate_fn = Seq2SeqDataCollator(tokenizer.pad_token_id)
    dataloader = get_dataloader(dataset, args.batch_size, collate_fn)
    
    # model.eval()
    # running_loss = 0.0
    # val_loss = 0.0
    # val_ppl = 0.0
    # for batch in tqdm(dataloader, desc='Evaluating', total=len(dataloader), ncols=100):
    #     inputs = {
    #         "input_ids": batch["input_ids"].cuda(),
    #         "attention_mask": batch["attention_mask"].cuda(),
    #         "decoder_input_ids": batch["decoder_input_ids"].cuda(),
    #         "labels": batch["labels"].cuda(),
    #     }
    #     with torch.set_grad_enabled(False):
    #         main_outputs = main_model(**inputs)
    #         loss = main_outputs.loss
    #         running_loss += loss.item()
    #         main_logits = main_outputs.logits

    #         toxic_outputs = model_toxic(**inputs)
    #         toxic_logits = main_outputs.logits                

    #         clean_outputs = model_clean(**inputs)
    #         clean_logits = clean_outputs.logits

    #         final_logit = 
    # # compute ppl
    # val_loss = running_loss / len(dataloader)
    # val_ppl = np.exp(val_loss)
    
    # header = ["model", "val_loss", "val_ppl"]
    # data = [args.checkpoint ,val_loss, val_ppl]
    # #str(args.run_mode_type)
    # with open((args.ppl_path+str(args.split)+"_ppl_"+str(args.checkpoint)+".csv").replace(".pth",''), "w", encoding='UTF8') as f:
    #     wrtr = writer(f)
    #     wrtr.writerow(header)
    #     wrtr.writerow(data) 

    # # ===================
    # # Evaluate Generation
    # # ===================
    # model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_and_path)
    # # model = torch.load('/home/leila/compNet/results/save.pth')
    # tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_and_path)

    # token_dict = {'sep_token': '__sep__'}
    # tokenizer.add_special_tokens(token_dict)

    # special_tokens_dict = {'additional_special_tokens': ['__bprs__','__eprs__']}
    # tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))

    # model.cuda()

    # train, target, gold_labels = get_contexted_for_generation(f"{args.data_folder}/{args.split}.txt", args.window_size)

    # tokenized_train_context = tokenizer(train, padding='max_length', pad_to_max_length = True, truncation=True, max_length=512,return_tensors="pt",add_special_tokens= True )
    # tokenized_train_labels = tokenizer(target, padding='max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= True)
    # tokenized_gold_labels_tmp = tokenizer(gold_labels, padding = 'max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= False)
    # tokenized_gold_labels = tokenized_gold_labels_tmp['input_ids']
    # # tokenized_gold_labels[tokenized_gold_labels==0] = -100

    # dataset = BAD(tokenized_train_context, tokenized_train_labels, tokenized_gold_labels)
    # collate_fn = Seq2SeqDataCollator(tokenizer.pad_token_id)
    # dataloader = get_dataloader(dataset, args.batch_size, collate_fn)
    #
    main_model.eval()
    #eval_model.eval()
    start_time = timeit.default_timer()
    if args.toxicity:
        inputs = []
        generations = []

        for i, batch in enumerate(tqdm(dataloader, desc='Generating', total=len(dataloader), ncols=100)):
            gen_kwargs = {
                "pad_token_id": tokenizer.pad_token_id,
                #"num_beams": model.config.num_beams, 
                "do_sample":True,
                "top_k":args.top_k,
                "top_p":args.top_p,
                "temperature": args.temperature,
                "max_length": args.length, 
                "repetition_penalty": args.repetition_penalty,
            }
            # Generation loop
            # while True:
  
            generated_sequences = main_model.generate(
                                input_ids=batch["data_ids"].cuda(),
                                attention_mask=batch["data_msk"].cuda(),
                                **gen_kwargs,
                                )

            decoded_inputs = tokenizer.batch_decode(batch["data_ids"], skip_special_tokens=True)
            decoded_sequences = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            for input_text, text in zip(decoded_inputs, decoded_sequences):
                input_text = input_text.replace("</s> <s>", "\t").replace("__start__", "").replace("__end__", "")
                inputs.append(input_text)
                generations.append(text) 

        elapsed = timeit.default_timer() - start_time 
        dataset_size = len(dataloader)*args.batch_size
        time_per_sample = elapsed/dataset_size        
        
        # Save the file for single-turn toxicity evaluation setting
        with open((args.gen_path+"gen_"+str(args.checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f1:
            for line in enumerate(generations):
                #print(line)
                f1.write(f"text:{line}\tlabels:__ok__\tepisode_done:True\n")
            #f1.write(f"Overal Decoding Time:{elapsed}\nAverage Per Sample Time: {time_per_sample}\n")

        # with open(("gen_sent_only_"+str(args.checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f3:
        #     for line in enumerate(generations):
        #         #print(line)
        #         f3.write(f"{line}\n")
        #     f3.write(f"Overal Decoding Time:{elapsed}\nAverage Per Sample Time: {time_per_sample}\n")

        # Save a seperate file with input and generation
        #with open(os.path.join("/home/leila/compNet/results/bb/bb_generations/bb_contxt_gen.txt", Path(args.checkpoint).stem+'.txt'), "a") as f2:
        with open((args.context_gen_path+"context_gen_"+str(args.checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f2:
            for index, line in enumerate(generations):
                #print(line)
                f2.write(f"input:{inputs[index]}\tgen_text:{line}\n")
            f2.write(f"Overal Decoding Time:{elapsed}\nAverage Per Sample Time: {time_per_sample}\n")
        # Comment the multi-turn toxicity evaluation setting
        # with open("save_gen_multi_turn.txt", "w") as f:
        #     for line, input_text in zip(generations, inputs):
        #         f.write(f"text:{input_text}\t{line}\tlabels:__ok__\tepisode_done:True\n")
    # if args.ppl:
    #     # gen_list=[]
    #     # gen_path = (args.gen_path+"gen_"+str(args.checkpoint)+".txt").replace(".pth",'') #'/home/leila/LOT_Neurips_2023/generations/bb/baselines/gen_bb_vanila.txt' 
    #     # with io.open(gen_path) as fr:
    #     #     for line in fr:
    #     #         tmp_gen = line.split(',',1)[1].split('labels')[0].replace(' "','').replace(" '",'').replace('")','').replace("')",'')+'__end__' #for temp in list(filter(None,re.split("\n",line.strip())))]
    #     #         #print(tmp_gen)
    #     #         gen_list.append(tmp_gen)
    #     perplex = conditional_perplexity(tokenized_train_context, tokenized_gold_labels, main_model) 
    #     with open((args.ppl_path+"ppl_"+str(args.checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f4:
    #         f4.write(f"perplexity:{perplex}")
    if args.ppl:
        #Get generations
        gen_list=[]
        gen_path = (args.gen_path+"gen_"+str(args.checkpoint)+".txt").replace(".pth",'') #'/home/leila/LOT_Neurips_2023/generations/bb/baselines/gen_bb_vanila.txt' 
        with io.open(gen_path) as fr:
            for line in fr:
                tmp_gen = line.split(',',1)[1].split('labels')[0].replace(' "','').replace(" '",'').replace('")','').replace("')",'')
                #tmp_gen = line.split(',',1)[1].split('labels')[0].replace(' "','').replace(" '",'').replace('")','').replace("')",'')+'__end__' 
                #for temp in list(filter(None,re.split("\n",line.strip())))]
                #print(tmp_gen)
                gen_list.append(tmp_gen)
        #Get LLAMA formatted context from the dataset
        train, target, prompt = get_LLAMA_contexted(args.data_folder, args.window_size, args.split)
        llama_ppl = LlaMA_perplexity(train, gen_list)

        train, target, label, toxicity = bb_get_contexted(args.data_folder, args.window_size, args.split)
        mname1 = 'facebook/blenderbot-400M-distill'
        mname2 = 'facebook/blenderbot-1B-distill'
        
        bb_400_ppl = model_perplexity(train, gen_list, mname1)     
        bb_1b_ppl = model_perplexity(train, gen_list, mname2) 
        #for i in range(len(train)):
            # print(f"{train[i]}\n")
            # print(label[i])
            # print(prompt[i])

        #dummy  = generate_with_llama(context=prompt, llama_checkpoint = args.llama_checkpoint, gen_path = args.gen_path, context_gen_path = args.context_gen_path)
        
        with open((args.ppl_path+"ppl_"+str(args.llama_checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f40:
            #LLAMA_perplex = LlaMA_perplexity(train, gen_list) 
            f40.write(f"LlaMA perplexity:{llama_ppl}\n")

        with open((args.ppl_path+"ppl_bb400.txt").replace(".pth",''), "w", encoding='UTF8') as f41:
            #LLAMA_perplex = LlaMA_perplexity(train, gen_list) 
            f41.write(f"bb_400 perplexity:{bb_400_ppl}\n")

        with open((args.ppl_path+"ppl_bb1b.txt").replace(".pth",''), "w", encoding='UTF8') as f42:
            #LLAMA_perplex = LlaMA_perplexity(train, gen_list) 
            f42.write(f"bb_1b perplexity:{bb_1b_ppl}\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_folder', type=str, default= '/home/leila/LOT_Neurips_2023/Data_models/BAD/dataset/bot_adversarial_dialogue_datasets_with_persona') #"/home/leila/compNet_AAAI/validation/data/bot_adversarial_dialogue_datasets_with_persona")  
    parser.add_argument('--split', type=str, default="test.txt")
    parser.add_argument('--window_size', type=int, default=6)
    # model
    parser.add_argument('--model_name_and_path', type=str, default="facebook/blenderbot_small-90M")
   # parser.add_argument('--save_folder', type=str, default="/home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_clean")
    parser.add_argument('--checkpoint', type=str, default= "") 
    parser.add_argument('--ppl_path', type=str, default="/home/leila/EACL2023/ppl_investigation/ppl/")
    parser.add_argument('--gen_path', type=str, default="/home/leila/EACL2023/ppl_investigation/generations/")
    parser.add_argument('--context_gen_path', type=str, default="/home/leila/EACL2023/ppl_investigation/generations/")
    
     # evaluation setings
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)

    parser.add_argument('--do_sample',action="store_true")
    parser.add_argument('--top_k',type=int, default=10)
    parser.add_argument('--top_p',type=float, default=0.6)
    parser.add_argument('--num_beam',type=int, default=1)
   
    parser.add_argument('--ppl', action="store_true", default=True)
    parser.add_argument('--toxicity', action="store_true", default=True)
    # parser.add_argument('--unigram_f1', action="store_true")
    parser.add_argument('-sp', '--add_special_tokens', action="store_true")
    parser.add_argument('--run_mode_type', type=str, default="Unlikelihood")
    parser.add_argument('--checkpoint_path', type=str, default="/home/leila/LOT_Neurips_2023/Data/DiaSafety/final_model/")
    parser.add_argument('--data_shrink',action='store_true')
    parser.add_argument('--mname', type=str, default="facebook/blenderbot_small-90M")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(args)


