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
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import timeit

import transformers
from transformers import (
    BlenderbotSmallForConditionalGeneration, 
    BlenderbotSmallTokenizer,
    top_k_top_p_filtering
)
#from typer import Exit

from data_utils.dataloader import Seq2SeqDataCollator, get_dataloader
from data_utils.bad import BAD, get_contexted, get_contexted_for_generation
from utils.utils import set_seed

# debugpy.listen(5678)
# print("socket waitiiiiiiiiiing for client")
# debugpy.wait_for_client()    

def conditional_perplexity(tokenized_context, generations, model, tokenizer, device='cuda'):
    perplexities = []
    ct = 0
    # for every prompt
    tokenized_context.to(device)
    for i, row in tqdm(enumerate(generations), total=len(generations), desc='Evaluating fluency'):
        context = tokenized_context["input_ids"][i]
        context = context.unsqueeze(0)
        attention_mask = tokenized_context["attention_mask"][i]
        attention_mask = attention_mask.unsqueeze(0)
        # for every generation conditioned on the prompt
        generation = tokenizer.encode(row, return_tensors='pt').to(device)
        generation[generation==tokenizer.pad_token_id] = -100
        x=generation.shape
        y=context.shape
        z=attention_mask.shape
        output = model(input_ids = context, attention_mask = attention_mask, labels = generation)
        loss = output.loss
        ppl = np.exp(loss.item())
        #if ppl < 1e4:   # for sanity
        perplexities.append(ppl)
    return np.nanmean(perplexities)

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

    #if args.toxicity:

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    # model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_and_path)
    #main_path = os.path.join(args.main_save_folder, args.main_checkpoint)
    Toxic_path = os.path.join(args.toxic_save_folder, args.toxic_checkpoint)
    Clean_path = os.path.join(args.clean_save_folder, args.clean_checkpoint)

    #print(f"##########################   This is checkpoint path: {path} ############################\n")
    #token_dict = {'sep_token': '__sep__'}
    #tokenizer.add_special_tokens(token_dict)
    #special_tokens_dict = {'additional_special_tokens': ['__bprs__','__eprs__']}
    #tokenizer.add_special_tokens(special_tokens_dict)
    #model.resize_token_embeddings(len(tokenizer))
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_and_path)

    toxic_Cpoint = torch.load(Toxic_path, map_location=device) 
    clean_Cpoint = torch.load(Clean_path, map_location=device) 
    # if isinstance(Cpoint, dict):
    model_toxic = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_and_path)
    model_clean = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_name_and_path)
    model_toxic.load_state_dict(toxic_Cpoint['model_state_dict'])
    model_clean.load_state_dict(clean_Cpoint['model_state_dict'])        
    # else:
    # model_toxic = torch.load(Toxic_path)
    # model_clean = torch.load(Clean_path)   
    
    mname='facebook/blenderbot_small-90M'
    main_model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
    main_model.to(device)
    model_toxic.to(device)
    model_clean.to(device)

    # train_tmp, target_tmp, gold_labels_tmp, toxicity_tmp, total, count = get_contexted(args.data_folder, 3, args.split)
    # #valData, valTarget, valLabels, valToxicity, total, count = get_valid_data('valid.txt', window_size = 3)
    # remove_term = 'Hey do you want to talk about something else? How about we talk about'
    # remove_list_idx = [i for i in range(len(target_tmp)) if remove_term in target_tmp[i] or remove_term in train_tmp[i]]

    # train = [ele for idx, ele in enumerate(train_tmp) if idx not in remove_list_idx]
    # target = [ele for idx, ele in enumerate(target_tmp) if idx not in remove_list_idx]
    # gold_labels = [ele for idx, ele in enumerate(gold_labels_tmp) if idx not in remove_list_idx]
    # toxicity = [ele for idx, ele in enumerate(toxicity_tmp) if idx not in remove_list_idx]
    train, target, gold_labels, toxicity, total, count = get_contexted(args.data_folder, args.window_size, args.split)

    tokenized_train_context = tokenizer(train, padding='max_length', pad_to_max_length = True, truncation=True, max_length=512,return_tensors="pt",add_special_tokens= True )
    tokenized_train_labels = tokenizer(target, padding='max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= True)
    tokenized_gold_labels_tmp = tokenizer(gold_labels, padding = 'max_length', pad_to_max_length = True, truncation=True, max_length=256,return_tensors="pt",add_special_tokens= False)
    tokenized_gold_labels = tokenized_gold_labels_tmp['input_ids']
    # tokenized_gold_labels[tokenized_gold_labels==0] = -100

    dataset = BAD(tokenized_train_context, tokenized_train_labels, tokenized_gold_labels)
    collate_fn = Seq2SeqDataCollator(tokenizer.pad_token_id)
    dataloader = get_dataloader(dataset, args.batch_size, collate_fn)
    
    #model.eval()
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

    if args.toxicity:
        main_model.eval()
        model_clean.eval()
        model_toxic.eval()
        context_ = []
        generations = []
        alpha = 0.0
        alpha = torch.tensor(alpha).to(device)

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
        start_time = timeit.default_timer()
        for i, batch in enumerate(tqdm(dataloader, desc='Generating', total=len(dataloader), ncols=100)):
            inputs = {
                "input_ids": batch["input_ids"].to(device),#.cuda(),
                "attention_mask": batch["attention_mask"].to(device),#.cuda(),
                "decoder_input_ids": batch["decoder_input_ids"].to(device),#.cuda(),
                "labels": batch["labels"].to(device)#.cuda(),
            }

            #batch_size, seqLen = inputs["input_ids"].shape
            unfinished_sents = torch.ones((args.batch_size,1), dtype=torch.long, device = device).to(device)

            generated_sequence = torch.full((args.batch_size,1), tokenizer.bos_token_id, requires_grad=False).to(device) #([[tokenizer.bos_token_id]]).cuda()  # initial token
            #input_ids = batch["input_ids"]
            # unfinished_sents = torch.ones(batch_size, dtype=torch.long, device = device)

            # generated_sequence = torch.tensor([[tokenizer.bos_token_id]])  # initial token
            #input_ids = batch["input_ids"]
            past_key_values = None

            with torch.no_grad():
                main_output = main_model(
                input_ids=inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                decoder_input_ids=generated_sequence,
                past_key_values=past_key_values,
                use_cache = True
                )
                #.main_encoder_outputs=main_output.encoder_last_hidden_state

                clean_output = model_clean(
                input_ids=inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                decoder_input_ids=generated_sequence,
                past_key_values=past_key_values
                )
                #clean_encoder_outputs=clean_output.encoder_last_hidden_state

                toxic_output = model_toxic(
                input_ids=inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                decoder_input_ids=generated_sequence,
                past_key_values=past_key_values
                )
                #toxic_encoder_outputs=toxic_output.encoder_last_hidden_state

            main_encoder_outputs=main_output.encoder_last_hidden_state
            toxic_encoder_outputs=toxic_output.encoder_last_hidden_state
            clean_encoder_outputs=clean_output.encoder_last_hidden_state

            # Generation loop
            while True: 
                   # From here on, use cached attention
                main_past_key_values = main_output.past_key_values
                toxic_past_key_values = toxic_output.past_key_values
                clean_past_key_values = clean_output.past_key_values

                main_logits = main_output.logits
                clean_logits = clean_output.logits
                toxic_logits = toxic_output.logits

                #main_logits = top_k_top_p_filtering(logits=main_logits, top_p=args.top_p, top_k=args.top_k)                
                ensemble_logits = main_logits + alpha * (clean_logits - toxic_logits)

                # next_main_token_logit = main_output.logits[:, -1, :]
                # next_clean_token_logit = clean_output.logits[:, -1, :]
                # next_toxic_token_logit = toxic_output.logits[:, -1, :]
                next_ensemble_token_logit = ensemble_logits[:, -1, :]

                filtered_ensemble_logits = top_k_top_p_filtering(logits=next_ensemble_token_logit, top_k=args.top_k, top_p=args.top_p)
                next_token_distribution = filtered_ensemble_logits.softmax(dim=-1)

                # Sample next token
                next_ensemble_token = torch.multinomial(next_token_distribution, 1)
                                
                # Append token to generated sequence
                generated_sequence = torch.cat((generated_sequence, next_ensemble_token), dim=1)
            
                # Stop if EOS token generated
                tokens_to_add = next_ensemble_token * unfinished_sents + tokenizer.pad_token_id * (1 - unfinished_sents)
                #if (generated_sequence.squeeze()[-1] == tokenizer.eos_token_id):
                    #break
                
                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                with torch.no_grad():
                    # output = model(
                    #     decoder_input_ids=torch.tensor([[generated_sequence.squeeze()[-1]]]),
                    #     past_key_values=past_key_values,
                    #     encoder_outputs=encoder_outputs
                    #     attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                    # )
                    main_output = main_model(
                    input_ids=inputs["input_ids"],
                    attention_mask = inputs["attention_mask"],
                    decoder_input_ids=generated_sequence[:,-1].unsqueeze(-1),
                    past_key_values=main_past_key_values,
                    use_cache = True
                    )
                    #.main_encoder_outputs=main_output.encoder_last_hidden_state

                    clean_output = model_clean(
                    input_ids=inputs["input_ids"],
                    attention_mask = inputs["attention_mask"],
                    decoder_input_ids=generated_sequence[:,-1].unsqueeze(-1),
                    past_key_values=clean_past_key_values
                    )
                    #clean_encoder_outputs=clean_output.encoder_last_hidden_state

                    toxic_output = model_toxic(
                    input_ids=inputs["input_ids"],
                    attention_mask = inputs["attention_mask"],
                    decoder_input_ids=generated_sequence[:,-1].unsqueeze(-1),
                    past_key_values=toxic_past_key_values
                    )
                  
                #=======================================================================================    
                # next_token = next_token_scores.argmax().unsqueeze(0).unsqueeze(0)  # greedy decoding
                # generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

                # main_generated_sequences = main_model(
                #                         **inputs,
                #                         **gen_kwargs,
                #                         )
                # main_logit = main_generated_sequences.logit

                # toxic_generated_sequences = model_toxic(
                #                         **inputs,
                #                         **gen_kwargs,
                #                         )
                # toxic_logit = toxic_generated_sequences.logit

                # clean_generated_sequences = model_clean.(
                #                         **inputs,
                #                         **gen_kwargs,
                #                         )
                # clean_logit = clean_generated_sequences.logit
        
                # final_dist = nn.functional.softmax(
                #     main_logit + alpha * (clean_logit - toxic_logit), dim=1).to(device)
                #====================================================================================

                decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                decoded_response = tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)
                for input_text, text in zip(decoded_inputs, decoded_response):
                    input_text = input_text.replace("</s> <s>", "\t").replace("__eprs__", "\t").replace("__bprs__", "\t").replace("__sep__", "\t").replace("__start__", "").replace("__end__", "")
                    context_.append(input_text)
                    generations.append(text) 
        
                   
        elapsed = timeit.default_timer() - start_time 
        dataset_size = len(dataloader)*batch_size
        time_per_sample = elapsed/dataset_size    
        #print(len(generations),generations[0])
         
        # Save the file for single-turn toxicity evaluation setting
        with open((args.gen_path+"gen_"+str(args.checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f1:
            for line in enumerate(generations):
                #print(line)
                f1.write(f"text:{line}\tlabels:__ok__\tepisode_done:True\n")
        
        # Save a seperate file with input and generation
        #with open(os.path.join("/home/leila/compNet/results/bb/bb_generations/bb_contxt_gen.txt", Path(args.checkpoint).stem+'.txt'), "a") as f2:
        with open((args.context_gen_path+"context_gen_"+str(args.checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f2:
            for index, line in enumerate(generations):
                #print(line)
                f2.write(f"input:{context_[index]}\tgen_text:{line}\n")
            f2.write(f"Overal Decoding Time:{elapsed}\nAverage Per Sample Time: {time_per_sample}\n")

        with open(("gen_sent_only_"+str(args.checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f3:
            for line in enumerate(generations):
                #print(line)
                f3.write(f"{line}\n")
            #f3.write(f"Overal Decoding Time:{elapsed}\nAverage Per Sample Time: {time_per_sample}\n")

        # Comment the multi-turn toxicity evaluation setting
        # with open("save_gen_multi_turn.txt", "w") as f:
        #     for line, input_text in zip(generations, inputs):
        #         f.write(f"text:{input_text}\t{line}\tlabels:__ok__\tepisode_done:True\n")

    if args.ppl:
        mname='facebook/blenderbot_small-90M'
        eval_model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
        eval_model.to(device)
        gen_list=[]
        gen_path = (args.gen_path+"gen_"+str(args.checkpoint)+".txt").replace(".pth",'') #'/home/leila/compNet/CN_filtered_newJS/generations/gen_1_1_3_1_CN_filtered.txt'
        with io.open(gen_path) as fr:
            for line in fr:
                tmp_gen = line.split(',')[1].split('\t')[0].replace(' "','').replace(" '",'').replace('")','').replace("')",'') #for temp in list(filter(None,re.split("\n",line.strip())))]
                gen_list.append(tmp_gen)
        perplex = conditional_perplexity(tokenized_train_context, gen_list, eval_model, tokenizer, device) 
        with open((args.ppl_path+"ppl_"+str(args.checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f4:
            f4.write(f"perplexity:{perplex}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_folder', type=str, default= "/home/leila/LOT_Neurips_2023/Data/DiaSafety/DiaSafety_dataset")#"/home/leila/LOT_Neurips_2023/Data/DiaSafety/DiaSafety_dataset")
    parser.add_argument('--split', type=str, default="test.txt")
    parser.add_argument('--window_size', type=int, default=3)
    # model
    parser.add_argument('--model_name_and_path', type=str, default="facebook/blenderbot_small-90M")
   # parser.add_argument('--save_folder', type=str, default="/home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_clean")
    parser.add_argument('--checkpoint', type=str, default="Dexperts.pth")
    parser.add_argument('--ppl_path', type=str, default="/home/leila/LOT_Neurips_2023/Data/DiaSafety/results/ppl/Dexperts/")#"/home/leila/compNet/Baselines/DexPerts/Metrics/Dexpert_PPL/")
    parser.add_argument('--gen_path', type=str, default="/home/leila/LOT_Neurips_2023/Data/DiaSafety/results/generations/Dexperts/")#"/home/leila/compNet/Baselines/DexPerts/generations/")
    parser.add_argument('--context_gen_path', type=str, default="/home/leila/LOT_Neurips_2023/Data/DiaSafety/results/generations/Dexperts/")#"/home/leila/compNet/Baselines/DexPerts/generations/")
    
    #parser.add_argument('--main_save_folder', type=str, default="/home/leila/compNet_AAAI/results/bb/generations/bb_cleanExpert_filtered_ok_data")
    #parser.add_argument('--main_checkpoint', type=str, default="/home/leila/compNet_AAAI/results/bb/generations/bb_cleanExpert_filtered_ok_data")
    parser.add_argument('--toxic_save_folder', type=str, default="/home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_toxic")
    parser.add_argument('--toxic_checkpoint', type=str, default="toxic_expert_0_filtered_clean_data.pth")
    parser.add_argument('--clean_save_folder', type=str, default="/home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_clean")
    parser.add_argument('--clean_checkpoint', type=str, default="BAD_filtered_clean_2.pth")

     # evaluation setings
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)

    parser.add_argument('--do_sample',action="store_true")
    parser.add_argument('--top_k',type=int, default=50)
    parser.add_argument('--top_p',type=float, default=0.95)
   
    parser.add_argument('--ppl', action="store_true")
    parser.add_argument('--toxicity', action="store_true")
    # parser.add_argument('--unigram_f1', action="store_true")
    parser.add_argument('-sp', '--add_special_tokens', action="store_true")
    parser.add_argument('--run_mode_type', type=str, default="Dexperts")


    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(args)


