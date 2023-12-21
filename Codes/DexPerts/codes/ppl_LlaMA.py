import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer

import torch
import bitsandbytes as bnb

from tqdm import tqdm
from numpy import array
import numpy as np

def LlaMA_perplexity(context, genlist):
        
    device='cuda'

    # # Activate 4-bit precision base model loading
    # use_4bit = True

    # # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    # # Quantization type (fp4 or nf4)
    # bnb_4bit_quant_type = "nf4"

    # # Activate nested quantization for 4-bit base models (double quantization)
    # use_nested_quant = False

    base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"
    # print(llama_tokenizer.pad_token_id)
    # print(llama_tokenizer.eos_token_id)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype, #torch.float16,
    bnb_4bit_use_double_quant=False,
    )


    base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # device1 = 'cpu'
    perplexities = []
    ct = 0
    loss = 0.000
    # for every prompt

    base_model.eval()
    
    #print(f"This is the shape of context: {array(context).shape}")

    for i, row in tqdm(enumerate(genlist), total=len(genlist), desc='Evaluating fluency'):

        ### for every generation conditioned on the prompt ###
        full_input = context[i]+row.strip()+'</s>'
        #print(full_input)
        #padding='max_length', pad_to_max_length = True,truncation=True, max_length=256,
        tokenized_full_context = llama_tokenizer.encode(full_input, return_tensors="pt")
        full_attention_mask = tokenized_full_context.eq(llama_tokenizer.pad_token_id)
        #attention_mask[i][:len(train[i])-len(label[i])]=llama_tokenizer.pad_token_id
        tokenized_context = llama_tokenizer.encode(context[i], return_tensors="pt")
        context_attention_mask = tokenized_context.eq(llama_tokenizer.pad_token_id)
        
        tokenized_full_context[tokenized_full_context==llama_tokenizer.pad_token_id] = -100
        tokenized_context[tokenized_context==llama_tokenizer.pad_token_id] = -100
        # print(tokenized_context.size())
        # print(tokenized_full_context.size())

        with torch.no_grad():    
            # full_loss_tmp = base_model(input_ids = tokenized_full_context.cuda(), labels = tokenized_full_context.cuda())[0]*(tokenized_full_context.shape[1]-1)
            full_loss_tmp = base_model(input_ids = tokenized_full_context.cuda(), labels = tokenized_full_context.cuda())
            full_loss = full_loss_tmp.loss * (tokenized_full_context.shape[1]-1)
            #print(f"tokenized_full_context: {tokenized_full_context.shape[1]}")
            # context_loss_tmp = base_model(input_ids = tokenized_context.cuda(), labels = tokenized_context.cuda())[0]*(tokenized_context.shape[1]-1)
            context_loss_tmp = base_model(input_ids = tokenized_context.cuda(), labels = tokenized_context.cuda())         
            context_loss = context_loss_tmp.loss * (tokenized_context.shape[1]-1) 
            #print(f"tokenized_full_context: {tokenized_context.shape[1]}")

            loss = (full_loss - context_loss) / (tokenized_full_context.shape[1] - tokenized_context.shape[1])
        #loss = output.loss
        ppl = np.exp(loss.item())
        if ppl < 1e4:   # for sanity
            perplexities.append(ppl)

    return np.nanmean(perplexities)


def generate_with_llama(context, llama_checkpoint, gen_path, context_gen_path):

    name = "NousResearch/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",    # finds GPU
    )

    generations=[]
    inputs=[]
    for i, row in tqdm(enumerate(context), total=len(context), desc='Generating with LlaMA'):
        sequences = generation_pipe(
            row,
            max_length=128,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=10,
            temperature=0.4,
            top_p=0.9
        )
        generations.append(sequences)
        inputs.append(row)
    
    with open((gen_path+"gen_"+str(llama_checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f4:
        for line in enumerate(generations): 
            f4.write(f"text:{line}\tlabels:__ok__\tepisode_done:True\n")

    with open((context_gen_path+"context_gen_"+str(llama_checkpoint)+".txt").replace(".pth",''), "w", encoding='UTF8') as f2:
        for index, line in enumerate(generations):
            f2.write(f"input:{inputs[index]}\tgen_text:{line}\n")

    return None