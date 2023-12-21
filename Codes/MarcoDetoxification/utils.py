from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from sacremoses import MosesDetokenizer
import numpy as np
import torch
import random
import html
import re
import ftfy
from nltk.tokenize.casual import casual_tokenize
import io 
import os

nl_tok = "[<NEW>]"
md = MosesDetokenizer(lang='en')

def detokenize(input):
    # return TreebankWordDetokenizer().detokenize(input)
    return md.detokenize(input)

def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def bool2str(cand):
    if cand:
        return "T"
    return "F"

def seed_everything(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Only useful for convolution
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def preprocess(text, preserve_lines = False):
    if preserve_lines:
        return ftfy.fix_text(html.unescape(text))
    # Remove linee break and excess spaces
    return ftfy.fix_text(html.unescape(re.sub(r'\s+', ' ', text).strip()))

def get_BAD_data(data_directory, window_size, data_split):
    #processes test data into a list of pure text for masking and filling
    with io.open(os.path.join(data_directory,data_split)) as f:
        total = 0
        count = 0
        train = []
        targets = []
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
                train_sample = context                                           
                target_sample = target 
                train.append(train_sample.strip())
                targets.append(target_sample.strip())
#flatten = lambda l: [item for sublist in l for item in sublist]
#contexted = flatten(contexted)             
#return contexted, targets, personas
            i=i+1
    return train , targets

def eval_format(gen_path_read,gen_path_write):
    with open(gen_path_write, 'w', encoding='UTF8') as f1:
        with io.open(gen_path_read) as fr:
            for line in fr:
                #tmp_gen = line.split(',')[1].split('\t')[0].replace(' "','').replace(" '",'').replace('")','').replace("')",'') #for temp in list(filter(None,re.split("\n",line.strip())))]
                f1.write(f"text:{line}\tlabels:__ok__\tepisode_done:True\n")

            with open(os.path.join(final_path, "orig.txt"), "w") as f:
                for l in inputs:
                    tmp = re.sub(r"\s+", " ", l).strip() + "\n"
                    f.write(f"text:{tmp}\tlabels:__ok__\tepisode_done:True\n")
            with open(os.path.join(final_path, "gen.txt"), "w") as f:
                for l in decoded_outputs:
                    tmp = re.sub(r"\s+", " ", l).strip() + "\n"
                    f.write(f"text:{tmp}\tlabels:__ok__\tepisode_done:True\n")

# Quick test
# TreebankWordDetokenizer.detokenize(TreebankWordTokenizer.tokenize("sh*t"))