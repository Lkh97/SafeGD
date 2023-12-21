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
import debugpy

# debugpy.listen(5678)
# print("socket waitiiiiiiiiiing for client")
# debugpy.wait_for_client()    

gen_original_path_write = '/home/leila/LOT_Neurips_2023/Codes/MarcoDetoxification/data/dexp_outputs/manual/masked_thresh1.5/aa0.5_ae2.5_ab1.0_basebase_antiantie_expertexper_temp1.0_sampleF_topk50_reppenalty1.0_filterp1.0_maxlength128_topp1.0/orig_formatted.txt'
gen_original_path_read = '/home/leila/LOT_Neurips_2023/Codes/MarcoDetoxification/data/dexp_outputs/manual/masked_thresh1.5/aa0.5_ae2.5_ab1.0_basebase_antiantie_expertexper_temp1.0_sampleF_topk50_reppenalty1.0_filterp1.0_maxlength128_topp1.0/orig.txt'

gen_marco_path_write = '/home/leila/LOT_Neurips_2023/Codes/MarcoDetoxification/data/dexp_outputs/manual/masked_thresh1.5/aa0.5_ae2.5_ab1.0_basebase_antiantie_expertexper_temp1.0_sampleF_topk50_reppenalty1.0_filterp1.0_maxlength128_topp1.0/gen_formatted.txt'
gen_marco_path_read = '/home/leila/LOT_Neurips_2023/Codes/MarcoDetoxification/data/dexp_outputs/manual/masked_thresh1.5/aa0.5_ae2.5_ab1.0_basebase_antiantie_expertexper_temp1.0_sampleF_topk50_reppenalty1.0_filterp1.0_maxlength128_topp1.0/gen.txt'

with open(gen_original_path_write, 'w', encoding='UTF8') as f1:
    with io.open(gen_original_path_read) as fr:
        for line in enumerate(fr):
            #tmp_gen = line.split(',')[1].split('\t')[0].replace(' "','').replace(" '",'').replace('")','').replace("')",'') #for temp in list(filter(None,re.split("\n",line.strip())))]
            line = str(line).replace('\\n','')
            f1.write(f"text:{line}\tlabels:__ok__\tepisode_done:True\n")

with open(gen_marco_path_write, 'w', encoding='UTF8') as f1:
    with io.open(gen_marco_path_read) as fr:
        for line in enumerate(fr):
            #tmp_gen = line.split(',')[1].split('\t')[0].replace(' "','').replace(" '",'').replace('")','').replace("')",'') #for temp in list(filter(None,re.split("\n",line.strip())))]
            line = str(line).replace('\\n','')
            f1.write(f"text:{line}\tlabels:__ok__\tepisode_done:True\n")
