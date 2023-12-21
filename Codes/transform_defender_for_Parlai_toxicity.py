
import io
generations=[]
gen_path = '/home/leila/LOT_Neurips_2023/Data/DiaSafety/results/generations/Defender/Defender_dialoGPT_contx_gen_tokenize_eos_speaker_safetyDataset.txt' #'/home/leila/compNet/Baselines/Robust-Agents/Defenses/LERG-main/results_BAD/diloGPT/last_run/Defender_dialoGPT_contx_gen_tokenize.txt' #(args.gen_path+"gen_"+str(args.checkpoint)+".txt").replace(".pth",'')
with io.open(gen_path) as fr:
    for line in fr:
        #tmp_gen = line.split(',')[1].split('\t')[0].replace(' "','').replace(" '",'').replace('")','').replace("')",'') #for temp in list(filter(None,re.split("\n",line.strip())))]
        tmp_gen_defender = (line.split(':')[1].split('\t')[0]).strip()
        print(tmp_gen_defender)
        generations.append(tmp_gen_defender)

with open('/home/leila/LOT_Neurips_2023/Data/DiaSafety/results/generations/Defender/defender_gen.txt', 'w', encoding='UTF8') as f1:
    for line in enumerate(generations):
        f1.write(f"text:{line}\tlabels:__ok__\tepisode_done:True\n")

            