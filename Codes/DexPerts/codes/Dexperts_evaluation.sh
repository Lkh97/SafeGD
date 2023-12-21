
#!/bin/bash
#i=0
#fileListName=("CN_new_tokenizer_1_0.02_0.005_0_new_tokenizer.pth" "toxic_expert_7_new_tokenizer_TS.pth" "CN_new_tokenizer_10000.0_3_6_0_new_tokenizer.pth" "CN_new_tokenizer_1_1_3_0_new_tokenizer.pth")
gen_path="/home/lkhalatbari/NAACL2023_Experiments/Bseline_results/test/Dexperts/"
diversity_path="/home/lkhalatbari/NAACL2023_Experiments/Bseline_results/test/Dexperts/"
ppl_path="/home/lkhalatbari/NAACL2023_Experiments/Bseline_results/test/Dexperts/"
toxicity_path="/home/lkhalatbari/NAACL2023_Experiments/Bseline_results/test/Dexperts/"
dataset_path=$1 
#"/home/lkhalatbari/compNet_AAAI/validation/data/bot_adversarial_dialogue_datasets_with_persona"
#'/home/lkhalatbari/LOT_Neurips_2023/Data_models/BBB/Data/dialogue_safety'
model_name_and_path="blenderbot-400M-distill"
dataset=$2 
#"BAD"
#i=$((i+1))
#filename="bb__10_3_0.015_9.pth"
filename='Dexperts.pth'
echo "####################################  This is the $filename checkpoint  #########################"
#myfilesize=$(wc -c "/home/leila/compNet/CN_filtered_newJS/parameters/$filename" | awk '{print $1}')
#echo "##########################   Filename is $filename and filesize is $myfilesize################"
export filename
#seed=$[$RANDOM % 100 + 1]
#echo $seed
echo "###################################  print start of generation ###############################"
# #p ython /home/leila/compNet_AAAI/validation/compNet/evaluate.py --data_folder /home/leila/compNet_AAAI/validation/data/bot_adversarial_dialogue_datasets_with_persona --split valid --save_folder /home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_clean/ --checkpoint "$filename"  --ppl --toxicity --top_k 50 --top_p 0.90 #--seed $seed 
python /home/lkhalatbari/LOT_Neurips_2023/Codes/DexPerts/codes/DexPerts.py \
--checkpoint $filename \
--ppl \
--data_folder $dataset_path \
--toxicity \
--dataset $dataset \
--split test.txt \
--ppl_path  $ppl_path  \
--gen_path  $gen_path \
--context_gen_path $gen_path #/home/leila/LOT_Neurips_2023/Data_models/BAD/Generations/Dexperts_ensemble_passed/ 

# --top_k 50\
# --top_p 0.90\
#--seed $seed 
echo "###################################  end of generation #################################"
# #export filename
# #python test.py
##OF=/var/my-backup-$(date +%Y%m%d).tgz
filename_wo_ext=$(basename $filename .pth)

for seed in  0 1 42 12345 
do
    echo "################################### start of toxicity evaluation for seed: ${seed} #######################"
    #parlai detect_offensive --task fromfile:parlaiformat --fromfile_datapath /home/leila/compNet/Baselines/Robust-Agents/Defenses/LERG-main/results_BAD/diloGPT/last_run/Defender_dialoGPT_contx_gen_tokenize.txt --display-examples True --safety classifier > /home/leila/compNet/Baselines/Robust-Agents/Defenses/LERG-main/results_BAD/diloGPT/last_run/metrics/toxicity_Defender.txt
    parlai detect_offensive --task fromfile:parlaiformat --fromfile_datapath  ${gen_path}gen_${filename_wo_ext}_${model_name_and_path}_${dataset}_seed_${seed}.txt   --display-examples True --safety classifier > ${toxicity_path}toxicity_${filename_wo_ext}_${model_name_and_path}_${dataset}_seed_${seed}.txt

    line=$(tail -n 1 "${toxicity_path}toxicity_${filename_wo_ext}_${model_name_and_path}_${dataset}_seed_${seed}.txt")
    IFS=$' '
    tmp=($line)
    tmp0=${tmp[0]}
    echo "${tmp[0]}" >> ${toxicity_path}Dexperts_toxicity_${dataset}_seed_$seed.csv

    echo "###################################  end of toxicity ##################################"

    echo "###################################  Beginning of Diversity ##################################"
    python  /home/lkhalatbari/LOT_Neurips_2023/Codes/automatic_eval_diversity.py \
    --test_set 'BAD.txt' \
    --test_results_dir $diversity_path \
    --model_name ${filename_wo_ext}_${model_name_and_path}_seed_$seed \
    --generation ${gen_path}gen_${filename_wo_ext}_${model_name_and_path}_${dataset}_seed_$seed.txt
done
# #### Get toxicity value and append to perplexity csv file #####
#export tmp0
#tail -1 /home/leila/compNet/results/valid_perplexity/loss_ppl.csv | awk -F, '{$4=ENVIRON["tmp0"]; print}'  >> /home/leila/compNet/results/test.csv    










