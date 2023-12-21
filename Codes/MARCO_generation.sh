
#!/bin/bash
#i=0
#fileListName=("CN_new_tokenizer_1_0.02_0.005_0_new_tokenizer.pth" "toxic_expert_7_new_tokenizer_TS.pth" "CN_new_tokenizer_10000.0_3_6_0_new_tokenizer.pth" "CN_new_tokenizer_1_1_3_0_new_tokenizer.pth")
gen_path='/home/leila/LOT_Neurips_2023/Codes/MarcoDetoxification/data/dexp_outputs/manual/masked_thresh1.5/aa0.5_ae2.5_ab1.0_basebase_antiantie_expertexper_temp1.0_sampleF_topk50_reppenalty1.0_filterp1.0_maxlength128_topp1.0/gen_formatted.txt'
div_results_dir='/home/leila/LOT_Neurips_2023/Data_models/BAD/Diversity/MARCO'
tox_results_dir='/home/leila/LOT_Neurips_2023/Data_models/BAD/toxicity/MARCO'

#python /home/leila/LOT_Neurips_2023/Codes/MarcoDetoxification/rewrite/rewrite_example.py

python /home/leila/LOT_Neurips_2023/Codes/MarcoDetoxification/gen_format.py

#i=$((i+1))
#filename="bb__10_3_0.015_9.pth"
filename='MARCO_BART.pth'
llama_checkpoint='LlAMa.pth'
echo "####################################  This is the $filename checkpoint  #########################"
#myfilesize=$(wc -c "/home/leila/compNet/CN_filtered_newJS/parameters/$filename" | awk '{print $1}')
#echo "##########################   Filename is $filename and filesize is $myfilesize################"
export filename
#seed=$[$RANDOM % 100 + 1]
#echo $seed
echo "###################################  print start of generation ###############################"
#p ython /home/leila/compNet_AAAI/validation/compNet/evaluate.py --data_folder /home/leila/compNet_AAAI/validation/data/bot_adversarial_dialogue_datasets_with_persona --split valid --save_folder /home/leila/compNet_AAAI/results/bb/parameters/bb_filtered_clean/ --checkpoint "$filename"  --ppl --toxicity --top_k 50 --top_p 0.90 #--seed $seed 
python /home/leila/LOT_Neurips_2023/Codes/MARCO.py \
--checkpoint $filename \
--llama_checkpoint $llama_checkpoint \
--ppl \
--gen_path $gen_path \
--ppl_path /home/leila/LOT_Neurips_2023/Data_models/BAD/ppl/MARCO/ 


echo "###################################  end of generation #################################"
# #export filename
# #python test.py
##OF=/var/my-backup-$(date +%Y%m%d).tgz
filename_wo_ext=$(basename $filename .pth)
echo "################################### start of toxicity evaluation #######################"
#######parlai detect_offensive --task fromfile:parlaiformat --fromfile_datapath /home/leila/compNet/Baselines/Robust-Agents/Defenses/LERG-main/results_BAD/diloGPT/last_run/Defender_dialoGPT_contx_gen_tokenize.txt --display-examples True --safety classifier > /home/leila/compNet/Baselines/Robust-Agents/Defenses/LERG-main/results_BAD/diloGPT/last_run/metrics/toxicity_Defender.txt
parlai detect_offensive --task fromfile:parlaiformat --fromfile_datapath $gen_path  --display-examples True --safety classifier > tox_results_dir/toxicity_$filename_wo_ext.txt
echo "###################################  end of toxicity ##################################"

echo "###################################  Beginning of Diversity ##################################"
python /home/leila/LOT_Neurips_2023/Codes/automatic_eval_diversity.py \
--test_set 'BAD.txt' \
--test_results_dir $div_results_dir \
--model_name $filename \
--generation $gen_path




# # # #### Get toxicity value and append to perplexity csv file #####
# # line=$(tail -n 1 "/home/leila/compNet/results/valid_toxicity/toxicity.txt")
# # IFS=$' '
# # tmp=($line)
# # tmp0=${tmp[0]}
# # echo "${tmp[0]}"
# # export tmp0
# # tail -1 /home/leila/compNet/results/valid_perplexity/loss_ppl.csv | awk -F, '{$4=ENVIRON["tmp0"]; print}'  >> /home/leila/compNet/results/test.csv    










