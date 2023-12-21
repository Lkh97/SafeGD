
import json
import csv

split = 'test.json'
path = '/home/leila/LOT_Neurips_2023/Data/DiaSafety/DiaSafety_dataset'
f = open(path+'/'+split)
data = json.load(f)

#text:Hello, how are you doing	labels:__ok__	episode_done:True	speaker_to_eval:human	bot_persona:your persona: i am a yoga instructor.\nyour persona: i like to write poetry in my spare time.

with open('/home/leila/LOT_Neurips_2023/Data/DiaSafety/DiaSafety_dataset/safety_data.txt', 'w', encoding='UTF8') as f1:
    for i in data:
        line = '\n'+'text:'+i['context']+'\\n'+i['response'].strip()+'\t'+'labels:__ok__'+'\t'+'episode_done:True'+'\t'+'speaker_to_eval:human'+'\t'+'bot_persona:your persona:'
        f1.writelines(line)
        # writer.writerow(line)
        # print(i['context'])
        # print('\n')
        # input(i['response'])