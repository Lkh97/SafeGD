import json

with open('/home/leila/LOT_Neurips_2023/Data_models/BBB/Data/dialogue_safety/multi_turn_safety.json') as f:
    d = json.load(f)

train = d["train"]
#print(train.type)
valid = d["valid"]
test = d["test"]
#print(f"{train[0:10]}\n")

with open("/home/leila/LOT_Neurips_2023/Data_models/BBB/Data/dialogue_safety/valid.txt", "w", encoding="UTF8") as f:
    for i in range(len(valid)):
        row_dict = valid[i]
        context = '\\n'.join(row_dict["text"].split('\n'))
        # print(context)
        # input()
        line = "text:"+context.strip()+"\t"+"labels:"+str(row_dict["labels"][0])+"\t"+"episode_done:True"+"\t"+"speaker_to_eval:human"+"\t"+"bot_persona:your persona:"
        f.write(f"{line}\n")

with open("/home/leila/LOT_Neurips_2023/Data_models/BBB/Data/dialogue_safety/test.txt", "w", encoding="UTF8") as f:
    for i in range(len(test)):
        row_dict = test[i]
        context = '\\n'.join(row_dict["text"].split('\n'))
        line = "text:"+context.strip()+"\t"+"labels:"+str(row_dict["labels"][0])+"\t"+"episode_done:True"+"\t"+"speaker_to_eval:human"+"\t"+"bot_persona:your persona:"
        f.write(f"{line}\n")

with open("/home/leila/LOT_Neurips_2023/Data_models/BBB/Data/dialogue_safety/train.txt", "w", encoding="UTF8") as f:
    for i in range(len(train)):
        row_dict = train[i]
        context = '\\n'.join(row_dict["text"].split('\n'))
        line = "text:"+context.strip()+"\t"+"labels:"+str(row_dict["labels"][0])+"\t"+"episode_done:True"+"\t"+"speaker_to_eval:human"+"\t"+"bot_persona:your persona:"
        f.write(f"{line}\n")