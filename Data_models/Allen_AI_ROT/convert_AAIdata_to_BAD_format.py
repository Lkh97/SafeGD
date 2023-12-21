
import json
import debugpy
import pyarrow as pa
import pandas as pd

# debugpy.listen(5678)
# print("socket waitiiiiiiiiiing for client")
# debugpy.wait_for_client()   

with open("/home/leila/LOT_Neurips_2023/Data_models/Allen_AI_ROT/json-validation.arrow", 'rb') as f:
  table = pa.RecordBatchStreamReader(f).read_all()
train_df = pd.DataFrame(table.to_pandas())

with open("/home/leila/LOT_Neurips_2023/Data_models/Allen_AI_ROT/valid.txt", "w", encoding="UTF8") as f:
    tmp = ""
    counter=0
    for idx, row in train_df.iterrows():
        counter+=1
        # print(f"\n{row}\n")
        # print(row["episode_done"])
        #if not bool(row["episode_done"]):
        if counter == 1:
          tmp += row["context"]+"\\n"+row["response"]
        else:
          tmp += "\\n"+row["context"]+"\\n"+row["response"]
        line = "text:"+tmp.strip()+"\t"+"labels:__ok__"+"\t"+"episode_done:True"+"\t"+"speaker_to_eval:"+"\t"+"bot_persona:your persona:"
        f.write(f"{line}\n")
        if bool(row["episode_done"]):
            tmp="" 
            counter = 0 




