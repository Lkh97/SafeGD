import pandas as pd
from collections import defaultdict
import collections

import debugpy
# Allow other computers to attach to debugpy at this IP address and port.
debugpy.listen(5678)
# Pause the program until a remote debugger is attached
debugpy.wait_for_client()

file = '/home/leila/LOT_Neurips_2023/Evaluation/Human/AMT_output/DEFENDER_SafeGD_AMT_results.csv'
df = pd.read_csv(file)

items_per_id = defaultdict(list)
for idx, row in df.iterrows():
    items_per_id[row['HITId']].append(row)

criteria = ['Engaging', 'Topical', 'Fluent', 'Toxic']

res_map = {
    0:'win',
    1:'lose',
    2:'tie'}

def get_result(score_list):
    if score_list == [1,1,1]:
        return 'tie'
    max_value = max(score_list)
    max_index = score_list.index(max_value)
    return res_map[max_index]

criteria_result = defaultdict(list)

for k, results in items_per_id.items():
    for cri in criteria:
        lose = sum([r[f'Answer.{cri}.1'] for r in results])
        win = sum([r[f'Answer.{cri}.2'] for r in results])
        tie = sum([r[f'Answer.{cri}.3'] for r in results])
        
        res = get_result([win, lose, tie])
        criteria_result[cri].append(res)
        
for cri in criteria:
    result_stat = collections.Counter(criteria_result[cri])
    if cri != 'Toxic':
        print(f"{cri}  :\twin {result_stat['win']/50} | tie {result_stat['tie']/50} | lose {result_stat['lose']/50}")
    else:
        print("toxic reversed already")
        print(f"ours is {result_stat['win']/50} of the time more toxic")
        print(f"comparison is {result_stat['lose']/50} of the time more toxic")