import pandas as pd

import json

def get_rank1(file):
    rank1 = {}
    with open(file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                rank1[data['eval_id']] = data['topk'][0]
            except:
                continue
    return rank1

v9_r1 = get_rank1('submission_v9_sota.csv')
v15_r1 = get_rank1('submission_v15_sota.csv')

diff_ids = [qid for qid in v9_r1 if v9_r1[qid] != v15_r1.get(qid)]
print("Differences in Rank 1:")
for qid in sorted(diff_ids):
    print(f"ID {qid}: v9={v9_r1[qid]} -> v15={v15_r1[qid]}")

print(f"\nTotal differences: {len(diff_ids)}")
