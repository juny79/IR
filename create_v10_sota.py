import json
import os
from dotenv import load_dotenv
from models.solar_client import SolarClient

load_dotenv()

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            data[obj['eval_id']] = obj
    return data

print("Loading v5 and best_9394...")
v5_data = load_jsonl('submission_bge_m3_sota_v5.csv')
best_data = load_jsonl('submission_best_9394.csv')

with open('solar_diff_analysis.json', 'r') as f:
    solar_diffs = json.load(f)

# We will create v10 by taking v5 and applying Solar's preference for the 18 diffs.
# For the 18 diffs, we already have Solar's choice (v5 or best).
# If Solar chose 'best', we take the topk[0] from best_9394 and put it at Rank 1 for v5.

v10_data = v5_data.copy()
changes = 0

for item in solar_diffs:
    qid = item['eval_id']
    pref = item['solar_choice']
    
    if qid in v10_data:
        v5_topk = v10_data[qid]['topk']
        best_topk = best_data[qid]['topk']
        
        if pref == 'best':
            # Move best's Rank 1 to the front of v5's list
            best_r1 = best_topk[0]
            if best_r1 in v5_topk:
                v5_topk.remove(best_r1)
            new_topk = [best_r1] + v5_topk
            v10_data[qid]['topk'] = new_topk[:5]
            changes += 1
            print(f"ID {qid}: Switched to best_9394 Rank 1 ({best_r1})")
        else:
            print(f"ID {qid}: Kept v5 Rank 1 ({v5_topk[0]})")

output_file = 'submission_v10_sota.csv'
with open(output_file, 'w', encoding='utf-8') as f:
    for qid in sorted(v10_data.keys()):
        f.write(json.dumps(v10_data[qid], ensure_ascii=False) + '\n')

print(f"\n✅ Created {output_file} with {changes} changes from v5.")
