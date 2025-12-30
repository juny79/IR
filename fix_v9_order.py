import json

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            data[obj['eval_id']] = obj
    return data

# Load the original order from eval.jsonl
eval_order = []
with open('/root/IR/data/eval.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        eval_order.append(json.loads(line)['eval_id'])

print("Loading v5 and best_9394...")
v5_data = load_jsonl('submission_bge_m3_sota_v5.csv')
best_data = load_jsonl('submission_best_9394.csv')

with open('solar_diff_analysis.json', 'r') as f:
    solar_diffs = json.load(f)

v10_data = v5_data.copy()
changes = 0

for item in solar_diffs:
    qid = item['eval_id']
    pref = item['solar_choice']
    
    if qid in v10_data:
        v5_topk = v10_data[qid]['topk']
        best_topk = best_data[qid]['topk']
        
        if pref == 'best':
            best_r1 = best_topk[0]
            if best_r1 in v5_topk:
                v5_topk.remove(best_r1)
            new_topk = [best_r1] + v5_topk
            v10_data[qid]['topk'] = new_topk[:5]
            changes += 1

output_file = 'submission_v9_sota.csv'
with open(output_file, 'w', encoding='utf-8') as f:
    for qid in eval_order:
        if qid in v10_data:
            f.write(json.dumps(v10_data[qid], ensure_ascii=False) + '\n')

print(f"✅ Re-generated {output_file} in original eval.jsonl order.")
