import json

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            data[obj['eval_id']] = obj
    return data

# Load v9_sota (which has 0.9409)
v9_data = load_jsonl('submission_v9_sota.csv')

# Load the 13 new improvements
with open('solar_low_gap_improvements.json', 'r') as f:
    improvements = json.load(f)

v11_data = v9_data.copy()
applied_count = 0

for imp in improvements:
    qid = imp['eval_id']
    new_r1 = imp['new_rank1']
    
    if qid in v11_data:
        topk = v11_data[qid]['topk']
        if new_r1 in topk:
            topk.remove(new_r1)
        new_topk = [new_r1] + topk
        v11_data[qid]['topk'] = new_topk[:5]
        applied_count += 1
        print(f"ID {qid}: Applied new Rank 1 ({new_r1})")

# Save in original order
eval_order = []
with open('/root/IR/data/eval.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        eval_order.append(json.loads(line)['eval_id'])

output_file = 'submission_v11_sota.csv'
with open(output_file, 'w', encoding='utf-8') as f:
    for qid in eval_order:
        if qid in v11_data:
            f.write(json.dumps(v11_data[qid], ensure_ascii=False) + '\n')

print(f"\n✅ Created {output_file} with {applied_count} additional improvements.")
