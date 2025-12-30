import json

def load_jsonl_submission(filepath):
    preds = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    preds[obj['eval_id']] = obj['topk'][0] if obj['topk'] else None
                except:
                    continue
    return preds

v5_preds = load_jsonl_submission('submission_bge_m3_sota_v5.csv')
best_preds = load_jsonl_submission('submission_best_9394.csv')

diffs = []
for eval_id in v5_preds:
    if v5_preds[eval_id] != best_preds.get(eval_id):
        diffs.append({
            'eval_id': eval_id,
            'v5_docid': v5_preds[eval_id],
            'best_docid': best_preds.get(eval_id)
        })

print(f"Total differences in Rank 1: {len(diffs)}")
for d in diffs:
    print(f"eval_id: {d['eval_id']}, v5: {d['v5_docid']}, best: {d['best_docid']}")

with open('submission_diffs.json', 'w', encoding='utf-8') as f:
    json.dump(diffs, f, ensure_ascii=False, indent=2)
