import json

def load_jsonl_submission(filepath):
    preds = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    preds[obj['eval_id']] = obj['topk']
                except:
                    continue
    return preds

v5_topk = load_jsonl_submission('submission_bge_m3_sota_v5.csv')
best_topk = load_jsonl_submission('submission_best_9394.csv')

with open('solar_diff_analysis.json', 'r', encoding='utf-8') as f:
    analysis = json.load(f)

for item in analysis:
    eval_id = item['eval_id']
    v5_rank1 = item['v5_docid']
    best_rank1 = item['best_docid']
    solar_choice = item['solar_choice']
    
    if solar_choice == 'best':
        # Check if best_rank1 is in v5's topk
        v5_list = v5_topk.get(eval_id, [])
        if best_rank1 in v5_list:
            rank = v5_list.index(best_rank1) + 1
            print(f"ID {eval_id}: Best Rank 1 was at Rank {rank} in v5. (Reranking/LLM issue)")
        else:
            print(f"ID {eval_id}: Best Rank 1 was NOT in v5 topk. (Retrieval issue)")
    else:
        # Check if v5_rank1 is in best's topk
        best_list = best_topk.get(eval_id, [])
        if v5_rank1 in best_list:
            rank = best_list.index(v5_rank1) + 1
            print(f"ID {eval_id}: v5 Rank 1 was at Rank {rank} in Best. (Reranking/LLM issue)")
        else:
            print(f"ID {eval_id}: v5 Rank 1 was NOT in Best topk. (Retrieval issue)")
