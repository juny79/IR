import json
from pathlib import Path

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data[str(obj['eval_id'])] = obj['topk']
    return data

sota = load_jsonl('/root/IR/submission_bge_m3_sota.csv')
best = load_jsonl('/root/IR/submission_best_9273.csv')

common_ids = set(sota.keys()) & set(best.keys())
print(f"Common IDs: {len(common_ids)}")

top1_match = 0
top5_overlap = 0
total_top5 = 0

for eid in sorted(common_ids, key=int):
    s_topk = sota[eid]
    b_topk = best[eid]
    
    if not s_topk and not b_topk:
        top1_match += 1
        continue
    
    if s_topk and b_topk:
        if s_topk[0] == b_topk[0]:
            top1_match += 1
        else:
            print(f"Mismatch ID {eid}:")
            print(f"  SOTA Top-1: {s_topk[0]}")
            print(f"  BEST Top-1: {b_topk[0]}")
        
        s_set = set(s_topk[:5])
        b_set = set(b_topk[:5])
        top5_overlap += len(s_set & b_set)
        total_top5 += 5
    elif s_topk or b_topk:
        print(f"Mismatch ID {eid} (One is empty):")
        print(f"  SOTA: {'Empty' if not s_topk else 'Not Empty'}")
        print(f"  BEST: {'Empty' if not b_topk else 'Not Empty'}")

print(f"Top-1 Match: {top1_match}/{len(common_ids)} ({top1_match/len(common_ids)*100:.2f}%)")
print(f"Top-5 Overlap: {top5_overlap}/{total_top5} ({top5_overlap/total_top5*100:.2f}%)")
