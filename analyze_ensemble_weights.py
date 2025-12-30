import json
import numpy as np
from pathlib import Path

def load_results(path):
    results = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            results[str(obj['eval_id'])] = obj.get('topk', [])
    return results

def calculate_map(gt_dict, pred_dict, k=5):
    aps = []
    for eid, gt_topk in gt_dict.items():
        if not gt_topk: continue # Skip empty gating
        if eid not in pred_dict:
            aps.append(0)
            continue
        
        pred_topk = pred_dict[eid][:k]
        gt_set = set([gt_topk[0]]) # SOTA의 Top-1을 유일한 정답으로 가정 (보수적 평가)
        
        precision_at_i = []
        hits = 0
        for i, p in enumerate(pred_topk):
            if p in gt_set:
                hits += 1
                precision_at_i.append(hits / (i + 1))
        
        if not precision_at_i:
            aps.append(0)
        else:
            aps.append(np.mean(precision_at_i))
    return np.mean(aps)

def rrf_ensemble(base_results, ft_results, w_ft, k_rrf=60):
    w_base = 1.0 - w_ft
    ensemble_results = {}
    
    all_ids = set(base_results.keys()) | set(ft_results.keys())
    
    for eid in all_ids:
        scores = {}
        # Base ranks
        for rank, docid in enumerate(base_results.get(eid, []), 1):
            scores[docid] = scores.get(docid, 0) + w_base * (1.0 / (k_rrf + rank))
        # FT ranks
        for rank, docid in enumerate(ft_results.get(eid, []), 1):
            scores[docid] = scores.get(docid, 0) + w_ft * (1.0 / (k_rrf + rank))
            
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ensemble_results[eid] = [doc for doc, score in sorted_docs[:5]]
        
    return ensemble_results

# 데이터 로드
sota_v9 = load_results('submission_v9_sota.csv')
finetuned = load_results('submission_bge_m3_finetuned_v9.csv')

print(f"{'FT Weight':<10} | {'Proxy MAP (vs SOTA v9)':<20}")
print("-" * 35)

best_w = 0
max_map = 0

for w in np.arange(0.0, 1.1, 0.1):
    ens = rrf_ensemble(sota_v9, finetuned, w)
    m = calculate_map(sota_v9, ens)
    print(f"{w:.1f}        | {m:.4f}")
    if m > max_map:
        max_map = m
        best_w = w

print("-" * 35)
print(f"Best Finetuned Weight: {best_w:.1f} (Proxy MAP: {max_map:.4f})")
