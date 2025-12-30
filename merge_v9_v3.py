import json
from collections import defaultdict

def weighted_rrf(submissions, weights, k=60, top_n=5):
    """
    submissions: list of dicts {eval_id: {topk: [...]}}
    weights: list of weights for each submission
    """
    all_ids = set()
    for s in submissions:
        all_ids.update(s.keys())
    
    final_results = {}
    
    for eid in sorted(all_ids, key=lambda x: int(x)):
        scores = defaultdict(float)
        
        for sub, weight in zip(submissions, weights):
            if eid in sub:
                tk = sub[eid].get('topk', [])
                for rank, docid in enumerate(tk):
                    scores[docid] += weight * (1.0 / (k + rank + 1))
        
        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        topk = [docid for docid, score in sorted_docs[:top_n]]
        
        # Use metadata from the first submission that has it
        meta = {}
        for sub in submissions:
            if eid in sub:
                meta['standalone_query'] = sub[eid].get('standalone_query', '')
                meta['answer'] = sub[eid].get('answer', '')
                break
        
        final_results[eid] = {
            'eval_id': int(eid),
            'standalone_query': meta.get('standalone_query', ''),
            'topk': topk,
            'answer': meta.get('answer', '')
        }
    
    return final_results

def load_submission(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            data[str(obj['eval_id'])] = obj
    return data

if __name__ == "__main__":
    v9 = load_submission('/root/IR/submission_v9_sota.csv')
    v3 = load_submission('/root/IR/submission_v3_final.csv')
    
    # 0.6 : 0.4 Weighted RRF
    merged = weighted_rrf([v9, v3], [0.6, 0.4])
    
    output_path = '/root/IR/submission_v3_v9_rrf_64.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        for eid in sorted(merged.keys(), key=lambda x: int(x)):
            f.write(json.dumps(merged[eid], ensure_ascii=False) + '\n')
    
    print(f"Merged submission saved to {output_path}")
