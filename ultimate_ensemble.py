import json
from collections import defaultdict

def load_submission(path):
    data = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                data[str(obj['eval_id'])] = obj
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return data

def ultimate_ensemble(submission_paths, weights, k=60, top_n=5):
    submissions = [load_submission(p) for p in submission_paths]
    
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
                    # RRF formula
                    scores[docid] += weight * (1.0 / (k + rank + 1))
        
        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        topk = [docid for docid, score in sorted_docs[:top_n]]
        
        # Metadata from the best available source (v9)
        meta = submissions[0].get(eid, {})
        if not meta:
            for s in submissions:
                if eid in s:
                    meta = s[eid]
                    break
        
        final_results[eid] = {
            'eval_id': int(eid),
            'standalone_query': meta.get('standalone_query', ''),
            'topk': topk,
            'answer': meta.get('answer', '')
        }
    
    return final_results

if __name__ == "__main__":
    paths = [
        '/root/IR/submission_v9_sota.csv',
        '/root/IR/submission_v17_conservative_from_v9_20251227_145004.csv',
        '/root/IR/submission_best_9394.csv',
        '/root/IR/submission_v3_final.csv'
    ]
    weights = [0.5, 0.2, 0.2, 0.1]
    
    merged = ultimate_ensemble(paths, weights)
    
    output_path = '/root/IR/submission_ultimate_ensemble_v1.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        for eid in sorted(merged.keys(), key=lambda x: int(x)):
            f.write(json.dumps(merged[eid], ensure_ascii=False) + '\n')
    
    print(f"Ultimate ensemble saved to {output_path}")
