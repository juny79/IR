import json
from pathlib import Path

def load_submission(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            # Ensure eval_id is string for consistent comparison
            data[str(obj['eval_id'])] = obj
    return data

def find_diffs(path1, path2):
    d1 = load_submission(path1)
    d2 = load_submission(path2)
    
    diffs = []
    common_ids = sorted(set(d1.keys()) & set(d2.keys()), key=lambda x: int(x))
    
    for eid in common_ids:
        tk1 = d1[eid].get('topk', [])
        tk2 = d2[eid].get('topk', [])
        
        t1_1 = tk1[0] if tk1 else None
        t1_2 = tk2[0] if tk2 else None
        
        if t1_1 != t1_2:
            diffs.append({
                'eval_id': eid,
                'query': d1[eid].get('standalone_query', ''),
                'v9_top1': t1_1,
                'v3_top1': t1_2,
                'v9_topk': tk1,
                'v3_topk': tk2
            })
    
    return diffs

if __name__ == "__main__":
    diffs = find_diffs('/root/IR/submission_v9_sota.csv', '/root/IR/submission_v3_final.csv')
    print(f"Found {len(diffs)} differences.")
    with open('/root/IR/v9_v3_diffs.json', 'w', encoding='utf-8') as f:
        json.dump(diffs, f, ensure_ascii=False, indent=2)
