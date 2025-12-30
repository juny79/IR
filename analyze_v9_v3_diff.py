import json
from pathlib import Path

def load_submission(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            data[obj['eval_id']] = obj
    return data

def analyze_diff(path_v9, path_v3):
    v9 = load_submission(path_v9)
    v3 = load_submission(path_v3)
    
    diff_ids = []
    for eid in v9:
        if eid in v3:
            tk9 = v9[eid].get('topk', [])
            tk3 = v3[eid].get('topk', [])
            
            t1_9 = tk9[0] if tk9 else None
            t1_3 = tk3[0] if tk3 else None
            
            if t1_9 != t1_3:
                diff_ids.append(eid)
    
    print(f"Total differences in Top-1: {len(diff_ids)}")
    
    # Sample some differences
    for eid in diff_ids[:10]:
        print(f"\n--- ID {eid} ---")
        print(f"Query: {v9[eid]['standalone_query']}")
        tk9 = v9[eid].get('topk', [])
        tk3 = v3[eid].get('topk', [])
        print(f"v9 Top-1: {tk9[0] if tk9 else 'None'}")
        print(f"v3 Top-1: {tk3[0] if tk3 else 'None'}")
        
        if tk9 and tk9[0] in tk3:
            rank = tk3.index(tk9[0]) + 1
            print(f"v9's Top-1 is at Rank {rank} in v3")
        else:
            print("v9's Top-1 is NOT in v3's Top-5")

if __name__ == "__main__":
    analyze_diff('/root/IR/submission_v9_sota.csv', '/root/IR/submission_v3_final.csv')
