import json

def load_jsonl(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Try JSON first
            try:
                item = json.loads(line)
                qid = item.get('eval_id')
                if qid is None: qid = item.get('id')
                topk = item.get('topk')
                if topk is None: topk = item.get('top_k')
                
                if qid is not None and topk is not None:
                    if isinstance(topk, str):
                        topk = topk.split()
                    data[int(qid)] = topk
                    continue
            except json.JSONDecodeError:
                pass
            
            # Try CSV
            if line.startswith('id,top_k'): continue
            parts = line.split(',', 1)
            if len(parts) == 2:
                try:
                    qid = int(parts[0])
                    topk = parts[1].strip().split()
                    data[qid] = topk
                except ValueError:
                    pass
    return data

v5_data = load_jsonl('submission_bge_m3_sota_v5.csv')
best_data = load_jsonl('submission_best_9394.csv')
v9_data = load_jsonl('submission_v9_sota.csv')

with open('solar_diff_analysis.json', 'r') as f:
    solar_analysis = json.load(f)

diff_ids = [int(k) for k in solar_analysis.keys()]

matches_v5 = 0
matches_best = 0
total = 0

for qid in v9_data:
    if qid in v5_data:
        total += 1
        if v9_data[qid][0] == v5_data[qid][0]:
            matches_v5 += 1
        if qid in best_data and v9_data[qid][0] == best_data[qid][0]:
            matches_best += 1

print(f"v9_sota matches v5 Rank 1: {matches_v5}/{total}")
print(f"v9_sota matches best_9394 Rank 1: {matches_best}/{total}")

print("\n--- Analysis of 18 Critical Differences ---")
solar_matches = 0
v5_matches_in_diff = 0
best_matches_in_diff = 0
other_matches = 0

for qid in diff_ids:
    v5_top = v5_data[qid][0] if qid in v5_data else None
    best_top = best_data[qid][0] if qid in best_data else None
    v9_top = v9_data[qid][0] if qid in v9_data else None
    
    pref = solar_analysis[str(qid)]['preferred']
    
    match_str = "Match Solar: "
    if (pref == 'v5' and v9_top == v5_top) or (pref == 'best' and v9_top == best_top):
        match_str += "YES"
        solar_matches += 1
    else:
        match_str += "NO"
    
    where = "Chose something else"
    if v9_top == v5_top:
        where = "Followed v5"
        v5_matches_in_diff += 1
    elif v9_top == best_top:
        where = "Followed best_9394"
        best_matches_in_diff += 1
    else:
        other_matches += 1
        
    print(f"ID {qid}: Solar preferred {pref} | v9 {where} | {match_str}")

print(f"\nv9 followed Solar's preference in {solar_matches}/{len(diff_ids)} cases.")
print(f"v9 followed v5 in {v5_matches_in_diff}/{len(diff_ids)} cases.")
print(f"v9 followed best_9394 in {best_matches_in_diff}/{len(diff_ids)} cases.")
print(f"v9 chose a third option in {other_matches}/{len(diff_ids)} cases.")
