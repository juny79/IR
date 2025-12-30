import json

def load_jsonl_rank1(filepath):
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

v5 = load_jsonl_rank1('submission_bge_m3_sota_v5.csv')
best_9394 = load_jsonl_rank1('submission_best_9394.csv')
v9_sota = load_jsonl_rank1('submission_v9_sota.csv')

# 1. Overall match count
match_v5 = sum(1 for eid in v9_sota if v9_sota[eid] == v5.get(eid))
match_best = sum(1 for eid in v9_sota if v9_sota[eid] == best_9394.get(eid))

print(f"v9_sota matches v5 Rank 1: {match_v5}/220")
print(f"v9_sota matches best_9394 Rank 1: {match_best}/220")

# 2. Analyze the 18 critical differences
with open('solar_diff_analysis.json', 'r', encoding='utf-8') as f:
    solar_analysis = json.load(f)

v9_wins = 0
v9_follows_solar = 0
v9_follows_v5 = 0
v9_follows_best = 0

print("\n--- Analysis of 18 Critical Differences ---")
for item in solar_analysis:
    eid = item['eval_id']
    v5_doc = item['v5_docid']
    best_doc = item['best_docid']
    solar_choice = item['solar_choice']
    v9_doc = v9_sota.get(eid)
    
    choice_str = ""
    if v9_doc == v5_doc:
        v9_follows_v5 += 1
        choice_str = "Followed v5"
    elif v9_doc == best_doc:
        v9_follows_best += 1
        choice_str = "Followed best_9394"
    else:
        choice_str = "Chose something else"
        
    solar_match = "YES" if (solar_choice == 'v5' and v9_doc == v5_doc) or (solar_choice == 'best' and v9_doc == best_doc) else "NO"
    if solar_match == "YES":
        v9_follows_solar += 1
        
    print(f"ID {eid}: Solar preferred {solar_choice} | v9 {choice_str} | Match Solar: {solar_match}")

print(f"\nv9 followed Solar's preference in {v9_follows_solar}/18 cases.")
print(f"v9 followed v5 in {v9_follows_v5}/18 cases.")
print(f"v9 followed best_9394 in {v9_follows_best}/18 cases.")
