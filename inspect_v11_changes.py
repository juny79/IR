import json

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                data[obj['eval_id']] = obj['topk']
            except: continue
    return data

v5 = load_jsonl('submission_bge_m3_sota_v5.csv')
best = load_jsonl('submission_best_9394.csv')
v9 = load_jsonl('submission_v9_sota.csv')
v11 = load_jsonl('submission_v11_sota.csv')

docs_dict = {}
with open('/root/IR/data/documents.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        docs_dict[obj['docid']] = obj['content']

eval_queries = {}
with open('/root/IR/data/eval.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        qid = obj['eval_id']
        if isinstance(obj['msg'], list):
            query = obj['msg'][-1]['content']
        else:
            query = obj['msg']
        eval_queries[qid] = query

# Find cases where v11 differs from v9
diff_ids = []
for qid in v11:
    if qid in v9 and v11[qid] and v9[qid]:
        if v11[qid][0] != v9[qid][0]:
            diff_ids.append(qid)

print(f"Total Rank 1 differences (v11 vs v9): {len(diff_ids)}")

# Sample 5 cases for detailed inspection
for qid in diff_ids[:5]:
    print(f"\n--- [ID {qid}] ---")
    print(f"Query: {eval_queries[qid]}")
    print(f"v5 Rank 1: {v5[qid][0] if qid in v5 else 'N/A'}")
    print(f"best Rank 1: {best[qid][0] if qid in best else 'N/A'}")
    print(f"v9 Rank 1: {v9[qid][0]}")
    print(f"v11 Rank 1: {v11[qid][0]}")
    
    print("\n[v9 Content]:")
    print(docs_dict.get(v9[qid][0], "Not found")[:200] + "...")
    print("\n[v11 Content]:")
    print(docs_dict.get(v11[qid][0], "Not found")[:200] + "...")
