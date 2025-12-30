import json
import torch
from FlagEmbedding import FlagReranker

def load_jsonl(p):
    d = {}
    with open(p, 'r') as f:
        for l in f:
            if not l.strip(): continue
            o = json.loads(l)
            d[o['eval_id']] = o
    return d

def load_docs(p):
    d = {}
    with open(p, 'r') as f:
        for l in f:
            if not l.strip(): continue
            o = json.loads(l)
            d[o['docid']] = o['content']
    return d

def main():
    v9 = load_jsonl('submission_v9_sota.csv')
    v2 = load_jsonl('submission_v2_final_rerank.csv')
    docs = load_docs('data/documents.jsonl')
    eval_data = load_jsonl('data/eval.jsonl')

    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    diff_ids = []
    for eid in v9:
        v9_t1 = v9[eid]['topk'][0] if v9[eid]['topk'] else None
        v2_t1 = v2[eid]['topk'][0] if v2.get(eid) and v2[eid]['topk'] else None
        if v9_t1 != v2_t1:
            diff_ids.append(eid)

    print(f"Found {len(diff_ids)} differences. Starting tie-break...")

    final_topk = {}
    for eid in v9:
        if eid not in diff_ids:
            final_topk[eid] = v9[eid]['topk']
            continue
        
        # Tie-break for diff_ids
        query = v9[eid].get('standalone_query', eval_data[eid]['msg'][-1]['content'])
        doc_v9_id = v9[eid]['topk'][0]
        doc_v2_id = v2[eid]['topk'][0]
        
        doc_v9_text = docs.get(doc_v9_id, "")
        doc_v2_text = docs.get(doc_v2_id, "")
        
        pairs = [[query, doc_v9_text], [query, doc_v2_text]]
        scores = reranker.compute_score(pairs)
        
        print(f"ID {eid}: V9({scores[0]:.4f}) vs V2({scores[1]:.4f})")
        
        if scores[1] > scores[0]:
            print(f"  -> Switching to V2 for ID {eid}")
            # We take V2's topk as the base for this ID
            final_topk[eid] = v2[eid]['topk']
        else:
            print(f"  -> Keeping V9 for ID {eid}")
            final_topk[eid] = v9[eid]['topk']

    # Save final submission
    with open('submission_precision_strike.csv', 'w') as f:
        for eid in sorted(v9.keys()):
            out = v9[eid].copy()
            out['topk'] = final_topk[eid]
            f.write(json.dumps(out, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
