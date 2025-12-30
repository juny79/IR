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
    # Load SOTA and V2
    v9 = load_jsonl('submission_v9_sota.csv')
    v2 = load_jsonl('submission_v2_final_rerank.csv')
    docs = load_docs('data/documents.jsonl')
    
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    # Identify differences in Top-1
    diff_ids = []
    for eid in v9:
        v9_t1 = v9[eid]['topk'][0] if v9[eid]['topk'] else None
        v2_t1 = v2[eid]['topk'][0] if v2.get(eid) and v2[eid]['topk'] else None
        if v9_t1 != v2_t1:
            diff_ids.append(eid)

    print(f"Found {len(diff_ids)} differences. Performing Ultimate Rerank on pooled Top-10...")

    final_results = {}
    for eid in sorted(v9.keys()):
        if eid not in diff_ids:
            final_results[eid] = v9[eid]
            continue
        
        # Pool Top-5 from both
        pool = []
        seen = set()
        for docid in v9[eid]['topk'][:5]:
            if docid not in seen:
                pool.append(docid)
                seen.add(docid)
        for docid in v2[eid]['topk'][:5]:
            if docid not in seen:
                pool.append(docid)
                seen.add(docid)
        
        query = v9[eid].get('standalone_query', "")
        if not query:
            # Fallback to last message if standalone_query is missing
            with open('data/eval.jsonl', 'r') as f:
                for line in f:
                    obj = json.loads(line)
                    if obj['eval_id'] == eid:
                        query = obj['msg'][-1]['content']
                        break
        
        # Rerank the pool
        pairs = [[query, docs.get(docid, "")] for docid in pool]
        scores = reranker.compute_score(pairs)
        
        # Sort pool by scores
        scored_pool = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
        new_topk = [x[0] for x in scored_pool]
        
        # Fill up to 5 if needed (though pool should have enough)
        if len(new_topk) < 5:
            for docid in v9[eid]['topk']:
                if docid not in new_topk:
                    new_topk.append(docid)
                if len(new_topk) >= 5: break
        
        print(f"ID {eid}: Top-1 changed from {v9[eid]['topk'][0][:8]} to {new_topk[0][:8]} (Score: {scored_pool[0][1]:.4f})")
        
        res = v9[eid].copy()
        res['topk'] = new_topk[:5]
        # Optional: Update answer if Top-1 changed? 
        # For now, keep V9's answer to be safe, or we could use V2's if V2's Top-1 won.
        if new_topk[0] == v2[eid]['topk'][0]:
            res['answer'] = v2[eid].get('answer', v9[eid]['answer'])
            
        final_results[eid] = res

    # Save final submission in the same order as v9
    with open('submission_ultimate_strike.csv', 'w') as f:
        # Get original order from v9 file
        original_order = []
        with open('submission_v9_sota.csv', 'r') as f_in:
            for line in f_in:
                if line.strip():
                    original_order.append(json.loads(line)['eval_id'])
        
        for eid in original_order:
            f.write(json.dumps(final_results[eid], ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
