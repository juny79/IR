import json
from sentence_transformers import CrossEncoder
import numpy as np

RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device='cuda')

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if 'eval_id' in obj: data[obj['eval_id']] = obj
                elif 'docid' in obj: data[obj['docid']] = obj
    return data

queries = load_jsonl('data/eval.jsonl')
docs = load_jsonl('data/documents.jsonl')

with open('submission_diffs.json', 'r', encoding='utf-8') as f:
    diffs = json.load(f)

for d in diffs:
    eval_id = d['eval_id']
    q = queries[eval_id]['msg'][0]['content']
    doc0 = docs[d['v5_docid']]['content']
    doc1 = docs[d['best_docid']]['content']
    
    scores = reranker.predict([[q, doc0], [q, doc1]])
    gap = abs(scores[0] - scores[1])
    print(f"ID {eval_id}: v5_score={scores[0]:.4f}, best_score={scores[1]:.4f}, gap={gap:.4f}")
