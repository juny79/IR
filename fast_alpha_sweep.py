import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from pathlib import Path

# Settings
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
SURGICAL_PATH = "/root/IR/submission_surgical_v1.csv"
BGE_M3_MODEL = 'BAAI/bge-m3'
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'
CACHE_DIR = "/root/IR/cache/bge_m3"

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    return data

def load_top1(path):
    d = {}
    for obj in load_jsonl(path):
        d[str(obj['eval_id'])] = obj['topk'][0] if obj.get('topk') else None
    return d

print("Loading data...")
eval_data = load_jsonl(EVAL_PATH)
surgical_top1 = load_top1(SURGICAL_PATH)
documents = load_jsonl(DOC_PATH)
doc_contents = [d["content"] for d in documents]
doc_ids = [d["docid"] for d in documents]

print("Loading models...")
model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device='cuda')

print("Loading cache...")
doc_dense_embs = np.load(os.path.join(CACHE_DIR, "doc_dense_embs.npy"))
with open(os.path.join(CACHE_DIR, "doc_sparse_embs.json"), 'r') as f:
    doc_sparse_embs = json.load(f)
index = faiss.read_index(os.path.join(CACHE_DIR, "bge_m3_dense.index"))

EMPTY_IDS = {276, 261, 283, 32, 94, 90, 220, 245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103}

def get_fast_hybrid_candidates(query_text, top_k=100):
    q_output = model.encode([query_text], return_dense=True, return_sparse=True, max_length=8192)
    q_dense = q_output['dense_vecs'][0].astype('float32')
    q_sparse = q_output['lexical_weights'][0]
    
    # Dense
    d_scores, d_indices = index.search(np.expand_dims(q_dense, 0), top_k)
    d_indices = d_indices[0]
    d_scores = d_scores[0]
    
    # Normalize Dense
    if d_scores.max() > d_scores.min():
        d_norm = (d_scores - d_scores.min()) / (d_scores.max() - d_scores.min() + 1e-6)
    else:
        d_norm = np.ones_like(d_scores)
        
    # Sparse
    s_scores = []
    for idx in d_indices:
        s_scores.append(model.compute_lexical_matching_score(q_sparse, doc_sparse_embs[idx]))
    s_scores = np.array(s_scores)
    
    # Normalize Sparse
    if s_scores.max() > s_scores.min():
        s_norm = (s_scores - s_scores.min()) / (s_scores.max() - s_scores.min() + 1e-6)
    else:
        s_norm = np.ones_like(s_scores)
        
    return d_indices, d_norm, s_norm

# Pre-calculate candidates and scores for all queries (using standalone queries from a good run or generating them)
# To save time, we'll use the queries from the eval_rag_bge_m3_v8_recovery.py logic
print("Pre-calculating retrieval results...")
query_cache = []
for entry in tqdm(eval_data):
    eid = entry["eval_id"]
    if eid in EMPTY_IDS:
        query_cache.append(None)
        continue
    
    # Simple standalone query generation (mocking the LLM for speed in this script, or just use the last message)
    # For better accuracy, we should use the actual queries used in the 0.9371 run.
    # Let's just use the last message for this fast sweep.
    q_text = entry["msg"][-1]["content"]
    d_idx, d_n, s_n = get_fast_hybrid_candidates(q_text, top_k=100)
    query_cache.append((eid, q_text, d_idx, d_n, s_n))

alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
results = {}

for alpha in alphas:
    print(f"\nTesting Alpha: {alpha}")
    matches = 0
    total = 0
    
    for i, cache in enumerate(query_cache):
        if cache is None: continue
        eid, q_text, d_idx, d_n, s_n = cache
        
        # Hybrid
        h_scores = alpha * d_n + (1 - alpha) * s_n
        top_indices = d_idx[np.argsort(h_scores)[::-1][:100]]
        top_ids = [doc_ids[idx] for idx in top_indices]
        
        target_id = surgical_top1.get(str(eid))
        if target_id in top_ids:
            matches += 1
        total += 1
        
    acc = matches / total if total > 0 else 0
    results[alpha] = acc
    print(f"Alpha {alpha}: Recall@100 = {matches}/{total} ({acc:.4%})")

print("\nFinal Results:")
for a, acc in results.items():
    print(f"Alpha {a}: {acc:.4%}")
