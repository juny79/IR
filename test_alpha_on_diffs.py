import os
import json
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

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

eval_data = load_jsonl(EVAL_PATH)
surgical_top1 = load_top1(SURGICAL_PATH)
documents = load_jsonl(DOC_PATH)
doc_contents = [d["content"] for d in documents]
doc_ids = [d["docid"] for d in documents]

model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device='cuda')

doc_dense_embs = np.load(os.path.join(CACHE_DIR, "doc_dense_embs.npy"))
with open(os.path.join(CACHE_DIR, "doc_sparse_embs.json"), 'r') as f:
    doc_sparse_embs = json.load(f)
index = faiss.read_index(os.path.join(CACHE_DIR, "bge_m3_dense.index"))

# The 23 IDs where Base (0.9371) differs from Surgical (0.9470)
target_ids = ['3', '7', '26', '30', '31', '35', '37', '43', '69', '84', '85', '91', '97', '102', '104', '106', '205', '214', '243', '246', '250', '271', '305']

alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

for alpha in alphas:
    matches = 0
    for eid_str in target_ids:
        eid = int(eid_str)
        entry = next(e for e in eval_data if e["eval_id"] == eid)
        q_text = entry["msg"][-1]["content"]
        
        # Retrieval
        q_output = model.encode([q_text], return_dense=True, return_sparse=True, max_length=8192)
        q_dense = q_output['dense_vecs'][0].astype('float32')
        q_sparse = q_output['lexical_weights'][0]
        
        d_scores, d_indices = index.search(np.expand_dims(q_dense, 0), 100)
        d_indices = d_indices[0]
        d_scores = d_scores[0]
        
        d_norm = (d_scores - d_scores.min()) / (d_scores.max() - d_scores.min() + 1e-6)
        
        s_scores = []
        for idx in d_indices:
            s_scores.append(model.compute_lexical_matching_score(q_sparse, doc_sparse_embs[idx]))
        s_scores = np.array(s_scores)
        s_norm = (s_scores - s_scores.min()) / (s_scores.max() - s_scores.min() + 1e-6)
        
        h_scores = alpha * d_norm + (1 - alpha) * s_norm
        top_indices = d_indices[np.argsort(h_scores)[::-1][:100]]
        
        # Rerank
        pairs = [[q_text, doc_contents[idx]] for idx in top_indices]
        r_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        best_id = doc_ids[top_indices[np.argmax(r_scores)]]
        
        if best_id == surgical_top1.get(eid_str):
            matches += 1
            
    print(f"Alpha {alpha}: Matches {matches}/{len(target_ids)}")
