import os
import json
import sys
import numpy as np
import faiss
import torch
from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ==========================================
# 1. 설정 및 데이터 로드
# ==========================================
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
OUTPUT_FILE = "/root/IR/submission_bge_m3_base_simple.csv"

# 모델 설정 (Base 모델 사용)
BGE_M3_MODEL = 'BAAI/bge-m3'

# 파라미터
TOP_CANDIDATES = 100
FINAL_TOPK = 5

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))
    return data

print("Loading documents and eval data...")
docs = load_jsonl(DOC_PATH)
eval_data = load_jsonl(EVAL_PATH)

# ==========================================
# 2. 모델 로드 및 임베딩 생성
# ==========================================
print(f"Loading BGE-M3 model from {BGE_M3_MODEL}...")
model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)

doc_contents = [d['content'] for d in docs]
doc_ids = [d['docid'] for d in docs]

print("Encoding documents (Dense + Sparse)...")
doc_embeddings = model.encode(doc_contents, return_dense=True, return_sparse=True, return_colbert_vecs=False)

# ==========================================
# 3. 검색 및 리랭킹
# ==========================================
print("Starting retrieval...")
results = []

for item in tqdm(eval_data):
    query = ""
    for m in reversed(item['msg']):
        if m['role'] == 'user':
            query = m['content']
            break
    
    eval_id = item['eval_id']
    
    q_embeddings = model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    # Dense Score
    dense_scores = q_embeddings['dense_vecs'] @ doc_embeddings['dense_vecs'].T
    dense_scores = dense_scores[0]
    
    # Sparse Score
    sparse_scores = model.compute_lexical_matching_score(q_embeddings['lexical_weights'], doc_embeddings['lexical_weights'])
    sparse_scores = sparse_scores[0]
    
    # Hybrid Score (0.5:0.5)
    hybrid_scores = 0.5 * dense_scores + 0.5 * sparse_scores
    
    # Top Candidates
    top_indices = np.argsort(hybrid_scores)[::-1][:TOP_CANDIDATES]
    candidate_ids = [doc_ids[idx] for idx in top_indices]
    
    results.append({
        "eval_id": eval_id,
        "topk": candidate_ids[:FINAL_TOPK]
    })

# ==========================================
# 4. 결과 저장
# ==========================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"Done! Saved to {OUTPUT_FILE}")
