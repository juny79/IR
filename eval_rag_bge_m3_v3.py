import os
import json
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from models.solar_client import solar_client

# .env 파일 로드
load_dotenv()

# ==========================================
# 1. 설정 및 데이터 로드
# ==========================================
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
OUTPUT_FILE = "/root/IR/submission_v3_final_rerank.csv"

# 모        설정 (새로 학습된 모델 경로 사용)
BGE_M3_MODEL = '/root/IR/finetuned_bge_m3_v3'
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'

# 파라미터
TOP_CANDIDATES = 100
FINAL_TOPK = 5
ALPHA = 0.5 # Hybrid weight (Dense vs Sparse)

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
doc_dict = {d['docid']: d['content'] for d in docs}
doc_ids = [d['docid'] for d in docs]
doc_contents = [d['content'] for d in docs]

# ==========================================
# 2. 모델 로드 및 임베딩 생성
# ==========================================
print(f"Loading BGE-M3 model from {BGE_M3_MODEL}...")
model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)

print(f"Loading Reranker model from {RERANK_MODEL}...")
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device='cuda')

# 문서 임베딩 (Dense + Sparse)
print("Encoding documents (Dense + Sparse)...")
doc_embeddings = model.encode(doc_contents, return_dense=True, return_sparse=True, return_colbert_vecs=False, batch_size=32)
doc_dense_embs = doc_embeddings['dense_vecs']
doc_sparse_embs = doc_embeddings['lexical_weights']

# ==========================================
# 3. 검색 및 리랭킹
# ==========================================
print("Starting retrieval and reranking...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in tqdm(eval_data):
        eval_id = item['eval_id']
        messages = item['msg']
        
        # 1. Solar Analysis (Gating & Standalone Query & HyDE)
        try:
            analysis = solar_client.analyze_query_and_hyde(messages)
            is_science = analysis.get("is_science", True)
            confidence = analysis.get("confidence", 1.0)
            standalone_query = analysis.get("standalone_query", "")
            hyde_text = analysis.get("hyde", "")
        except Exception as e:
            is_science = True
            confidence = 1.0
            standalone_query = ""
            hyde_text = ""

        # Gating: 비과학 질문은 검색 건너뛰기
        if not is_science and confidence > 0.5:
            try:
                answer = solar_client.generate_answer(messages, "")
            except:
                answer = ""
            output = {
                "eval_id": eval_id,
                "standalone_query": standalone_query or messages[-1]['content'],
                "topk": [],
                "answer": answer
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            f.flush()
            continue

        if not standalone_query:
            for m in reversed(messages):
                if m['role'] == 'user':
                    standalone_query = m['content']
                    break
        
        search_query = f"{standalone_query}\n{hyde_text}" if hyde_text else standalone_query
        
        # 2. Hybrid Search (BGE-M3 Dense + Sparse)
        q_embeddings = model.encode([search_query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
        q_dense = q_embeddings['dense_vecs'][0]
        q_sparse = q_embeddings['lexical_weights'][0]
        
        # Dense Score (Dot Product)
        dense_scores = q_dense @ doc_dense_embs.T
        
        # Sparse Score (Loop to avoid ValueError)
        sparse_scores = []
        for d_sparse in doc_sparse_embs:
            score = model.compute_lexical_matching_score(q_sparse, d_sparse)
            sparse_scores.append(score)
        sparse_scores = np.array(sparse_scores)
        
        # Normalization (Min-Max)
        if len(dense_scores) > 0:
            d_min, d_max = dense_scores.min(), dense_scores.max()
            if d_max > d_min:
                dense_scores = (dense_scores - d_min) / (d_max - d_min)
            else:
                dense_scores = np.ones_like(dense_scores)
        
        if len(sparse_scores) > 0:
            s_min, s_max = sparse_scores.min(), sparse_scores.max()
            if s_max > s_min:
                sparse_scores = (sparse_scores - s_min) / (s_max - s_min)
            else:
                sparse_scores = np.ones_like(sparse_scores)
        
        # Hybrid Score
        hybrid_scores = ALPHA * dense_scores + (1 - ALPHA) * sparse_scores
        
        # Top Candidates for Reranking
        top_indices = np.argsort(hybrid_scores)[::-1][:TOP_CANDIDATES]
        candidate_ids = [doc_ids[idx] for idx in top_indices]
        candidate_texts = [doc_dict[cid] for cid in candidate_ids]
        
        # 3. Reranking
        pairs = [[standalone_query, text] for text in candidate_texts]
        rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        
        rerank_indices = np.argsort(rerank_scores)[::-1]
        final_topk_ids = [candidate_ids[idx] for idx in rerank_indices[:FINAL_TOPK]]
        
        # 4. Answer Generation
        context = "\n".join([doc_dict[cid] for cid in final_topk_ids[:3]])
        try:
            answer = solar_client.generate_answer(messages, context)
        except:
            answer = ""
        
        # 5. 결과 저장
        output = {
            "eval_id": eval_id,
            "standalone_query": standalone_query,
            "topk": final_topk_ids,
            "answer": answer
        }
        f.write(json.dumps(output, ensure_ascii=False) + "\n")
        f.flush()

print(f"Done! Saved to {OUTPUT_FILE}")
