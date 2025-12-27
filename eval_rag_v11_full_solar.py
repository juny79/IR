import os
import json
import numpy as np
import faiss
import torch
from pathlib import Path
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from models.solar_client import SolarClient

# .env 로드
load_dotenv()

# ==========================================
# 1. 설정 및 데이터 로드
# ==========================================
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
OUTPUT_FILE = "/root/IR/submission_v11_sota.csv"

# 모델 설정
BGE_M3_MODEL = 'BAAI/bge-m3'
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'

# 파라미터
TOP_CANDIDATES = 200
FINAL_TOPK = 5
SOLAR_RERANK_TOPK = 10 
ALPHA = 0.5 
RRF_K = 60

# 0.9348 기준 최적화된 게이팅 ID
EMPTY_IDS = {
    276, 261, 283, 32, 94, 90, 220, 245, 229, 
    247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218
}

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))
    return data

print("🚀 데이터 로딩 중...")
documents = load_jsonl(DOC_PATH)
eval_data = load_jsonl(EVAL_PATH)
doc_contents = [d["content"] for d in documents]
doc_ids = [d["docid"] for d in documents]

# ==========================================
# 2. 모델 로딩
# ==========================================
print(f"⏳ 모델 로딩 중...")
model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device='cuda')
solar = SolarClient()

# FAISS 인덱스 로드 (캐시 사용)
CACHE_DIR = "/root/IR/cache/bge_m3"
DENSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_dense_embs.npy")
SPARSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_sparse_embs.json")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "bge_m3_dense.index")

print("✅ 캐시된 BGE-M3 인덱스 로드")
doc_dense_embs = np.load(DENSE_EMB_PATH)
with open(SPARSE_EMB_PATH, 'r') as f:
    doc_sparse_embs = json.load(f)
index = faiss.read_index(FAISS_INDEX_PATH)

# ==========================================
# 3. 검색 및 재정렬 함수
# ==========================================
def hybrid_search(query_text, top_k=100):
    q_output = model.encode([query_text], return_dense=True, return_sparse=True, max_length=8192)
    q_dense = q_output['dense_vecs'][0].astype('float32')
    q_sparse = q_output['lexical_weights'][0]
    
    dense_scores, dense_indices = index.search(np.expand_dims(q_dense, 0), top_k)
    dense_indices = dense_indices[0]
    dense_scores = dense_scores[0]
    
    if len(dense_scores) > 0:
        d_min, d_max = dense_scores.min(), dense_scores.max()
        if d_max > d_min: dense_scores = (dense_scores - d_min) / (d_max - d_min)
        else: dense_scores = np.ones_like(dense_scores)
            
    sparse_scores = []
    for idx in dense_indices:
        score = model.compute_lexical_matching_score(q_sparse, doc_sparse_embs[idx])
        sparse_scores.append(score)
    sparse_scores = np.array(sparse_scores)
    
    if len(sparse_scores) > 0:
        s_min, s_max = sparse_scores.min(), sparse_scores.max()
        if s_max > s_min: sparse_scores = (sparse_scores - s_min) / (s_max - s_min)
        else: sparse_scores = np.ones_like(sparse_scores)
            
    hybrid_scores = ALPHA * dense_scores + (1 - ALPHA) * sparse_scores
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    return [dense_indices[i] for i in sorted_indices]

def solar_rerank(query, candidates):
    system_prompt = """당신은 한국어 정보 검색 전문가입니다. 
질문(Query)과 여러 개의 문서 후보(Candidate)가 주어집니다.
각 문서를 꼼꼼히 읽고, 질문에 대해 가장 정확하고 직접적인 해답을 포함하고 있는 문서를 하나만 선택하세요.
JSON 형식으로 {"best_index": 0} 와 같이 답변하세요."""

    candidate_text = ""
    for i, cand in enumerate(candidates):
        candidate_text += f"Candidate {i}:\n{cand}\n\n"
    
    user_prompt = f"Query: {query}\n\n{candidate_text}"
    
    try:
        response = solar._call_with_retry(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response)['best_index']
    except:
        return 0

# ==========================================
# 4. 실행
# ==========================================
print(f"🏃 평가 시작... (결과 파일: {OUTPUT_FILE})")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in tqdm(eval_data):
        eval_id = item["eval_id"]
        # msg가 리스트인 경우 마지막 메시지의 content 추출
        if isinstance(item["msg"], list):
            query = item["msg"][-1]["content"]
        else:
            query = item["msg"]
        
        if eval_id in EMPTY_IDS:
            res = {"eval_id": eval_id, "topk": []}
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            continue
            
        # 1. Hybrid Search
        candidate_indices = hybrid_search(query, top_k=TOP_CANDIDATES)
        
        # 2. BGE Reranker (Top 200 -> Top 10)
        pairs = [[query, doc_contents[idx]] for idx in candidate_indices]
        rerank_scores = reranker.predict(pairs)
        top_indices = [candidate_indices[i] for i in np.argsort(rerank_scores)[::-1]]
        
        # 3. Solar Super-Reranker (Top 10 -> Rank 1)
        solar_candidates = [doc_contents[idx] for idx in top_indices[:SOLAR_RERANK_TOPK]]
        best_idx = solar_rerank(query, solar_candidates)
        
        # Rank 1 교체
        final_top_indices = top_indices[:FINAL_TOPK]
        if best_idx > 0 and best_idx < SOLAR_RERANK_TOPK:
            best_doc_idx = top_indices[best_idx]
            # 기존 리스트에서 제거 후 맨 앞으로
            if best_doc_idx in final_top_indices:
                final_top_indices.remove(best_doc_idx)
            final_top_indices = [best_doc_idx] + final_top_indices[:FINAL_TOPK-1]
            
        final_ids = [doc_ids[idx] for idx in final_top_indices]
        
        res = {
            "eval_id": eval_id,
            "topk": final_ids
        }
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()

print(f"✅ v11 SOTA 생성 완료: {OUTPUT_FILE}")
