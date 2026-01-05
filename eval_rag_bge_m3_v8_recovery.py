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

# LLM 클라이언트
from models.openai_client import openai_client

# ==========================================
# 1. 설정 및 데이터 로드
# ==========================================
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
# 파일명에 전략 명시
TIMESTAMP = os.getenv("TIMESTAMP", "recovery")
OUTPUT_FILE = os.getenv("SUBMISSION_FILE") or f"/root/IR/submission_v8_recovery_{TIMESTAMP}.csv"

# 모델 설정
BGE_M3_MODEL = 'BAAI/bge-m3'
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'

# 파라미터 (최적화된 기본값)
TOP_CANDIDATES = int(os.getenv("TOP_CANDIDATES", "150"))
FINAL_TOPK = int(os.getenv("FINAL_TOPK", "5"))
ALPHA = float(os.getenv("ALPHA", "0.5"))
RRF_K = int(os.getenv("RRF_K", "60"))

# 옵션 (V2 실패를 교훈삼아 MQ는 항상 ON)
USE_MULTI_QUERY = True 
USE_LLM_RERANK_TOP3 = True
SKIP_ANSWER = True

# ID 218 제거 (사용자 요청 및 분석 결과 반영)
EMPTY_IDS = {
    276, 261, 283, 32, 94, 90, 220, 245, 229, 
    247, 67, 57, 2, 227, 301, 222, 83, 64, 103
    # 218 is removed here
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
# 2. BGE-M3 모델 및 인덱싱
# ==========================================
print(f"⏳ BGE-M3 모델 로딩 중 ({BGE_M3_MODEL})...")
model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)

CACHE_DIR = "/root/IR/cache/bge_m3"
os.makedirs(CACHE_DIR, exist_ok=True)
DENSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_dense_embs.npy")
SPARSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_sparse_embs.json")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "bge_m3_dense.index")

print("✅ 캐시된 BGE-M3 인덱스 로드")
doc_dense_embs = np.load(DENSE_EMB_PATH)
with open(SPARSE_EMB_PATH, 'r') as f:
    doc_sparse_embs = json.load(f)
index = faiss.read_index(FAISS_INDEX_PATH)

# ==========================================
# 3. 검색 및 리랭킹 함수
# ==========================================
print(f"⏳ Reranker 로딩 중 ({RERANK_MODEL})...")
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device='cuda')

def get_multi_queries(messages):
    try:
        system_prompt = "당신은 질문 최적화 전문가입니다. 사용자의 대화 맥락을 분석하여, 검색 엔진이 가장 정확한 문서를 찾을 수 있도록 질문을 구체적인 단일 문장으로 정제하고 확장하세요."
        user_prompt = f"대화 맥락:\n{json.dumps(messages, ensure_ascii=False, indent=2)}\n\n위 대화를 바탕으로, 마지막 질문의 의도를 담은 구체적인 검색어 3개를 JSON 리스트 ['질문1', '질문2', '질문3'] 형식으로 출력하세요."
        
        resp = openai_client._call_with_retry(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        parsed = json.loads(resp)
        if isinstance(parsed, dict):
            queries = parsed.get("queries", parsed.get("questions", []))
        elif isinstance(parsed, list):
            queries = parsed
        else:
            queries = []
            
        if not queries:
            queries = [messages[-1]["content"]]
        return queries[:3]
    except:
        return [messages[-1]["content"]]

def rerank_with_llm_v2(messages, candidates):
    if len(candidates) <= 1: return 0
    system_prompt = """당신은 검색 결과의 정확도를 판별하는 전문가입니다.
사용자의 질문(대화 맥락 포함)과 3개의 검색 결과(Candidate)가 주어집니다.
질문에 대해 가장 정확하고, 직접적이며, 완결된 답변을 제공하는 문서를 하나만 골라주세요.
반드시 JSON 형식으로 {"best_index": 0} 와 같이 답변하세요. (0, 1, 2 중 선택)"""
    candidate_text = ""
    for i, (doc_id, content) in enumerate(candidates):
        candidate_text += f"Candidate {i}:\n{content[:1500]}\n\n"
    user_prompt = f"대화 맥락:\n{json.dumps(messages, ensure_ascii=False, indent=2)}\n\n{candidate_text}"
    try:
        resp = openai_client._call_with_retry(
            prompt=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return int(json.loads(resp).get("best_index", 0))
    except: return 0

def hybrid_search_multi(queries, top_k=100):
    all_results = []
    for q_text in queries:
        q_output = model.encode([q_text], return_dense=True, return_sparse=True, max_length=8192)
        q_dense = q_output['dense_vecs'][0].astype('float32')
        q_sparse = q_output['lexical_weights'][0]
        
        dense_scores, dense_indices = index.search(np.expand_dims(q_dense, 0), top_k)
        dense_indices = dense_indices[0]
        dense_scores = dense_scores[0]
        
        if len(dense_scores) > 0:
            d_min, d_max = dense_scores.min(), dense_scores.max()
            dense_scores = (dense_scores - d_min) / (d_max - d_min + 1e-6)
                
        sparse_scores = []
        for idx in dense_indices:
            score = model.compute_lexical_matching_score(q_sparse, doc_sparse_embs[idx])
            sparse_scores.append(score)
        sparse_scores = np.array(sparse_scores)
        
        if len(sparse_scores) > 0:
            s_min, s_max = sparse_scores.min(), sparse_scores.max()
            sparse_scores = (sparse_scores - s_min) / (s_max - s_min + 1e-6)
                
        hybrid_scores = ALPHA * dense_scores + (1 - ALPHA) * sparse_scores
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        all_results.append([dense_indices[i] for i in sorted_indices])
        
    rrf_scores = {}
    for results in all_results:
        for rank, idx in enumerate(results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank)
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

# ==========================================
# 4. 실행
# ==========================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in tqdm(eval_data):
        eval_id = entry["eval_id"]
        if eval_id in EMPTY_IDS:
            f.write(json.dumps({"eval_id": eval_id, "topk": [], "answer": ""}, ensure_ascii=False) + "\n")
            continue

        queries = get_multi_queries(entry["msg"])
        candidate_indices = hybrid_search_multi(queries, top_k=TOP_CANDIDATES)
        
        if candidate_indices:
            rerank_query = queries[0]
            pairs = [[rerank_query, doc_contents[idx]] for idx in candidate_indices]
            rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            sorted_ranks = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
            final_topk_indices = [idx for idx, _ in sorted_ranks[:FINAL_TOPK]]
            
            if USE_LLM_RERANK_TOP3:
                top3_candidates = [(doc_ids[idx], doc_contents[idx]) for idx in final_topk_indices[:3]]
                best_idx = rerank_with_llm_v2(entry["msg"], top3_candidates)
                if 0 < best_idx < len(final_topk_indices):
                    val = final_topk_indices.pop(best_idx)
                    final_topk_indices.insert(0, val)
            
            f.write(json.dumps({"eval_id": eval_id, "topk": [doc_ids[idx] for idx in final_topk_indices], "answer": ""}, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps({"eval_id": eval_id, "topk": []}, ensure_ascii=False) + "\n")
        f.flush()

print(f"✅ Recovery 완료: {OUTPUT_FILE}")
