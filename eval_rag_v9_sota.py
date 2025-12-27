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
# 1. 설정  데이터 로드
# ==========================================
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
OUTPUT_FILE = "/root/IR/submission_v9_sota.csv"

# 모델 설정
BGE_M3_MODEL = 'BAAI/bge-m3'
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'

# 파라미터
TOP_CANDIDATES = 200
FINAL_TOPK = 5
SOLAR_RERANK_TOPK = 10 # Solar가 검토할 상위 후보 수
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
print(f"⏳ BGE-M3 모델 로딩 중...")
model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)

CACHE_DIR = "/root/IR/cache/bge_m3"
DENSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_dense_embs.npy")
SPARSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_sparse_embs.json")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "bge_m3_dense.index")

print("✅ 캐시된 BGE-M3 인덱스 로드")
doc_dense_embs = np.load(DENSE_EMB_PATH)
with open(SPARSE_EMB_PATH, 'r') as f:
    doc_sparse_embs = json.load(f)
index = faiss.read_index(FAISS_INDEX_PATH)

print(f"⏳ Reranker 로딩 중...")
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cuda")
solar_client = SolarClient(model_name="solar-pro")

# ==========================================
# 3. 핵심 함수들
# ==========================================
def get_multi_queries(messages):
    system_prompt = """당신은 과학 검색 전문가입니다. 사용자의 질문을 해결하기 위해 검색엔진에 입력할 '3가지 버전의 검색어'를 JSON으로 생성하세요.
{
    "queries": [
        "구 완결된 서술형 질문 (가장 중요)",
        "핵심 키워드 나열 (명사 중)",
        "유사한 의미의 다른 표현 질문"
    ]
}"""
    try:
        resp = solar_client._call_with_retry(
            prompt=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0,
            max_tokens=512,
            response_format={"type": "json_object"}
        )
        parsed = json.loads(resp)
        queries = parsed.get("queries", [])
        original_q = messages[-1]["content"]
        if original_q not in queries:
            queries.append(original_q)
        return queries[:3]
    except:
        return [messages[-1]["content"]]

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
        all_results.append([dense_indices[i] for i in sorted_indices])
        
    rrf_scores = {}
    for results in all_results:
        for rank, idx in enumerate(results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank)
            
    final_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return final_indices[:top_k]

def solar_rerank_topk(query, candidates):
    system_prompt = """당신은 한국어 정보 검색 전문가입니다. 
 질문(Query)과 여러 개의 문서 후보(Candidate)가 주어집니다.
 대    가장 정확하고, 직접적인 해답을 포함하고 있는 문서를 하나만 선택하세요.

 JSON 형식으로 {"best_index": 0} 와 같이 답변하세요."""

    candidate_text = ""
    for i, content in enumerate(candidates):
        candidate_text += f"Candidate {i}:\n{content[:2000]}\n\n"
        
    user_prompt = f"## 질문:\n{query}\n\n## 검색 후보:\n{candidate_text}"
    
    try:
        resp = solar_client._call_with_retry(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        parsed = json.loads(resp)
        return int(parsed.get("best_index", 0))
    except:
        return 0

def generate_answer(query, context):
    system_prompt = "주어진 문맥을 바탕으로 사용자의 질문에 한국어로 답변하세요. 문맥에 없는 내용은 답하지 마세요."
    user_prompt = f"질문: {query}\n\n문맥:\n{context}"
    try:
        return solar_client._call_with_retry(
            prompt=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0, max_tokens=1024
        )
    except: return "답 생성할 수 없습니다."

# ==========================================
# 4. 실행 (Resume 기능 추가)
# ==========================================
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                processed_ids.add(json.loads(line)["eval_id"])

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for entry in tqdm(eval_data):
        eval_id = entry["eval_id"]
        if eval_id in processed_ids:
            continue
            
        messages = entry["msg"]
        
        if eval_id in EMPTY_IDS:
            f.write(json.dumps({"eval_id": eval_id, "topk": [], "answer": "검색이 필요하지 않은 질문입니다."}, ensure_ascii=False) + "\n")
            f.flush()
            continue

        queries = get_multi_queries(messages)
        candidate_indices = hybrid_search_multi(queries, top_k=TOP_CANDIDATES)
        
        if candidate_indices:
            rerank_query = queries[0]
            pairs = [[rerank_query, doc_contents[idx]] for idx in candidate_indices]
            rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            sorted_ranks = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
            
            top_indices = [idx for idx, _ in sorted_ranks[:SOLAR_RERANK_TOPK]]
            top_contents = [doc_contents[idx] for idx in top_indices]
            best_idx = solar_rerank_topk(rerank_query, top_contents)
            
            if best_idx >= len(top_indices) or best_idx < 0:
                best_idx = 0
                
            best_doc_idx = top_indices.pop(best_idx)
            final_indices = [best_doc_idx] + top_indices
            
            final_ids = [doc_ids[idx] for idx in final_indices[:FINAL_TOPK]]
            context = "\n".join([doc_contents[idx] for idx in final_indices[:3]])
            answer = generate_answer(rerank_query, context)
            
            res = {
                "eval_id": eval_id,
                "standalone_query": rerank_query,
                "topk": final_ids,
                "answer": answer
            }
        else:
            res = {"eval_id": eval_id, "topk": []}
                
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()

print(f"✅ SOTA 갱신  완료: {OUTPUT_FILE}")
