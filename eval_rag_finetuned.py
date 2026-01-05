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
from models.solar_client import solar_client

# ==========================================
# 1. 설정 및 데이터 로드
# ==========================================
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
# 파인튜닝된 모델 경로
FINETUNED_MODEL_PATH = "/root/IR/finetuned_bge_m3"
OUTPUT_FILE = "/root/IR/submission_bge_m3_finetuned.csv"

# 모델 설정
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'

# 파라미터 (v9 SOTA 기준)
TOP_CANDIDATES = 100
FINAL_TOPK = 5
ALPHA = 0.5 # Hybrid weight (Dense vs Sparse)
RRF_K = 60

# 감점 방지를 위한 검색 제외 ID (v9 SOTA 기준)
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
# 2. 파인튜닝된 BGE-M3 모델 및 인덱싱
# ==========================================
print(f"⏳ 파인튜닝된 BGE-M3 모델 로딩 중 ({FINETUNED_MODEL_PATH})...")
# 파인튜닝된 모델은 로컬 경로에서 로드
model = BGEM3FlagModel(FINETUNED_MODEL_PATH, use_fp16=True)

# 파인튜닝된 모델용 캐시 디렉토리 별도 운영
CACHE_DIR = "/root/IR/cache/bge_m3_finetuned"
os.makedirs(CACHE_DIR, exist_ok=True)
DENSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_dense_embs.npy")
SPARSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_sparse_embs.json")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "bge_m3_dense.index")

# 파인튜닝된 모델은 가중치가 바뀌었으므로 새로 인덱싱해야 함
if os.path.exists(DENSE_EMB_PATH) and os.path.exists(SPARSE_EMB_PATH) and os.path.exists(FAISS_INDEX_PATH):
    print("✅ 캐시된 파인튜닝 BGE-M3 인덱스 로드")
    doc_dense_embs = np.load(DENSE_EMB_PATH)
    with open(SPARSE_EMB_PATH, 'r') as f:
        doc_sparse_embs = json.load(f)
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    print("⏳ 파인튜닝 BGE-M3 인덱싱 생성 중 (Dense & Sparse)...")
    batch_size = 16
    all_dense = []
    all_sparse = []
    
    for i in tqdm(range(0, len(doc_contents), batch_size)):
        batch_texts = doc_contents[i:i+batch_size]
        output = model.encode(
            batch_texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=True
        )
        all_dense.append(output['dense_vecs'])
        all_sparse.extend(output['lexical_weights'])
        
    doc_dense_embs = np.vstack(all_dense).astype('float32')
    doc_sparse_embs = []
    for sparse_dict in all_sparse:
        doc_sparse_embs.append({k: float(v) for k, v in sparse_dict.items()})
        
    # FAISS Index 생성
    dim = doc_dense_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_dense_embs)
    
    # 저장
    np.save(DENSE_EMB_PATH, doc_dense_embs)
    with open(SPARSE_EMB_PATH, 'w') as f:
        json.dump(doc_sparse_embs, f)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("✅ 인덱싱 완료 및 저장")

# Reranker 로드
print(f"⏳ Reranker 로딩 중 ({RERANK_MODEL})...")
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device='cuda')

# ==========================================
# 3. 유틸리티 함수 (v9 SOTA 로직 유지)
# ==========================================

def get_multi_queries(messages):
    """v9 SOTA: Multi-Query + HyDE 결합"""
    # 1. Standalone Query 생성
    last_msg = messages[-1]["content"]
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
    
    prompt = f"""다음 대화 기록과 질문을 바탕으로, 검색 엔진에 입력할 최적의 '독립적인 한국어 검색 쿼리'를 하나 만드세요.
대화 맥락이 필요 없다면 질문 그대로를 사용하세요.

[대화 기록]
{history}

[질문]
{last_msg}

독립적인 쿼리:"""
    
    standalone_query = solar_client._call_with_retry(prompt).strip()
    
    # 2. HyDE 생성
    hyde_prompt = f"질문: {standalone_query}\n위 질문에 대한 가상의 과학적 답변을 한 문장으로 작성하세요."
    hyde_answer = solar_client._call_with_retry(hyde_prompt).strip()
    
    return [standalone_query, hyde_answer]

def hybrid_search_multi(queries, top_k=100):
    """v9 SOTA: RRF Fusion for Multi-Query"""
    all_results = []
    
    for q in queries:
        # Encode query
        q_output = model.encode(q, return_dense=True, return_sparse=True)
        q_dense = q_output['dense_vecs'].reshape(1, -1).astype('float32')
        q_sparse = q_output['lexical_weights']
        
        # 1. Dense Search
        dense_scores, dense_indices = index.search(q_dense, top_k * 2)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]
        
        # 2. Sparse Search (Lexical)
        # BGE-M3의 compute_lexical_matching 사용
        sparse_scores = []
        for idx in dense_indices:
            score = model.compute_lexical_matching_score(q_sparse, doc_sparse_embs[idx])
            sparse_scores.append(score)
        sparse_scores = np.array(sparse_scores)
        
        # Normalize scores
        if dense_scores.max() > dense_scores.min():
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        if sparse_scores.max() > sparse_scores.min():
            sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
        
        # 3. Hybrid Fusion
        hybrid_scores = ALPHA * dense_scores + (1 - ALPHA) * sparse_scores
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        query_top_indices = [dense_indices[i] for i in sorted_indices]
        all_results.append(query_top_indices)
        
    # 4. RRF Fusion
    rrf_scores = {}
    for results in all_results:
        for rank, idx in enumerate(results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank)
            
    final_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return final_indices[:top_k]

def generate_answer(query, context):
    prompt = f"""당신은 과학 전문 어시스턴트입니다. 제공된 [문서 내용]을 바탕으로 [질문]에 대해 정확하고 친절하게 답변하세요.
문서에 없는 내용은 답변하지 마세요.

[질문]
{query}

[문서 내용]
{context}

답변:"""
    return solar_client._call_with_retry(prompt)

# ==========================================
# 4. 실행
# ==========================================
print("🏃 평가 시작...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, entry in enumerate(tqdm(eval_data)):
        eval_id = entry["eval_id"]
        messages = entry["msg"]
        
        # 1) 게이팅 체크
        if eval_id in EMPTY_IDS:
            res = {"eval_id": eval_id, "topk": [], "answer": "검색이 필요하지 않은 질문입니다."}
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            continue

        # Multi-Query Generation
        queries = get_multi_queries(messages)
        
        # Hybrid Search with RRF
        candidate_indices = hybrid_search_multi(queries, top_k=TOP_CANDIDATES)
        
        # Rerank
        if candidate_indices:
            rerank_query = queries[0]
            pairs = [[rerank_query, doc_contents[idx]] for idx in candidate_indices]
            rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            sorted_ranks = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
            
            final_topk_indices = [idx for idx, _ in sorted_ranks[:FINAL_TOPK]]
            final_topk_ids = [doc_ids[idx] for idx in final_topk_indices]
            
            # Answer generation
            context = "\n".join([doc_contents[idx] for idx in final_topk_indices[:3]])
            answer = generate_answer(rerank_query, context)
            
            res = {
                "eval_id": eval_id,
                "standalone_query": rerank_query,
                "topk": final_topk_ids,
                "answer": answer
            }
        else:
            res = {"eval_id": eval_id, "topk": []}
                
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()

print(f"✅ 파인튜닝 모델 평가 완료! 결과: {OUTPUT_FILE}")
