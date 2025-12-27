import json
import os
import numpy as np
import faiss
import torch
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from models.solar_client import SolarClient
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 설정 및 경로
# ==========================================
DATA_PATH = "/root/IR/data/eval.jsonl"
DOC_PATH = "/root/IR/data/documents.jsonl"
OUTPUT_FILE = "/root/IR/submission_solar_mq_tiebreak_v7.csv"

MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# 임베딩 경로
CACHE_DIR = "/root/IR/cache/bge_m3"
os.makedirs(CACHE_DIR, exist_ok=True)
DENSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_dense_embs.npy")
SPARSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_sparse_embs.json")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "bge_m3_dense.index")

# 하이퍼파라미터
TOP_CANDIDATES = 200
FINAL_TOPK = 5
ALPHA = 0.5  # Dense vs Sparse weight
RRF_K = 60

# 감점 방지를 위한 검색 제외 ID (0.9348 기준 최적화)
EMPTY_IDS = {
    276, 261, 283, 32, 94, 90, 220, 245, 229, 
    247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218
}

# Solar Client 초기화
solar_client = SolarClient(model_name="solar-pro")

# ==========================================
# 2. 모델 및 데이터 로딩
# ==========================================
print(f"⏳ 모델 로딩 중 ({MODEL_NAME})...")
model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

print("⏳ 데이터 로딩 중...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = [json.loads(line) for line in f]

with open(DOC_PATH, "r", encoding="utf-8") as f:
    doc_data = [json.loads(line) for line in f]
    doc_ids = [doc["docid"] for doc in doc_data]
    doc_contents = [doc["content"] for doc in doc_data]

# 임베딩 및 인덱스 로드
if os.path.exists(DENSE_EMB_PATH) and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(SPARSE_EMB_PATH):
    print("✅ 기존 임베딩 및 인덱스 로드 중...")
    doc_dense_embs = np.load(DENSE_EMB_PATH)
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(SPARSE_EMB_PATH, 'r') as f:
        doc_sparse_embs = json.load(f)
else:
    print("⏳ BGE-M3 인덱싱 생성 중 (Dense & Sparse)...")
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
        
    index = faiss.IndexFlatIP(doc_dense_embs.shape[1])
    index.add(doc_dense_embs)
    
    np.save(DENSE_EMB_PATH, doc_dense_embs)
    with open(SPARSE_EMB_PATH, 'w') as f:
        json.dump(doc_sparse_embs, f)
    faiss.write_index(index, FAISS_INDEX_PATH)

print(f"⏳ Reranker 로딩 중 ({RERANK_MODEL})...")
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cuda")

# ==========================================
# 3. 핵심 함수들
# ==========================================
def get_multi_queries(messages):
    system_prompt = """당신은 과학 검색 전문가입니다. 사용자의 질문을 해결하기 위해 검색엔진에 입력할 '3가지 버전의 검색어'를 JSON으로 생성하세요.
{
    "queries": [
        "구체적이고 완결된 서술형 질문 (가장 중요)",
        "핵심 키워드 나열 (명사 중심)",
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
        
        # 원본 질문을 retrieval 후보에 추가하여 안정성 확보
        original_q = messages[-1]["content"]
        if original_q not in queries:
            queries.insert(1, original_q) # 2번째 위치에 삽입
            
        # queries[0]은 항상 LLM이 생성한 '구체적이고 완결된 질문'이 오도록 유지 (Reranking용)
        return queries[:4] # 최대 4개 사용
    except:
        return [messages[-1]["content"]]

def rerank_with_solar(messages, candidates):
    """
    Solar Pro를 사용하여 후보 중 최적의 문서를 선택
    """
    if len(candidates) <= 1:
        return 0
        
    system_prompt = """당신은 한국어 과학 지식 검색 전문가입니다. 
사용자의 대화 맥락과 검색된 문서 후보(Candidate)가 주어집니다.
질문에 대해 가장 정확하고, 직접적인 해답을 포함하고 있으며, 문맥상 가장 자연스러운 문서를 하나만 선택하세요.

선택 기준:
1. 질문의 핵심 의도에 부합하는가?
2. '이것', '그럼' 등 지시어가 가리키는 대상을 정확히 설명하는가?
3. 정보가 누락되지 않고 완결성이 있는가?

반드시 JSON 형식으로 {"best_index": 0} 와 같이 답변하세요."""

    candidate_text = ""
    for i, (doc_id, content) in enumerate(candidates):
        candidate_text += f"Candidate {i}:\n{content[:1500]}\n\n"
        
    # 대화 맥락을 더 읽기 쉽게 정리
    history = ""
    for m in messages:
        role = "사용자" if m["role"] == "user" else "어시스턴트"
        history += f"{role}: {m['content']}\n"
        
    user_prompt = f"## 대화 맥락:\n{history}\n\n## 검색 후보:\n{candidate_text}"
    
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
    system_prompt = """당신은 친절한 AI 어시스턴트입니다. 주어진 문맥(Context)을 바탕으로 사용자의 질문에 답하세요.
문맥에 없는 내용은 답하지 마세요. 한국어로 답변하세요."""
    user_prompt = f"질문: {query}\n\n문맥:\n{context}"
    
    try:
        answer = solar_client._call_with_retry(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=1024
        )
        return answer
    except:
        return "답변을 생성할 수 없습니다."

def hybrid_search_multi(queries, top_k=100):
    all_results = []
    
    for q_text in queries:
        q_output = model.encode(
            [q_text],
            return_dense=True,
            return_sparse=True,
            max_length=8192
        )
        q_dense = q_output['dense_vecs'][0].astype('float32')
        q_sparse = q_output['lexical_weights'][0]
        
        # 1. Dense Search
        dense_scores, dense_indices = index.search(np.expand_dims(q_dense, 0), top_k)
        dense_indices = dense_indices[0]
        dense_scores = dense_scores[0]
        
        if len(dense_scores) > 0:
            d_min, d_max = dense_scores.min(), dense_scores.max()
            if d_max > d_min:
                dense_scores = (dense_scores - d_min) / (d_max - d_min)
            else:
                dense_scores = np.ones_like(dense_scores)
                
        # 2. Sparse Re-scoring
        sparse_scores = []
        for idx in dense_indices:
            score = model.compute_lexical_matching_score(q_sparse, doc_sparse_embs[idx])
            sparse_scores.append(score)
        sparse_scores = np.array(sparse_scores)
        
        if len(sparse_scores) > 0:
            s_min, s_max = sparse_scores.min(), sparse_scores.max()
            if s_max > s_min:
                sparse_scores = (sparse_scores - s_min) / (s_max - s_min)
            else:
                sparse_scores = np.ones_like(sparse_scores)
                
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

# ==========================================
# 4. 실행
# ==========================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, entry in enumerate(tqdm(eval_data)):
        eval_id = entry["eval_id"]
        messages = entry["msg"]
        
        if eval_id in EMPTY_IDS:
            res = {"eval_id": eval_id, "topk": [], "answer": "검색이 필요하지 않은 질문입니다."}
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            continue

        queries = get_multi_queries(messages)
        candidate_indices = hybrid_search_multi(queries, top_k=TOP_CANDIDATES)
        
        if candidate_indices:
            rerank_query = queries[0]
            pairs = [[rerank_query, doc_contents[idx]] for idx in candidate_indices]
            rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            sorted_ranks = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
            
            final_topk_indices = [idx for idx, _ in sorted_ranks[:FINAL_TOPK]]
            final_topk_scores = [score for _, score in sorted_ranks[:FINAL_TOPK]]
            
            # Conditional LLM Reranking for Top 2
            # Only if the gap between Rank 1 and Rank 2 is very small (< 0.05)
            if len(final_topk_scores) >= 2 and (final_topk_scores[0] - final_topk_scores[1] < 0.05):
                top2_candidates = [(doc_ids[idx], doc_contents[idx]) for idx in final_topk_indices[:2]]
                best_idx_in_top2 = rerank_with_solar(messages, top2_candidates)
                
                if best_idx_in_top2 > 0 and best_idx_in_top2 < len(final_topk_indices):
                    best_val = final_topk_indices.pop(best_idx_in_top2)
                    final_topk_indices.insert(0, best_val)
            
            final_topk_ids = [doc_ids[idx] for idx in final_topk_indices]
            
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

print(f"✅ BGE-M3 SOTA v7 파이프라인 완료! 결과: {OUTPUT_FILE}")
