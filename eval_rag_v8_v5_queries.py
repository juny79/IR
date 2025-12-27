import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from models.solar_client import SolarClient
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 설정 및 경로
# ==========================================
V5_SUBMISSION = "/root/IR/submission_bge_m3_sota_v5.csv"
DATA_PATH = "/root/IR/data/eval.jsonl"
DOC_PATH = "/root/IR/data/documents.jsonl"
OUTPUT_FILE = "/root/IR/submission_v8_v5_queries_solar_tiebreak.csv"

MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# 임베딩 경로
CACHE_DIR = "/root/IR/cache/bge_m3"
DENSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_dense_embs.npy")
SPARSE_EMB_PATH = os.path.join(CACHE_DIR, "doc_sparse_embs.json")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "bge_m3_dense.index")

# 하이퍼파라미터
TOP_CANDIDATES = 200
FINAL_TOPK = 5
ALPHA = 0.5
RRF_K = 60
TIE_BREAK_GAP = 0.05

EMPTY_IDS = {
    276, 261, 283, 32, 94, 90, 220, 245, 229, 
    247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218
}

# ==========================================
# 2. 데이터 및 모델 로딩
# ==========================================
print("⏳ v5 쿼리 추출 중...")
v5_queries = {}
with open(V5_SUBMISSION, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        v5_queries[int(obj["eval_id"])] = obj.get("standalone_query", "")

print(f"⏳ 모델 로딩 중 ({MODEL_NAME})...")
model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

print("⏳ 데이터 로딩 중...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    eval_data = [json.loads(line) for line in f]

with open(DOC_PATH, "r", encoding="utf-8") as f:
    doc_data = [json.loads(line) for line in f]
    doc_ids = [doc["docid"] for doc in doc_data]
    doc_contents = [doc["content"] for doc in doc_data]

# 임베딩 로드
doc_dense_embs = np.load(DENSE_EMB_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(SPARSE_EMB_PATH, 'r') as f:
    doc_sparse_embs = json.load(f)

print(f"⏳ Reranker 로딩 중 ({RERANK_MODEL})...")
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cuda")
solar_client = SolarClient(model_name="solar-pro")

# ==========================================
# 3. 핵심 함수
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

def solar_tiebreak(messages, candidates):
    system_prompt = """당신은 한국어 과학 지식 검색 전문가입니다. 
가장 정확하고 직접적인 해답을 포함하는 문서를 하나만 선택하세요. 불확실하면 Candidate 0을 선택하세요.
반드시 JSON 형식으로 {"best_index": 0} 와 같이 답변하세요."""
    
    candidate_text = ""
    for i, (doc_id, content) in enumerate(candidates):
        candidate_text += f"Candidate {i}:\n{content[:1500]}\n\n"
    
    history = ""
    for m in messages:
        role = "사용자" if m["role"] == "user" else "어시스턴트"
        history += f"{role}: {m['content']}\n"
        
    user_prompt = f"## 대화 맥락:\n{history}\n\n## 검색 후보:\n{candidate_text}"
    
    try:
        resp = solar_client._call_with_retry(
            prompt=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0, max_tokens=50, response_format={"type": "json_object"}
        )
        return int(json.loads(resp).get("best_index", 0))
    except: return 0

def generate_answer(query, context):
    system_prompt = "주어진 문맥을 바탕으로 사용자의 질문에 한국어로 답변하세요. 문맥에 없는 내용은 답하지 마세요."
    user_prompt = f"질문: {query}\n\n문맥:\n{context}"
    try:
        return solar_client._call_with_retry(
            prompt=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0, max_tokens=1024
        )
    except: return "답변을 생성할 수 없습니다."

# ==========================================
# 4. 실행
# ==========================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in tqdm(eval_data):
        eval_id = entry["eval_id"]
        messages = entry["msg"]
        
        if eval_id in EMPTY_IDS:
            f.write(json.dumps({"eval_id": eval_id, "topk": [], "answer": "검색이 필요하지 않은 질문입니다."}, ensure_ascii=False) + "\n")
            continue

        # v5에서 사용했던 고품질 쿼리 사용 (없으면 원본 질문)
        query = v5_queries.get(eval_id) or messages[-1]["content"]
        
        candidate_indices = hybrid_search(query, top_k=TOP_CANDIDATES)
        
        if candidate_indices:
            pairs = [[query, doc_contents[idx]] for idx in candidate_indices]
            rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            sorted_ranks = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
            
            final_indices = [idx for idx, _ in sorted_ranks[:FINAL_TOPK]]
            final_scores = [score for _, score in sorted_ranks[:FINAL_TOPK]]
            
            # Solar Tie-break (Top 2)
            if len(final_scores) >= 2 and (final_scores[0] - final_scores[1] < TIE_BREAK_GAP):
                top2 = [(doc_ids[idx], doc_contents[idx]) for idx in final_indices[:2]]
                best_idx = solar_tiebreak(messages, top2)
                if best_idx == 1:
                    final_indices[0], final_indices[1] = final_indices[1], final_indices[0]
            
            final_ids = [doc_ids[idx] for idx in final_indices]
            context = "\n".join([doc_contents[idx] for idx in final_indices[:3]])
            answer = generate_answer(query, context)
            
            f.write(json.dumps({"eval_id": eval_id, "standalone_query": query, "topk": final_ids, "answer": answer}, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps({"eval_id": eval_id, "topk": []}, ensure_ascii=False) + "\n")
        f.flush()

print(f"✅ v8 파이프라인 완료: {OUTPUT_FILE}")
