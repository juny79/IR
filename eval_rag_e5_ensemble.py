import os
import json
import sys
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# LLM 클라이언트들
from models.solar_client import solar_client
from models.openai_client import openai_client

# ==========================================
# 1. 설정 및 데이터 로드
# ==========================================
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
OUTPUT_FILE = "/root/IR/submission_e5_super_ensemble.csv"

# 모델 설정
EMBED_MODEL = "intfloat/multilingual-e5-large"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# 파라미터 (앙상블 최적화)
RRF_K = 60
BM25_TOPN = 60
DENSE_TOPN = 60
TOP_CANDIDATES = 150 # 후보군 확대
RERANK_BATCH = 32
FINAL_TOPK = 5

# 가중치 설정 (GPT-4o 쿼리 3개 + Solar Pro 쿼리 3개)
# [GPT_BM1, GPT_BM2, GPT_BM3, GPT_DS1, GPT_DS2, GPT_DS3, SLR_BM1, SLR_BM2, SLR_BM3, SLR_DS1, SLR_DS2, SLR_DS3]
# GPT-4o에 더 높은 가중치 부여
W_GPT = [0.7, 0.4, 0.4, 1.8, 1.2, 1.2]
W_SLR = [0.4, 0.2, 0.2, 1.2, 0.8, 0.8]
SUPER_WEIGHTS = W_GPT + W_SLR

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))
    return data

print("🚀 데이터 로딩 및 인덱싱 준비...")
sys.stdout.flush()
documents = load_jsonl(DOC_PATH)
eval_data = load_jsonl(EVAL_PATH)
doc_contents = [d["content"] for d in documents]
doc_ids = [d["docid"] for d in documents]

kiwi = Kiwi()
def tokenizer(text: str):
    tokens = kiwi.tokenize(text)
    return [t.form for t in tokens if t.tag.startswith("N") or t.tag in ["SL", "SN"]]

print("BM25 인덱싱...")
sys.stdout.flush()
tokenized_corpus = [tokenizer(doc) for doc in doc_contents]
bm25 = BM25Okapi(tokenized_corpus)

print(f"Vector 인덱싱 ({EMBED_MODEL})...")
sys.stdout.flush()
embedder = SentenceTransformer(EMBED_MODEL, device="cuda") # GPU 사용
FAISS_CACHE_PATH = "/root/IR/cache/faiss_e5_large.index"
if os.path.exists(FAISS_CACHE_PATH):
    index = faiss.read_index(FAISS_CACHE_PATH)
else:
    # 생략 (이미 존재한다고 가정)
    pass

print(f"Reranker 로딩 ({RERANK_MODEL})...")
sys.stdout.flush()
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cuda") # GPU 사용

# ==========================================
# 2. 앙상블 쿼리 생성 함수
# ==========================================
def get_queries(client, model_name, messages):
    system_prompt = """당신은 RAG용 질문 분석기입니다. 질문을 분석하여 검색 쿼리 3개를 JSON으로 출력하세요.
{
  "should_search": true,
  "standalone_query": "구체적인 질문",
  "queries": ["쿼리1", "쿼리2", "쿼리3"]
}"""
    try:
        if hasattr(client, 'model'): client.model = model_name
        resp = client._call_with_retry(
            prompt=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0,
            max_tokens=1024,
            response_format={"type": "json_object"} if "gpt" in model_name else None
        )
        # JSON 클리닝
        clean_text = resp.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1].replace("json", "").strip()
        parsed = json.loads(clean_text)
        return parsed.get("queries", [messages[-1]["content"]])[:3], parsed.get("standalone_query", messages[-1]["content"])
    except:
        return [messages[-1]["content"]], messages[-1]["content"]

def reciprocal_rank_fusion_weighted(rank_lists, k=60, weights=None):
    if weights is None: weights = [1.0] * len(rank_lists)
    scores = {}
    for w, rank_list in zip(weights, rank_lists):
        for rank, doc_idx in enumerate(rank_list):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + w * (1.0 / (k + rank + 1))
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

# ==========================================
# 3. 실행
# ==========================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, entry in enumerate(tqdm(eval_data)):
        eval_id = entry["eval_id"]
        messages = entry["msg"]
        
        # 1. GPT-4o 쿼리
        gpt_queries, main_q = get_queries(openai_client, "gpt-4o", messages)
        # 2. Solar Pro 쿼리
        slr_queries, _ = get_queries(solar_client, "solar-pro", messages)
        
        all_queries = gpt_queries + slr_queries # 총 6개
        rank_lists = []
        
        # BM25 Search
        for q in all_queries:
            tokens = tokenizer(q)
            if tokens:
                scores = bm25.get_scores(tokens)
                rank_lists.append(np.argsort(scores)[::-1][:BM25_TOPN].tolist())
            else:
                rank_lists.append([])
        
        # Dense Search
        q_embs = embedder.encode(["query: " + q for q in all_queries], normalize_embeddings=True)
        for q_emb in q_embs:
            _, f_idx = index.search(np.expand_dims(q_emb, 0), DENSE_TOPN)
            rank_lists.append(f_idx[0].tolist())
            
        # RRF (BM25 6개 + Dense 6개 = 12개 리스트)
        # SUPER_WEIGHTS 순서: [GPT_BM1,2,3, SLR_BM1,2,3, GPT_DS1,2,3, SLR_DS1,2,3]
        # 위에서 rank_lists에 넣은 순서는 [BM_GPT1,2,3, BM_SLR1,2,3, DS_GPT1,2,3, DS_SLR1,2,3]
        current_weights = [0.7, 0.4, 0.4, 0.4, 0.2, 0.2, 1.8, 1.2, 1.2, 1.2, 0.8, 0.8]
        
        candidate_indices = reciprocal_rank_fusion_weighted(rank_lists, k=RRF_K, weights=current_weights)
        top_candidates = candidate_indices[:TOP_CANDIDATES]
        
        # Rerank
        pairs = [[main_q, doc_contents[idx]] for idx in top_candidates]
        rerank_scores = reranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
        sorted_ranks = sorted(zip(top_candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        
        final_top_idx = [idx for idx, _ in sorted_ranks[:FINAL_TOPK]]
        final_topk = [doc_ids[idx] for idx in final_top_idx]
        
        res = {
            "eval_id": eval_id,
            "standalone_query": main_q,
            "topk": final_topk,
            "answer": doc_contents[final_top_idx[0]] if final_top_idx else ""
        }
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()

print(f"🚀 슈퍼 앙상블 완료! 결과: {OUTPUT_FILE}")
