import os
import json
import sys
import numpy as np
import faiss
import re
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
OUTPUT_FILE = "/root/IR/submission_e5_ultimate.csv"

# 모델 설정
EMBED_MODEL = "intfloat/multilingual-e5-large"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# 파라미터
RRF_K = 60
BM25_TOPN = 60
DENSE_TOPN = 60
TOP_CANDIDATES = 150
RERANK_BATCH = 32
FINAL_TOPK = 5
LLM_RERANK_TOPN = 10 # LLM이 다시 볼 상위 문서 수

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
embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
FAISS_CACHE_PATH = "/root/IR/cache/faiss_e5_large.index"
if os.path.exists(FAISS_CACHE_PATH):
    index = faiss.read_index(FAISS_CACHE_PATH)
else:
    print("FAISS 인덱스가 없습니다. 먼저 생성해야 합니다.")
    sys.exit(1)

print(f"Reranker 로딩 ({RERANK_MODEL})...")
sys.stdout.flush()
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cuda")

# ==========================================
# 2. 핵심 함수들
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
        clean_text = resp.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1].replace("json", "").strip()
        parsed = json.loads(clean_text)
        return parsed.get("queries", [messages[-1]["content"]])[:3], parsed.get("standalone_query", messages[-1]["content"])
    except:
        return [messages[-1]["content"]], messages[-1]["content"]

def llm_rerank(query, docs, top_n=10):
    if not docs: return []
    
    doc_texts = ""
    for idx, doc in enumerate(docs[:top_n]):
        # 너무 길면 자름
        doc_texts += f"[{idx}] {doc[:600]}\n\n"
        
    prompt = f"""질문에 대해 가장 관련성이 높은 문서의 번호를 순서대로 나열하세요.
질문: {query}

문서 리스트:
{doc_texts}

가장 관련 있는 문서부터 번호만 나열하세요. (예: 2, 0, 1, 3...)
출력은 반드시 숫자와 쉼표로만 구성하세요. 리스트에 없는 번호는 쓰지 마세요."""

    try:
        resp = openai_client._call_with_retry(
            prompt=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        order = [int(s) for s in re.findall(r'\d+', resp)]
        # 유효한 인덱스만 필터링
        valid_order = [i for i in order if 0 <= i < len(docs[:top_n])]
        # 누락된 인덱스 추가
        for i in range(len(docs[:top_n])):
            if i not in valid_order:
                valid_order.append(i)
        return valid_order
    except Exception as e:
        print(f"LLM Rerank Error: {e}")
        return list(range(len(docs[:top_n])))

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
        
        # 1. 쿼리 생성 (GPT-4o + Solar Pro)
        gpt_queries, main_q = get_queries(openai_client, "gpt-4o", messages)
        slr_queries, _ = get_queries(solar_client, "solar-pro", messages)
        
        all_queries = gpt_queries + slr_queries
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
            
        # RRF 병합 (가중치 적용)
        current_weights = [0.7, 0.4, 0.4, 0.4, 0.2, 0.2, 1.8, 1.2, 1.2, 1.2, 0.8, 0.8]
        candidate_indices = reciprocal_rank_fusion_weighted(rank_lists, k=RRF_K, weights=current_weights)
        top_candidates = candidate_indices[:TOP_CANDIDATES]
        
        # 1차 Rerank (BGE-Reranker)
        pairs = [[main_q, doc_contents[idx]] for idx in top_candidates]
        rerank_scores = reranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
        sorted_ranks = sorted(zip(top_candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        
        # 2차 Rerank (GPT-4o 최종 검수)
        top_bge_indices = [idx for idx, _ in sorted_ranks[:LLM_RERANK_TOPN]]
        top_bge_contents = [doc_contents[idx] for idx in top_bge_indices]
        
        new_order = llm_rerank(main_q, top_bge_contents, top_n=LLM_RERANK_TOPN)
        
        # 최종 순위 재조정
        final_top_idx = [top_bge_indices[i] for i in new_order]
        # 나머지 (10위 이후) 추가
        final_top_idx += [idx for idx, _ in sorted_ranks[LLM_RERANK_TOPN:]]
        
        final_topk = [doc_ids[idx] for idx in final_top_idx[:FINAL_TOPK]]
        
        res = {
            "eval_id": eval_id,
            "standalone_query": main_q,
            "topk": final_topk,
            "answer": doc_contents[final_top_idx[0]] if final_top_idx else ""
        }
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()

print(f"🏆 얼티밋 앙상블 완료! 결과: {OUTPUT_FILE}")
