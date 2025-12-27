import os
import json
import sys
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
print(f"🔑 OpenAI API Key loaded: {os.getenv('OPENAI_API_KEY')[:5]}***")

# LLM 클라이언트들 임포트
from models.solar_client import solar_client
from models.gemini_client import gemini_client
from models.openai_client import openai_client

# ==========================================
# 0. LLM 선택 (인자 처리)
# ==========================================
# 사용법: python eval_rag_e5_repro.py [solar_pro|gemini|gpt4o|solar_mini]
llm_type = sys.argv[1] if len(sys.argv) > 1 else "solar_mini"

if llm_type == "solar_pro":
    active_client = solar_client
    active_client.model = "solar-pro"
    model_label = "Solar Pro"
elif llm_type == "gemini":
    active_client = gemini_client
    active_client.model_name = "gemini-3-flash-preview"
    import google.generativeai as genai
    generation_config = {
        "temperature": 0.1,
        "response_mime_type": "application/json",
        "max_output_tokens": 2048,
    }
    active_client.model = genai.GenerativeModel(
        model_name=active_client.model_name,
        generation_config=generation_config
    )
    model_label = "Gemini 3 Flash Preview"
elif llm_type == "gpt4o":
    active_client = openai_client
    active_client.model = "gpt-4o"
    model_label = "GPT-4o"
else:
    active_client = solar_client
    active_client.model = "solar-1-mini-chat"
    model_label = "Solar 1 Mini"

print(f"🌟 Active LLM: {model_label}")

# ==========================================
# 1. 설정 및 데이터 로드
# ==========================================
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
OUTPUT_FILE = f"/root/IR/submission_e5_{llm_type}.csv"

# 모델 설정
EMBED_MODEL = "intfloat/multilingual-e5-large"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# 파라미터 (동료의 0.9174 세팅)
RRF_K = 60
BM25_TOPN = 50
DENSE_TOPN = 50
TOP_CANDIDATES = 100
RERANK_BATCH = 32
FINAL_TOPK = 5
W3_WEIGHTS = [0.6, 0.3, 0.3, 1.6, 1.0, 1.0]

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
# 2. 인덱싱 (Kiwi + E5)
# ==========================================
kiwi = Kiwi()
def tokenizer(text: str):
    tokens = kiwi.tokenize(text)
    return [t.form for t in tokens if t.tag.startswith("N") or t.tag in ["SL", "SN"]]

print("BM25 인덱싱 중...")
tokenized_corpus = [tokenizer(doc) for doc in doc_contents]
bm25 = BM25Okapi(tokenized_corpus)

print(f"Vector 인덱싱 로드 중 ({EMBED_MODEL})...")
embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
# 기존에 생성된 FAISS 인덱스가 있다면 로드, 없으면 생성 (여기서는 새로 생성하는 로직 포함)
FAISS_CACHE_PATH = "/root/IR/cache/faiss_e5_large.index"
EMB_CACHE_PATH = "/root/IR/cache/doc_embeddings_e5_large.npy"

if os.path.exists(FAISS_CACHE_PATH) and os.path.exists(EMB_CACHE_PATH):
    print("✅ 캐시된 FAISS 인덱스 로드")
    index = faiss.read_index(FAISS_CACHE_PATH)
else:
    print("⏳ FAISS 인덱스 생성 중...")
    os.makedirs("/root/IR/cache", exist_ok=True)
    passage_texts = ["passage: " + doc for doc in doc_contents]
    doc_embeddings = embedder.encode(passage_texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")
    np.save(EMB_CACHE_PATH, doc_embeddings)
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    faiss.write_index(index, FAISS_CACHE_PATH)

print(f"Reranker 로딩 중 ({RERANK_MODEL})...")
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cuda")

# ==========================================
# 3. 핵심 함수 (동료 로직 복제)
# ==========================================
def process_query_expanded(messages):
    system_prompt = """당신은 RAG(문서검색)용 질문 분석기입니다.
아래 기준으로 "검색이 필요한 질문인지" 판단하고, 검색에 쓸 쿼리 3개와 HyDE를 생성하세요.

[판단 기준]
- should_search=true:
  지식/기술/역사/사회/문화/과학/설명/정의/원리/비교/원인/방법 등
  코퍼스에서 근거 문서를 찾아야 정확해지는 질문
- should_search=false:
  순수 잡담/인사/감정표현/메타 대화("고마워", "안녕", "너 누구야")
  또는 실시간 정보(날씨/현재시각/주가 등) 같이 코퍼스로 해결 불가한 질문

[출력 JSON 형식만!]
{
  "should_search": true/false,
  "confidence": 0.0~1.0,
  "standalone_query": "독립적으로 이해 가능한 질문문(가장 구체적)",
  "queries": [
    "구체적 서술형(standalone_query와 같거나 유사)",
    "핵심 키워드 나열",
    "유사 표현/다른 관점의 질문"
  ],
  "hyde": "가설적 답변(200자 이내, 문서에 있을 법한 내용으로)"
}"""
    
    try:
        response_text = active_client._call_with_retry(
            prompt=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        if not response_text:
            raise ValueError("Empty response from LLM")
            
        # Markdown code block 제거 (있을 경우)
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            # ```json ... ``` 또는 ``` ... ``` 제거
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
            
        parsed = json.loads(clean_text)
        
        standalone_query = parsed.get("standalone_query", "")
        queries = parsed.get("queries", [])
        if not queries: queries = [standalone_query]
        if queries[0] != standalone_query:
            queries = [standalone_query] + [q for q in queries if q != standalone_query]
        
        return {
            "is_science": bool(parsed.get("should_search", True)),
            "queries": queries[:3],
            "standalone_query": standalone_query
        }
    except Exception as e:
        print(f"⚠️ LLM 호출 실패: {e}")
        last_content = messages[-1]["content"]
        return {
            "is_science": True,
            "queries": [last_content],
            "standalone_query": last_content
        }

def reciprocal_rank_fusion_weighted(rank_lists, k=60, weights=None):
    if weights is None:
        weights = [1.0] * len(rank_lists)
    
    scores = {}
    for w, rank_list in zip(weights, rank_lists):
        for rank, doc_idx in enumerate(rank_list):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + w * (1.0 / (k + rank + 1))
    
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

# 저장
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i, entry in enumerate(tqdm(eval_data)):
        eval_id = entry["eval_id"]
        messages = entry["msg"]
        
        print(f"[{i+1}/{len(eval_data)}] Processing eval_id: {eval_id}...")
        processed = process_query_expanded(messages)
        
        if not processed["is_science"]:
            res = {
                "eval_id": eval_id,
                "standalone_query": "",
                "topk": [],
                "answer": "과학 상식과 관련 없는 대화입니다.",
                "references": []
            }
        else:
            queries = processed["queries"]
            main_query = queries[0]
            
            all_bm25_lists = []
            all_faiss_lists = []
            
            for q in queries:
                # BM25
                tokens = tokenizer(q)
                if tokens:
                    bm25_scores = bm25.get_scores(tokens)
                    top_idx = np.argsort(bm25_scores)[::-1][:BM25_TOPN]
                    all_bm25_lists.append(top_idx.tolist())
                else:
                    all_bm25_lists.append([])
                    
                # Dense
                q_emb = embedder.encode(["query: " + q], normalize_embeddings=True)
                _, f_idx = index.search(q_emb, DENSE_TOPN)
                all_faiss_lists.append(f_idx[0].tolist())
                
            rank_lists = all_bm25_lists + all_faiss_lists
            weights = W3_WEIGHTS if len(rank_lists) == 6 else [1.0] * len(rank_lists)
            
            candidate_indices = reciprocal_rank_fusion_weighted(rank_lists, k=RRF_K, weights=weights)
            top_candidates = candidate_indices[:TOP_CANDIDATES]
            
            if not top_candidates:
                res = {
                    "eval_id": eval_id,
                    "standalone_query": main_query,
                    "topk": [],
                    "answer": "문서를 찾지 못했습니다.",
                    "references": []
                }
            else:
                # Rerank
                pairs = [[main_query, doc_contents[idx]] for idx in top_candidates]
                scores = reranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
                sorted_ranks = sorted(zip(top_candidates, scores), key=lambda x: x[1], reverse=True)
                
                final_top_idx = [idx for idx, _ in sorted_ranks[:FINAL_TOPK]]
                final_topk = [doc_ids[idx] for idx in final_top_idx]
                final_contents = [doc_contents[idx] for idx in final_top_idx]
                
                res = {
                    "eval_id": eval_id,
                    "standalone_query": main_query,
                    "topk": final_topk,
                    "answer": final_contents[0] if final_contents else "문서를 찾지 못했습니다.",
                    "references": [{"score": 0, "content": c} for c in final_contents]
                }
        
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()

print(f"✅ 재현 완료! 파일 저장됨: {OUTPUT_FILE}")

print(f"✅ 재현 완료! 파일 저장됨: {OUTPUT_FILE}")
