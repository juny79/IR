import os
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import faiss
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

# .env 로드 (Gemini client import 이전에 수행)
load_dotenv()

from models.gemini_client import gemini_client

# ==========================================
# 1. 설정 / 데이터 로드
# ==========================================
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
OUTPUT_FILE = os.getenv("SUBMISSION_FILE") or "/root/IR/submission_v16_gemini_rerank.csv"

BGE_M3_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

TOP_CANDIDATES = int(os.getenv("TOP_CANDIDATES", "200"))
FINAL_TOPK = int(os.getenv("FINAL_TOPK", "5"))
GEMINI_RERANK_TOPK = int(os.getenv("GEMINI_RERANK_TOPK", "10"))
ALPHA = float(os.getenv("ALPHA", "0.5"))
RRF_K = int(os.getenv("RRF_K", "60"))

# v9 기준 튜닝된 게이팅(empty topk)
EMPTY_IDS = {
    276, 261, 283, 32, 94, 90, 220, 245, 229,
    247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218
}

CACHE_DIR = Path(os.getenv("CACHE_DIR", "/root/IR/cache/v16_gemini"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


print("🚀 데이터 로딩 중...")
documents = load_jsonl(DOC_PATH)
eval_data = load_jsonl(EVAL_PATH)
doc_contents = [d["content"] for d in documents]
doc_ids = [d["docid"] for d in documents]

docid_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

# ==========================================
# 2. 모델 로딩
# ==========================================
print("⏳ BGE-M3 모델 로딩 중...")
model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=True)

BGE_CACHE_DIR = "/root/IR/cache/bge_m3"
DENSE_EMB_PATH = os.path.join(BGE_CACHE_DIR, "doc_dense_embs.npy")
SPARSE_EMB_PATH = os.path.join(BGE_CACHE_DIR, "doc_sparse_embs.json")
FAISS_INDEX_PATH = os.path.join(BGE_CACHE_DIR, "bge_m3_dense.index")

print("✅ 캐시된 BGE-M3 인덱스 로드")
doc_dense_embs = np.load(DENSE_EMB_PATH)
with open(SPARSE_EMB_PATH, "r", encoding="utf-8") as f:
    doc_sparse_embs = json.load(f)
index = faiss.read_index(FAISS_INDEX_PATH)

print("⏳ Reranker 로딩 중...")
reranker = CrossEncoder(RERANK_MODEL, max_length=512, device="cuda")


# ==========================================
# 3. 핵심 유틸
# ==========================================

def _safe_extract_json_object(text: str):
    if not text:
        return None

    cleaned = str(text).strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = max((p.strip() for p in parts if p.strip()), key=len, default=cleaned)
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = cleaned[start : end + 1].strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def get_multi_queries_gemini(messages):
    """Gemini로 3개 멀티쿼리 생성 (v9의 Solar 멀티쿼리 대체)."""
    # Cache by conversation text to avoid repeated Gemini calls across reruns/resume.
    try:
        conversation_text = "\n".join(
            [f"{m.get('role','user')}: {m.get('content','')}" for m in (messages or [])]
        ).strip()
    except Exception:
        conversation_text = str(messages)

    cache_key = hashlib.md5(conversation_text.encode("utf-8", errors="ignore")).hexdigest()
    cache_path = CACHE_DIR / f"gemini_multiq_{cache_key}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            queries = cached.get("queries", [])
            queries = [q for q in queries if isinstance(q, str) and q.strip()]
            if queries:
                return queries[:3]
        except Exception:
            pass

    system_prompt = """당신은 과학 검색 전문가입니다.
사용자의 질문을 해결하기 위해 검색엔진에 입력할 '3가지 버전의 검색어'를 JSON으로 생성하세요.

반드시 아래 형식의 JSON만 출력하세요.
{
  "queries": [
    "구체적이고 완결된 서술형 질문 (가장 중요)",
    "핵심 키워드 나열 (명사 위주)",
    "유사한 의미의 다른 표현 질문"
  ]
}
"""

    resp = gemini_client._call_with_retry(
        prompt=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0,
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    parsed = _safe_extract_json_object(resp) or {}
    queries = parsed.get("queries", [])

    # fallback: 원 질문 포함 보장
    original_q = messages[-1].get("content", "") if messages else ""
    if original_q and original_q not in queries:
        queries.append(original_q)

    # 문자열만 추출
    queries = [q for q in queries if isinstance(q, str) and q.strip()]

    final = queries[:3] if queries else ([original_q] if original_q else [])
    try:
        cache_path.write_text(json.dumps({"queries": final}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return final


def hybrid_search_multi(queries, top_k=200):
    """v9와 동일한 방식:
    - 각 쿼리별로 dense top_k 후보를 뽑고
    - 그 후보들에 대해 lexical matching score 계산
    - ALPHA로 혼합 후 쿼리별 랭킹을 RRF로 합침
    """
    if not queries:
        return []

    all_results = []
    for q_text in queries:
        q_output = model.encode([q_text], return_dense=True, return_sparse=True, max_length=8192)
        q_dense = q_output["dense_vecs"][0].astype("float32")
        q_sparse = q_output["lexical_weights"][0]

        dense_scores, dense_indices = index.search(np.expand_dims(q_dense, 0), top_k)
        dense_indices = dense_indices[0]
        dense_scores = dense_scores[0]

        if len(dense_scores) > 0:
            d_min, d_max = dense_scores.min(), dense_scores.max()
            if d_max > d_min:
                dense_scores = (dense_scores - d_min) / (d_max - d_min)
            else:
                dense_scores = np.ones_like(dense_scores)

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

        hybrid_scores = ALPHA * dense_scores + (1 - ALPHA) * sparse_scores
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        all_results.append([dense_indices[i] for i in sorted_indices])

    rrf_scores = {}
    for results in all_results:
        for rank, idx in enumerate(results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank)

    final_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return final_indices[:top_k]


def gemini_rerank_topk(query: str, candidates: list[str], candidate_docids: list[str] | None = None) -> int:
    """Gemini로 top-k 후보 중 best_index 선택 (캐싱 포함)."""
    if len(candidates) <= 1:
        return 0

    # cache key
    h = hashlib.md5()
    h.update(query.encode("utf-8", errors="ignore"))
    if candidate_docids:
        h.update("|".join(candidate_docids).encode("utf-8", errors="ignore"))
    else:
        h.update("|".join(str(len(c)) for c in candidates).encode("utf-8", errors="ignore"))

    cache_path = CACHE_DIR / f"gemini_rerank_{h.hexdigest()}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return int(cached.get("best_index", 0))
        except Exception:
            pass

    system_prompt = """당신은 한국어 과학 지식 검색 전문가입니다.
사용자의 질문과 검색된 문서 후보(Candidate)들이 주어집니다.
질문에 대해 가장 정확하고, 직접적인 해답을 포함하고 있는 문서를 하나만 선택하세요.

선택 기준:
1. 질문의 핵심 의도에 완벽히 부합하는가?
2. 과학적 사실이 정확한가?
3. 질문에서 요구하는 구체적인 정보를 담고 있는가?

반드시 JSON 형식으로 {"best_index": 0} 와 같이 답변하세요."""

    candidate_text = ""
    for i, content in enumerate(candidates):
        candidate_text += f"Candidate {i}:\n{content[:2000]}\n\n"

    user_prompt = f"## 질문:\n{query}\n\n## 검색 후보:\n{candidate_text}"

    resp = gemini_client._call_with_retry(
        prompt=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=120,
        response_format={"type": "json_object"},
    )

    parsed = _safe_extract_json_object(resp) or {}
    best_index = parsed.get("best_index", 0)

    try:
        best_index = int(best_index)
    except Exception:
        best_index = 0

    if best_index < 0 or best_index >= len(candidates):
        best_index = 0

    try:
        cache_path.write_text(json.dumps({"best_index": best_index}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return best_index


# ==========================================
# 4. 실행 (Resume)
# ==========================================
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    processed_ids.add(json.loads(line)["eval_id"])
                except Exception:
                    continue
    except Exception:
        pass

limit = os.getenv("EVAL_LIMIT")
limit = int(limit) if limit and str(limit).isdigit() else None

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    written = 0
    for entry in tqdm(eval_data):
        eval_id = entry["eval_id"]
        if eval_id in processed_ids:
            continue

        messages = entry["msg"]

        if eval_id in EMPTY_IDS:
            f.write(json.dumps({"eval_id": eval_id, "topk": [], "answer": "검색이 필요하지 않은 질문입니다."}, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
        else:
            # 1) multi-query (Gemini)
            queries = get_multi_queries_gemini(messages)
            rerank_query = queries[0]

            # 2) retrieve candidates
            candidate_indices = hybrid_search_multi(queries, top_k=TOP_CANDIDATES)

            if candidate_indices:
                # 3) cross-encoder rerank
                pairs = [[rerank_query, doc_contents[idx]] for idx in candidate_indices]
                rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
                sorted_ranks = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)

                top_indices = [idx for idx, _ in sorted_ranks[:GEMINI_RERANK_TOPK]]
                top_docids = [doc_ids[idx] for idx in top_indices]
                top_contents = [doc_contents[idx] for idx in top_indices]

                # 4) gemini choose best among top-k
                best_idx = gemini_rerank_topk(rerank_query, top_contents, candidate_docids=top_docids)

                if best_idx >= len(top_indices) or best_idx < 0:
                    best_idx = 0

                best_doc_idx = top_indices.pop(best_idx)
                final_indices = [best_doc_idx] + top_indices

                final_ids = [doc_ids[idx] for idx in final_indices[:FINAL_TOPK]]

                res = {
                    "eval_id": eval_id,
                    "standalone_query": rerank_query,
                    "topk": final_ids,
                    "answer": ""  # 리더보드 채점은 topk 중심이라 답변은 비워둠
                }
            else:
                res = {"eval_id": eval_id, "topk": [], "answer": ""}

            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            written += 1

        if limit is not None and written >= limit:
            break

print(f"✅ Created/updated: {OUTPUT_FILE}")
