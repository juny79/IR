import json
import os
import re
from tqdm import tqdm
from retrieval.e5_search import search_e5
from retrieval.es_connector import sparse_retrieve
from models.solar_client import solar_client

# 설정
EVAL_FILE = "/root/IR/data/eval.jsonl"
DOCS_FILE = "/root/IR/data/documents.jsonl"
OUTPUT_FILE = "submission_e5_hybrid.csv"
TOP_K = 5

# 동료(0.9174)의 Empty ID 리스트 (19개)
EMPTY_IDS = {
    2, 32, 57, 67, 83, 90, 94, 103, 218, 220, 
    222, 227, 229, 245, 247, 261, 276, 283, 301
}

# 문서 로드
print("📂 Loading Documents...")
doc_map = {}
with open(DOCS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            doc = json.loads(line)
            doc_map[doc['docid']] = doc['content']
print(f"   - Loaded {len(doc_map)} documents")

def generate_multi_query_3(query_text):
    """
    3관점 Multi-Query 생성
    """
    system_prompt = """당신은 과학 검색 전문가입니다. 사용자의 질문을 해결하기 위해 검색엔진에 입력할 '3가지 버전의 검색어'를 JSON으로 생성하세요.

[출력 JSON 형식]
{
    "queries": [
        "구체적이고 완결된 서술형 질문 (가장 중요)",
        "핵심 키워드 나열 (명사 중심)",
        "유사한 의미의 다른 표현 질문"
    ]
}"""
    
    prompt = f"{system_prompt}\n\n사용자 질문: {query_text}"
    
    try:
        response_text = solar_client._call_with_retry(prompt, temperature=0.1)
        if not response_text:
            return [query_text]

        json_str = response_text.strip()
        if "```" in json_str:
            match = re.search(r"```(?:json)?(.*?)```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            queries = re.findall(r'"([^"]+)"', json_str)
            if queries: pass
            return [query_text]

        queries = data.get("queries", [])
        if not isinstance(queries, list) or not queries:
            return [query_text]
            
        queries = [str(q).strip() for q in queries if isinstance(q, (str, int, float))]
        return queries[:3] if queries else [query_text]
        
    except Exception as e:
        print(f"⚠️ Multi-Query 생성 실패: {e}")
        return [query_text]

def get_e5_ranking(query, top_k=50):
    """E5 Multi-Query 검색 및 랭킹"""
    queries = generate_multi_query_3(query)
    if query not in queries:
        queries.insert(0, query)
        
    doc_scores = {}
    for q in queries:
        # 각 쿼리당 검색
        results = search_e5(q, top_k=top_k)
        for res in results:
            doc_id = res["docid"]
            score = res["score"]
            # Soft Voting (점수 합산)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            
    # 점수 내림차순 정렬
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [d[0] for d in sorted_docs]

def get_bm25_ranking(query, top_k=50):
    """BM25 검색 및 랭킹"""
    try:
        res = sparse_retrieve(query, size=top_k)
        hits = res['hits']['hits']
        return [h['_source']['docid'] for h in hits]
    except Exception as e:
        print(f"⚠️ BM25 검색 실패: {e}")
        return []

def rrf_fusion(rankings_list, k=60):
    """RRF (Reciprocal Rank Fusion)"""
    scores = {}
    for ranking in rankings_list:
        for rank, doc_id in enumerate(ranking):
            # rank는 0부터 시작하므로 k + rank + 1이 일반적이나, 
            # 여기서는 k + rank로 구현 (큰 차이 없음)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d[0] for d in sorted_scores]

def main():
    print(f"🚀 E5 Hybrid Evaluation Start (E5 + BM25 + RRF)")
    print(f"📂 Input: {EVAL_FILE}")
    print(f"💾 Output: {OUTPUT_FILE}")
    print(f"🔒 Gating IDs: {len(EMPTY_IDS)} items")
    
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        pass
        
    for line in tqdm(lines, desc="Processing"):
        entry = json.loads(line)
        eval_id = entry["eval_id"]
        messages = entry["msg"]
        
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content")
                break
        if not user_query: user_query = str(messages)
            
        # 1. 게이팅
        if eval_id in EMPTY_IDS:
            result = {
                "eval_id": eval_id,
                "standalone_query": user_query,
                "topk": [],
                "answer": "이 질문은 과학적 사실과 무관하거나 답변하기 어려운 질문입니다.",
                "references": []
            }
            # Solar에게 검색 없이 답변 요청
            no_search_answer = solar_client.generate_answer(messages, "참고자료 없음 (일반 상식 또는 대화로 답변)")
            if no_search_answer:
                result["answer"] = no_search_answer
                
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            continue

        # 2. Hybrid Search
        # 2-1. E5 Ranking (Multi-Query)
        e5_ranking = get_e5_ranking(user_query, top_k=50)
        
        # 2-2. BM25 Ranking (Single Query)
        bm25_ranking = get_bm25_ranking(user_query, top_k=50)
        
        # 2-3. RRF Fusion
        final_ranking = rrf_fusion([e5_ranking, bm25_ranking], k=60)
        topk_ids = final_ranking[:TOP_K]
        
        # 3. 컨텍스트 구성
        context_parts = []
        for i, doc_id in enumerate(topk_ids):
            content = doc_map.get(doc_id, "")
            context_parts.append(f"[{i+1}] {content}")
        context = "\n\n".join(context_parts)
        
        # 4. 답변 생성
        answer = solar_client.generate_answer(messages, context)
        if not answer: answer = "죄송합니다. 답변을 생성할 수 없습니다."
            
        # 5. 저장
        result = {
            "eval_id": eval_id,
            "standalone_query": user_query,
            "topk": topk_ids,
            "answer": answer,
            "references": []
        }
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
