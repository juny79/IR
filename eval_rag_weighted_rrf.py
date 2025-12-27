"""
🎯 동료 코드(MAP 0.9174) 핵심 전략 반영 버전

핵심 변경점:
1. 가중치 RRF (Weighted RRF): Dense에 더 높은 가중치
   - W3_WEIGHTS = [0.6, 0.3, 0.3, 1.6, 1.0, 1.0]
   - BM25: q1=0.6, q2=0.3, q3=0.3
   - Dense: q1=1.6, q2=1.0, q3=1.0
   
2. Multi-Query 3관점:
   - q1: 구체적 서술형 (가장 중요)
   - q2: 핵심 키워드 나열
   - q3: 유사 표현/다른 관점

3. 파라미터:
   - RRF_K = 60
   - TOP_CANDIDATES = 100
   - FINAL_TOPK = 5
"""

import json
import os
from models.solar_client import solar_client
from retrieval.es_connector import es, sparse_retrieve, dense_retrieve
from retrieval.reranker import reranker
from collections import defaultdict

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파라미터 설정 (동료 코드 기준)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RRF_K = 60
BM25_TOPN = 50
DENSE_TOPN = 50
TOP_CANDIDATES = 100
FINAL_TOPK = 5
HYDE_MAX_LENGTH = 200

# 가중치 RRF: [BM25_q1, BM25_q2, BM25_q3, Dense_q1, Dense_q2, Dense_q3]
W3_WEIGHTS = [0.6, 0.3, 0.3, 1.6, 1.0, 1.0]


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


USE_WEIGHTED_RRF = _env_bool("USE_WEIGHTED_RRF", True)
USE_MULTI_QUERY_3 = _env_bool("USE_MULTI_QUERY_3", True)


def reciprocal_rank_fusion_weighted(rank_lists, k=60, weights=None):
    """
    가중치 RRF (동료 코드 방식)
    
    Args:
        rank_lists: List[List[docid]] - 각 검색 결과의 docid 순위 리스트
        k: RRF 파라미터
        weights: 각 rank_list에 대한 가중치 (길이가 rank_lists와 같아야 함)
    
    Returns:
        정렬된 docid 리스트
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)
    
    # 길이 불일치 방어
    if len(rank_lists) != len(weights):
        m = min(len(rank_lists), len(weights))
        rank_lists = rank_lists[:m]
        weights = weights[:m]
    
    scores = {}
    for w, rank_list in zip(weights, rank_lists):
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank + 1))
    
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def generate_multi_query_3(query_text, messages=None):
    """
    3관점 Multi-Query 생성 (동료 코드 방식)
    
    Returns:
        queries: [구체적 서술형, 핵심 키워드, 유사 표현]
    """
    system_prompt = """당신은 과학 검색 전문가입니다. 사용자의 질문을 해결하기 위해 검색엔진에 입력할 '3가지 버전의 검색어'를 JSON으로 생성하세요.

[출력 JSON 형식]
{
    "is_science": true/false,
    "queries": [
        "구체적이고 완결된 서술형 질문 (가장 중요)",
        "핵심 키워드 나열 (명사 중심)",
        "유사한 의미의 다른 표현 질문"
    ],
    "hyde": "가설적 답변 (200자 이내, 문서에 있을 법한 내용)"
}

[판단 기준]
- is_science=true: 지식/과학/기술/역사/사회/문화 등 코퍼스에서 근거를 찾아야 하는 질문
- is_science=false: 인사/잡담/감정표현/메타대화 (안녕, 고마워, 너 누구야)"""

    try:
        if messages and isinstance(messages, list):
            call_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            call_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_text}
            ]
        
        response = solar_client.client.chat.completions.create(
            model=solar_client.model,
            messages=call_messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content
        result = json.loads(raw)
        
        is_science = bool(result.get("is_science", True))
        queries = result.get("queries", [])
        hyde = result.get("hyde", "")
        
        if not isinstance(queries, list):
            queries = []
        queries = [str(q).strip() for q in queries if str(q).strip()]
        
        # queries 비었으면 원본 쿼리로 fallback
        if not queries:
            queries = [query_text]
        
        # 3개까지만
        queries = queries[:3]
        
        return {
            "is_science": is_science,
            "queries": queries,
            "hyde": hyde
        }
        
    except Exception as e:
        # fallback
        return {
            "is_science": True,
            "queries": [query_text],
            "hyde": ""
        }


def get_documents_batch(docids):
    """여러 문서를 한번에 가져오기"""
    try:
        result = es.search(
            index="test",
            query={"terms": {"docid": docids}},
            size=len(docids),
            _source=["docid", "content"]
        )
        
        docs_dict = {}
        for hit in result['hits']['hits']:
            doc_id = hit['_source']['docid']
            content = hit['_source'].get('content', '')[:1000]
            docs_dict[doc_id] = content
        
        return docs_dict
    except:
        return {}


def answer_question_weighted_rrf(messages):
    """
    동료 코드(0.9174) 핵심 전략 반영 버전
    - 3관점 Multi-Query
    - 가중치 RRF (Dense > BM25)
    - TOP_CANDIDATES=100
    """
    res = {"standalone_query": "", "topk": [], "answer": ""}
    
    # 원본 사용자 질문
    original_user_query = ""
    try:
        if isinstance(messages, list) and messages:
            original_user_query = messages[-1].get('content', '')
        else:
            original_user_query = str(messages)
    except:
        original_user_query = str(messages)
    
    # Step 1: 3관점 Multi-Query + 게이팅 + HyDE 생성
    mq_result = generate_multi_query_3(original_user_query, messages)
    
    is_science = mq_result.get("is_science", True)
    queries = mq_result.get("queries", [original_user_query])
    hyde = mq_result.get("hyde", "")
    
    # 가장 구체적인 첫 번째 쿼리를 standalone_query로
    main_query = queries[0] if queries else original_user_query
    res["standalone_query"] = main_query
    
    # 게이팅: 비과학 질문은 topk=[]
    if not is_science:
        res["topk"] = []
        res["answer"] = solar_client.generate_answer(messages, "")
        return res
    
    # Step 2: 각 쿼리마다 BM25 + Dense 검색
    all_bm25_lists = []
    all_dense_lists = []
    
    for q in queries:
        # BM25 검색 (HyDE 확장)
        hyde_q = f"{q}\n{hyde}" if hyde else q
        bm25_res = sparse_retrieve(hyde_q, BM25_TOPN)
        bm25_docids = [hit['_source']['docid'] for hit in bm25_res['hits']['hits']]
        all_bm25_lists.append(bm25_docids)
        
        # Dense 검색 (SBERT)
        dense_res = dense_retrieve(q, DENSE_TOPN, "embeddings_sbert")
        dense_docids = [hit['_source']['docid'] for hit in dense_res['hits']['hits']]
        all_dense_lists.append(dense_docids)
    
    # Step 3: 가중치 RRF 융합
    # rank_lists = [BM25_q1, BM25_q2, BM25_q3, Dense_q1, Dense_q2, Dense_q3]
    rank_lists = all_bm25_lists + all_dense_lists
    
    # 3쿼리일 때만 W3 가중치 적용 (길이 6)
    if USE_WEIGHTED_RRF and len(rank_lists) == 6:
        weights = W3_WEIGHTS
    else:
        weights = [1.0] * len(rank_lists)
    
    candidate_docids = reciprocal_rank_fusion_weighted(
        rank_lists,
        k=RRF_K,
        weights=weights
    )
    
    top_candidates = candidate_docids[:TOP_CANDIDATES]
    
    if not top_candidates:
        res["topk"] = []
        res["answer"] = "문서를 찾지 못했습니다."
        return res
    
    # Step 4: 문서 내용 가져오기 + Reranker
    docs_dict = get_documents_batch(top_candidates)
    docs_with_content = [(doc_id, docs_dict.get(doc_id, ''))
                         for doc_id in top_candidates if docs_dict.get(doc_id)]
    
    if docs_with_content:
        # Reranker: 가장 구체적인 쿼리(main_query)로 리랭킹
        final_ranked = reranker.rerank(
            main_query,
            docs_with_content,
            top_k=FINAL_TOPK,
            batch_size=32
        )
    else:
        final_ranked = top_candidates[:FINAL_TOPK]
    
    res["topk"] = final_ranked
    
    # Step 5: 답변 생성
    context_docs = []
    for docid in final_ranked[:3]:
        content = docs_dict.get(docid, '')
        if content:
            context_docs.append(content)
    
    context = " ".join(context_docs)
    res["answer"] = solar_client.generate_answer(messages, context)
    
    return res


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import sys
    from tqdm import tqdm
    
    # 평가 데이터 로드 (JSONL 형식)
    eval_path = "/root/IR/data/eval.jsonl"
    
    eval_data = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))
    
    print(f"📊 평가 데이터: {len(eval_data)}개")
    print(f"⚙️ 설정:")
    print(f"   - USE_WEIGHTED_RRF: {USE_WEIGHTED_RRF}")
    print(f"   - USE_MULTI_QUERY_3: {USE_MULTI_QUERY_3}")
    print(f"   - W3_WEIGHTS: {W3_WEIGHTS}")
    print(f"   - RRF_K: {RRF_K}")
    print(f"   - TOP_CANDIDATES: {TOP_CANDIDATES}")
    
    results = []
    empty_count = 0
    
    for entry in tqdm(eval_data, desc="Processing"):
        eval_id = entry["eval_id"]
        messages = entry["msg"]
        
        result = answer_question_weighted_rrf(messages)
        
        if not result["topk"]:
            empty_count += 1
        
        results.append({
            "eval_id": eval_id,
            "standalone_query": result["standalone_query"],
            "topk": result["topk"],
            "answer": result["answer"],
            "references": []
        })
    
    # 결과 저장
    output_path = "submission_weighted_rrf.csv"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 결과 저장: {output_path}")
    print(f"📌 Empty topk: {empty_count}/{len(eval_data)} ({empty_count/len(eval_data)*100:.1f}%)")
