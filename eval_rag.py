import json
import os
from models.solar_client import solar_client
from retrieval.hybrid_search import run_hybrid_search
from retrieval.es_connector import es

# 🎯 Phase 4D: 고성능 하이브리드 검색 (게이팅 없음)
# MAP 0.8424 달성 설정
# - Solar Pro 2 HyDE (300자)
# - Hard Voting [5,4,2]
# - SBERT + Gemini embedding
# - 게이팅 정책 제거 → 모든 질문 검색 수행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "t", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default


    def _env_float(name: str, default: float) -> float:
        v = os.getenv(name)
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default
    try:
        return int(v)
    except Exception:
        return default


def _env_json_list(name: str, default_list):
    v = os.getenv(name)
    if not v:
        return default_list
    try:
        parsed = json.loads(v)
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            return parsed
    except Exception:
        pass
    return default_list


# 기본값(현재 최고점 근처 설정). 실험 러너에서 환경변수로 오버라이드 가능.
VOTING_WEIGHTS = _env_json_list("VOTING_WEIGHTS_JSON", [5, 4, 2])
USE_MULTI_EMBEDDING = _env_bool("USE_MULTI_EMBEDDING", True)   # SBERT + Gemini
USE_GEMINI_ONLY = _env_bool("USE_GEMINI_ONLY", False)
TOP_K_RETRIEVE = _env_int("TOP_K_RETRIEVE", 80)
USE_RRF = _env_bool("USE_RRF", False)
RRF_K = _env_int("RRF_K", 60)
USE_GATING = _env_bool("USE_GATING", False)
HYDE_MAX_LENGTH = _env_int("HYDE_MAX_LENGTH", 200)
USE_SOLAR_ANALYZER = _env_bool("USE_SOLAR_ANALYZER", True)

# is_science=false일 때(topk=[] 정책) 적용할 최소 신뢰도. 낮으면 검색으로 강제(오판 방지).
NO_SEARCH_CONFIDENCE_THRESHOLD = _env_float("NO_SEARCH_CONFIDENCE_THRESHOLD", 0.0)

# 후보군(리랭커 입력) 크기
CANDIDATE_POOL_SIZE = _env_int("CANDIDATE_POOL_SIZE", 80)

def answer_question_optimized(messages):
    res = {"standalone_query": "", "topk": [], "answer": ""}
    solar_analysis = None
    # 원문 사용자 질문(검색/리랭커 쿼리로 사용): LLM rewrite로 인한 성능 저하 방지
    original_user_query = ""
    try:
        if isinstance(messages, list) and messages:
            original_user_query = messages[-1].get('content', '')
        else:
            original_user_query = str(messages)
    except Exception:
        original_user_query = str(messages)
    if USE_SOLAR_ANALYZER:
        solar_analysis = solar_client.analyze_query_and_hyde(messages, hyde_max_chars=HYDE_MAX_LENGTH)
        is_science_question = bool(solar_analysis.get("is_science", False))
        solar_confidence = float(solar_analysis.get("confidence", 0.0) or 0.0)
        # 제출용(standalone_query)은 Solar 정규화 결과 사용
        query_text = solar_analysis.get("standalone_query", "") or original_user_query
    else:
        # (레거시) 안전장치: Solar analyzer 비활성화 시 모든 질문 검색
        is_science_question = True
        solar_confidence = 1.0
        query_text = original_user_query
    
    res["standalone_query"] = query_text
    
    # 비과학(검색 불필요) 질문: topk=[]
    # 단, 신뢰도가 낮으면 오판 방지를 위해 검색으로 강제
    if (not is_science_question) and (solar_confidence >= NO_SEARCH_CONFIDENCE_THRESHOLD):
        res["topk"] = []
        if solar_analysis and solar_analysis.get("direct_answer"):
            res["answer"] = solar_analysis.get("direct_answer")
        else:
            res["answer"] = solar_client.generate_answer(messages, "")
        return res
    
    # Solar analyzer가 HyDE를 함께 생성했다면 재사용(LLM 호출 1회 절감)
    hypothetical_answer = ""
    if solar_analysis and solar_analysis.get("hyde"):
        hypothetical_answer = solar_analysis.get("hyde")
    else:
        hypothetical_answer = solar_client.generate_hypothetical_answer(query_text)
    
    # HyDE 확장 쿼리 생성
    if hypothetical_answer:
        # Sparse는 정규화 standalone_query + HyDE로 재현율 확보
        hyde_query = f"{query_text}\n{hypothetical_answer}"
    else:
        hyde_query = query_text
    
    # Hybrid Search with Reranker 실행 (Phase 4D: [5,4,2] + TopK=50)
    final_ranked_results = run_hybrid_search(
        # Dense/리랭커는 원문 질문 유지(LLM rewrite로 의미가 바뀌는 리스크 감소)
        original_query=original_user_query,
        sparse_query=hyde_query,
        reranker_query=original_user_query,
        voting_weights=VOTING_WEIGHTS,  # [5, 4, 2]
        use_multi_embedding=USE_MULTI_EMBEDDING,  # SBERT + Gemini
        top_k_retrieve=TOP_K_RETRIEVE,  # 50개
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        use_gemini_only=USE_GEMINI_ONLY,
        use_rrf=USE_RRF,  # False (Hard Voting)
        rrf_k=RRF_K,
        multi_queries=[]  # Phase 4D: 멀티 쿼리 없음
    )
    
    # 검색 결과 설정: 항상 반환 (게이팅 없음)
    res["topk"] = final_ranked_results[:5]  # 상위 5개
    
    # 컨텍스트 생성: Top-3 문서 내용 사용
    context_docs = []
    for docid in final_ranked_results[:3]:
        search_result = es.search(
            index="test",
            query={"term": {"docid": docid}},
            size=1
        )
        if search_result['hits']['hits']:
            context_docs.append(search_result['hits']['hits'][0]['_source']['content'])
    
    context = " ".join(context_docs)
    # Solar Pro 2로 최종 답변 생성
    res["answer"] = solar_client.generate_answer(messages, context)
    
    return res