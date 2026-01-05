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
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
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
USE_MULTI_QUERY = _env_bool("USE_MULTI_QUERY", False)

# is_science=false일 때(topk=[] 정책) 적용할 최소 신뢰도. 낮으면 검색으로 강제(오판 방지).
NO_SEARCH_CONFIDENCE_THRESHOLD = _env_float("NO_SEARCH_CONFIDENCE_THRESHOLD", 0.0)

# intent 기반으로 topk=[] 케이스를 더 잡아내는 옵션(기본 off: 기존 최고점 재현 안전)
ENABLE_INTENT_NO_SEARCH = _env_bool("ENABLE_INTENT_NO_SEARCH", False)
# intent/규칙 기반 보조 게이팅을 적용할 때 사용하는 임계값(보수적으로 시작 권장)
NO_SEARCH_STRICT_THRESHOLD = _env_float("NO_SEARCH_STRICT_THRESHOLD", NO_SEARCH_CONFIDENCE_THRESHOLD)
NO_SEARCH_HEURISTIC_THRESHOLD = _env_float("NO_SEARCH_HEURISTIC_THRESHOLD", 0.85)
NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF = _env_float("NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF", 0.10)

# 후보군(리랭커 입력) 크기
CANDIDATE_POOL_SIZE = _env_int("CANDIDATE_POOL_SIZE", 80)


def _last_user_text(messages) -> str:
    try:
        if isinstance(messages, list) and messages:
            for m in reversed(messages):
                if str(m.get("role", "user")) == "user":
                    return str(m.get("content", "") or "")
        return str(messages)
    except Exception:
        return str(messages)


def _looks_like_science(text: str) -> bool:
    t = (text or "").lower()
    science_markers = [
        # Korean science/tech keywords
        "광합성", "dna", "rna", "세포", "미토콘드리아", "단백질", "효소", "유전자",
        "뉴턴", "법칙", "물리", "화학", "생물", "지구과학", "천문", "우주", "양자",
        "전기", "자기", "전자", "원자", "분자", "반응", "촉매", "산화", "환원",
        "알고리즘", "복잡도", "빅오", "머신러닝", "딥러닝", "신경망",
    ]
    return any(m in t for m in science_markers)


def _looks_like_knowledge_query(text: str) -> bool:
    """지식 검색이 필요한 질문 패턴 감지 (게이팅 오버라이드용)
    
    주의: AI 메타 질문("너 뭐야?", "너 잘하는게 뭐야?")은 제외해야 함
    """
    t = (text or "").lower()
    
    # AI 메타 질문 패턴 제외 (검색 불필요)
    ai_meta_patterns = ["너 ", "넌 ", "니가 ", "너는 ", "네가 "]
    if any(p in t for p in ai_meta_patterns):
        return False
    
    # 일상 대화/감정 표현 제외 (검색 불필요)
    casual_patterns = ["힘드", "우울", "기분", "즐거", "신나", "반가", "안녕"]
    if any(p in t for p in casual_patterns) and len(t) < 30:
        return False
    
    # 장점/단점/효과/이유 등을 묻는 질문은 문서 검색이 도움될 수 있음
    knowledge_markers = [
        # 분석/평가 질문
        "좋은 점", "나쁜 점", "장점", "단점", "이점", "혜택", "효과", "가치",
        "원인", "이유", "방법", "방안", "과정", "원리", "구조", "특징",
        "차이", "비교", "정의", "개념", "역사", "유래", "발전",
        "종류", "분류", "예시", "사례", "영향", "결과", "현황", "상황",
        # 의문사
        "무엇", "뭐야", "뭐지", "뭔가", "어떻게", "왜", "얼마나",
        "어떤", "어디", "언제",
        # 요청형
        "알려", "설명", "조사", "연구", "말해", "가르쳐",
        # 지식 주제
        "자유", "권리", "제도", "정책", "법률", "경제", "사회", "문화",
        "과학", "기술", "환경", "건강", "교육", "역사",
    ]
    return any(m in t for m in knowledge_markers)


def _looks_like_chitchat(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True

    # very short messages are often chit-chat/meta
    if len(t) <= 6 and any(x in t for x in ["안녕", "hi", "hello", "ㅇㅇ", "ㄴㄴ", "ㅋㅋ", "ㅎㅎ"]):
        return True

    markers = [
        # greetings
        "안녕", "반가", "좋은 아침", "좋은밤", "굿모닝", "굿나잇",
        # thanks/apology
        "고마", "감사", "죄송", "미안",
        # emotions
        "힘들", "우울", "기분", "짜증", "행복", "슬퍼",
        # daily talk
        "날씨", "밥", "점심", "저녁", "뭐해", "뭐함",
        # laughter/slang
        "ㅋㅋ", "ㅎㅎ", "lol",
        # assistant meta
        "너는 누구", "넌 누구", "너 누구", "할 수 있어", "가능해", "도와줘",
    ]
    return any(m in t for m in markers)

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
    last_user_text = _last_user_text(messages)
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
    # - 기본: Solar is_science=false + confidence 임계값
    # - 옵션(ENABLE_INTENT_NO_SEARCH=true): 규칙 기반(잡담/메타) 보조 게이팅으로 추가 케이스 포착
    # - 지식 질문 패턴 감지 시 검색 강제(게이팅 오버라이드)
    predicted_no_search = False
    force_search = _looks_like_knowledge_query(last_user_text)
    
    if force_search:
        # 지식 질문 패턴이 감지되면 검색 강제
        predicted_no_search = False
    elif (not is_science_question) and (solar_confidence >= NO_SEARCH_STRICT_THRESHOLD):
        predicted_no_search = True
    elif ENABLE_INTENT_NO_SEARCH:
        rule_chitchat = _looks_like_chitchat(last_user_text)
        rule_science_block = _looks_like_science(last_user_text)
        if rule_chitchat and (not rule_science_block):
            # Solar도 비과학으로 보거나(conf 충분) / 또는 Solar가 과학이라 해도 확신이 매우 낮으면 override
            if ((not is_science_question) and (solar_confidence >= NO_SEARCH_HEURISTIC_THRESHOLD)) or (
                is_science_question and (solar_confidence <= NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF)
            ):
                predicted_no_search = True

    if predicted_no_search:
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
    multi_queries = []
    if USE_MULTI_QUERY:
        try:
            # 멀티 쿼리는 standalone_query 기반(원문 의미 보존)
            multi_queries = solar_client.generate_multi_query(query_text) or []
        except Exception:
            multi_queries = []

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
        multi_queries=multi_queries
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