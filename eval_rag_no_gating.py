import json
from models.llm_client import llm_client
from models.solar_client import solar_client
from retrieval.hybrid_search import run_hybrid_search
from retrieval.es_connector import es

# 🎯 Phase 4D-NoGating: 게이팅 정책 제거 테스트
# Phase 4D [5,4,2] 설정만 유지, topk=[] 정책 제거
VOTING_WEIGHTS = [5, 4, 2]  # Phase 4D 최고점 설정
USE_MULTI_EMBEDDING = True  # SBERT + Gemini embedding 조합
USE_GEMINI_ONLY = False
TOP_K_RETRIEVE = 50
USE_RRF = False
RRF_K = 60

def answer_question_optimized_no_gating(messages):
    res = {"standalone_query": "", "topk": [], "answer": ""}
    analysis = llm_client.analyze_query(messages)
    
    # ⭐ 게이팅 정책 제거: tool_calls 상관없이 항상 검색 수행
    query_text = ""
    if analysis.tool_calls:
        query_text = json.loads(analysis.tool_calls[0].function.arguments)['standalone_query']
    else:
        # 비과학 질문도 원본 메시지로 검색
        query_text = messages[0]['content']
    
    res["standalone_query"] = query_text
    
    # ⭐ Phase 4D: Solar Pro 2 HyDE
    hypothetical_answer = solar_client.generate_hypothetical_answer(query_text)
    
    if hypothetical_answer:
        hyde_query = f"{query_text}\n{hypothetical_answer}"
    else:
        hyde_query = query_text
    
    # Hybrid Search
    final_ranked_results = run_hybrid_search(
        original_query=query_text,
        sparse_query=hyde_query,
        reranker_query=query_text,
        voting_weights=VOTING_WEIGHTS,
        use_multi_embedding=USE_MULTI_EMBEDDING,
        top_k_retrieve=TOP_K_RETRIEVE,
        use_gemini_only=USE_GEMINI_ONLY,
        use_rrf=USE_RRF,
        rrf_k=RRF_K
    )
    
    # 검색 결과 설정: 항상 반환 (게이팅 정책 없음)
    res["topk"] = final_ranked_results[:5]
    
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
    # Phase 4D: Solar Pro 2로 최종 답변 생성
    res["answer"] = solar_client.generate_answer(messages, context)
    
    return res


if __name__ == "__main__":
    # 테스트
    test_messages = [{'role': 'user', 'content': '광합성이란?'}]
    print('=== Phase 4D-NoGating 테스트 ===')
    print('설정:')
    print('  - Phase 4D 기본 설정 [5,4,2]')
    print('  - 게이팅 정책 제거')
    print('  - 모든 질문에 대해 검색 수행\n')
    
    result = answer_question_optimized_no_gating(test_messages)
    print(f'원본 쿼리: {result["standalone_query"]}')
    print(f'Top-5 문서: {len(result["topk"])}개')
    print(f'답변: {result["answer"][:100]}...')
