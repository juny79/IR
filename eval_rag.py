import json
from models.llm_client import llm_client
from retrieval.hybrid_search import run_hybrid_search
from retrieval.es_connector import es

# 🎯 파라미터: Hard Voting 가중치 (환경에 따라 변경)
VOTING_WEIGHTS = [7, 4, 2]  # 테스트: [7, 4, 2] (기본: [5, 3, 1], 튜닝: [6, 3, 1])

def answer_question_optimized(messages):
    res = {"standalone_query": "", "topk": [], "answer": ""}
    analysis = llm_client.analyze_query(messages)
    
    if analysis.tool_calls:
        query = json.loads(analysis.tool_calls[0].function.arguments)['standalone_query']
        res["standalone_query"] = query
        
        # ⭐ Phase 2: HyDE를 전체에 적용 (일관된 파이프라인)
        hypothetical_answer = llm_client.generate_hypothetical_answer(query)
        
        # HyDE 확장 쿼리 생성
        if hypothetical_answer:
            hyde_query = f"{query}\n{hypothetical_answer}"
        else:
            hyde_query = query
        
        # Hybrid Search with Reranker 실행 (HyDE 전체 적용)
        # - Sparse: HyDE 확장 쿼리 사용
        # - Dense: HyDE 확장 쿼리 사용 (일관성)
        # - Reranker: 원본 쿼리 사용 (정확한 relevance 판단) ⭐
        # - Hard Voting: 최적화된 가중치 사용
        final_ranked_results = run_hybrid_search(
            original_query=query,
            sparse_query=hyde_query,
            reranker_query=query,  # 원본 쿼리로 복구
            voting_weights=VOTING_WEIGHTS  # 파라미터 튜닝용 ⭐
        )
        
        # final_ranked_results는 이제 docid 리스트 형태
        res["topk"] = final_ranked_results[:5]  # 상위 5개
        
        # 컨텍스트 생성: Top-3 문서 내용 사용
        context_docs = []
        for docid in final_ranked_results[:3]:
            # ES에서 docid 필드로 검색하여 실제 content 가져오기
            search_result = es.search(
                index="test",
                query={"term": {"docid": docid}},
                size=1
            )
            if search_result['hits']['hits']:
                context_docs.append(search_result['hits']['hits'][0]['_source']['content'])
        
        context = " ".join(context_docs)
        res["answer"] = llm_client.generate_answer(messages, context)
    else:
        res["answer"] = analysis.content # 일상 대화 응답
    
    return res