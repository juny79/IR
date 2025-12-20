import json
from models.llm_client import llm_client
from retrieval.hybrid_search import run_hybrid_search
from retrieval.es_connector import es

def answer_question_optimized(messages):
    res = {"standalone_query": "", "topk": [], "answer": ""}
    analysis = llm_client.analyze_query(messages)
    
    if analysis.tool_calls:
        query = json.loads(analysis.tool_calls[0].function.arguments)['standalone_query']
        res["standalone_query"] = query
        
        # ⭐ 전략 A: Sparse와 Reranker에 HyDE 적용
        hypothetical_answer = llm_client.generate_hypothetical_answer(query)
        
        # HyDE 확장 쿼리 생성
        if hypothetical_answer:
            hyde_query = f"{query}\n{hypothetical_answer}"
        else:
            hyde_query = query
        
        # Hybrid Search with Reranker 실행
        # - Sparse: HyDE 확장 쿼리 사용 (키워드 풍부화)
        # - Dense: 원본 쿼리 사용 (임베딩 품질 유지)
        # - Reranker: HyDE 쿼리 사용 (Sparse와 일관성 확보) ⭐
        final_ranked_results = run_hybrid_search(
            original_query=query,
            sparse_query=hyde_query,
            reranker_query=hyde_query  # 새로 추가
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