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
        
        # Hybrid Search 실행 (5:3:1 가중치 앙상블)
        final_ranked_results = run_hybrid_search(query)
        
        # Top-K docid 추출
        res["topk"] = [doc["docid"] for doc in final_ranked_results]
        
        # 컨텍스트 생성: Top-3 문서 내용 사용
        context_docs = []
        for doc in final_ranked_results[:3]:
            # ES에서 docid 필드로 검색하여 실제 content 가져오기
            search_result = es.search(
                index="test",
                query={"term": {"docid": doc["docid"]}},
                size=1
            )
            if search_result['hits']['hits']:
                context_docs.append(search_result['hits']['hits'][0]['_source']['content'])
        
        context = " ".join(context_docs)
        res["answer"] = llm_client.generate_answer(messages, context)
    else:
        res["answer"] = analysis.content # 일상 대화 응답
    
    return res