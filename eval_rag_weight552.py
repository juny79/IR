#!/usr/bin/env python3
"""
Phase 4D-Weight[5,5,2]: VOTING_WEIGHTS를 [5,4,2]에서 [5,5,2]로 변경
2위 문서에 더 많은 가중치 부여
"""

import json
from models.llm_client import llm_client
from models.solar_client import solar_client
from retrieval.hybrid_search import run_hybrid_search
from retrieval.es_connector import es

# 🎯 Phase 4D-Weight552: 가중치 조정
VOTING_WEIGHTS = [5, 5, 2]  # ⭐ [5,4,2] → [5,5,2]
USE_MULTI_EMBEDDING = True
USE_GEMINI_ONLY = False
TOP_K_RETRIEVE = 50
USE_RRF = False
RRF_K = 60
USE_GATING = True

def answer_question_optimized(messages):
    res = {"standalone_query": "", "topk": [], "answer": ""}
    analysis = llm_client.analyze_query(messages)
    
    if analysis.tool_calls:
        query = json.loads(analysis.tool_calls[0].function.arguments)['standalone_query']
        res["standalone_query"] = query
        
        hypothetical_answer = solar_client.generate_hypothetical_answer(query)
        
        if hypothetical_answer:
            hyde_query = f"{query}\n{hypothetical_answer}"
        else:
            hyde_query = query
        
        final_ranked_results = run_hybrid_search(
            original_query=query,
            sparse_query=hyde_query,
            reranker_query=query,
            voting_weights=VOTING_WEIGHTS,  # [5,5,2]
            use_multi_embedding=USE_MULTI_EMBEDDING,
            top_k_retrieve=TOP_K_RETRIEVE,
            use_gemini_only=USE_GEMINI_ONLY,
            use_rrf=USE_RRF,
            rrf_k=RRF_K
        )
        
        res["topk"] = final_ranked_results[:5]
        
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
        res["answer"] = solar_client.generate_answer(messages, context)
    else:
        res["standalone_query"] = ""
        res["topk"] = []
        res["answer"] = analysis.content
    
    return res
