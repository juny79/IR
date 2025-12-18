import json
import traceback
from models.llm_client import llm_client
from retrieval.hybrid_search import run_hybrid_search

# LLM과 검색엔진을 활용한 RAG 구현 (최고점 전략 적용)
def answer_question_optimized(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}
    
    # 1. 의도 분석 및 쿼리 생성 (MAP 극대화 핵심)
    llm_message = llm_client.analyze_query(messages)
    
    if llm_message is None:
        # LLM 호출 실패 시
        response["answer"] = "죄송합니다. 시스템 오류로 답변을 생성할 수 없습니다."
        return response
    
    # 2. 검색 및 리랭킹/앙상블
    if hasattr(llm_message, 'tool_calls') and llm_message.tool_calls:
        # 검색이 필요한 경우 (Function Call 발생)
        tool_call = llm_message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query", "")
        
        response["standalone_query"] = standalone_query

        # 하이브리드 검색 및 Hard Voting 실행
        final_ranked_results = run_hybrid_search(standalone_query)
        
        retrieved_context = []
        for rst in final_ranked_results:
            retrieved_context.append(rst["content"]) # 문서 내용
            response["topk"].append(rst["docid"])
            response["references"].append({"score": rst["score"], "content": rst["content"]})

        # 3. 최종 QA (GPT-5.1/5.2 또는 Claude 4.5 Opus 활용)
        context_str = "\n---\n".join(retrieved_context)
        response["answer"] = llm_client.generate_answer(messages, context_str)
    
    # 검색이 필요하지 않은 경우 (비과학 상식 질문)
    else:
        # MAP 최고점 전략: 비과학 상식 질문은 topk가 비어있어야 높은 점수 획득
        response["topk"] = [] 
        response["answer"] = llm_message.content # LLM이 생성한 일상 대화 답변

    return response

# 평가 실행 함수 (기존 베이스라인과 동일)
def eval_rag(eval_filename, output_filename):
    # ... (기존 베이스라인 eval_rag 함수 코드 삽입) ...
    pass

# 파일명은 jsonl이지만 파일명은 csv 사용
# eval_rag("./data/eval.jsonl", "submission.csv")