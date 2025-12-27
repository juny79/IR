import json
import os
from tqdm import tqdm
from retrieval.e5_search import search_e5
from models.solar_client import solar_client

# 설정
EVAL_FILE = "/root/IR/data/eval.jsonl"
DOCS_FILE = "/root/IR/data/documents.jsonl"
OUTPUT_FILE = "submission_e5_multi.csv"
TOP_K = 5

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
    3관점 Multi-Query 생성 (동료 코드 방식)
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
    
    # SolarClient를 사용하여 JSON 생성 시도
    # SolarClient.generate_answer는 일반 텍스트 반환이므로, 프롬프트를 조정하여 JSON을 유도해야 함.
    # 하지만 SolarClient에는 chat completion 인터페이스가 명시적으로 노출되지 않았음 (requests 직접 사용).
    # 따라서 _call_with_retry를 직접 호출하거나, generate_answer를 변형해서 사용해야 함.
    
    # 여기서는 solar_client._call_with_retry를 사용하여 구현
    prompt = f"{system_prompt}\n\n사용자 질문: {query_text}"
    
    try:
        # JSON 포맷 강제를 위해 프롬프트에 명시
        response_text = solar_client._call_with_retry(prompt, temperature=0.1)
        
        # JSON 파싱 시도
        # 응답이 ```json ... ``` 형태일 수 있음
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text
            
        data = json.loads(json_str)
        queries = data.get("queries", [])
        if not queries:
            return [query_text]
        return queries[:3]
        
    except Exception as e:
        print(f"⚠️ Multi-Query 생성 실패: {e}")
        return [query_text]

def main():
    print(f"🚀 E5 Multi-Query Evaluation Start")
    print(f"📂 Input: {EVAL_FILE}")
    print(f"💾 Output: {OUTPUT_FILE}")
    
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        pass
        
    for line in tqdm(lines, desc="Processing"):
        entry = json.loads(line)
        eval_id = entry["eval_id"]
        messages = entry["msg"]
        
        # 마지막 사용자 질문 추출
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content")
                break
        
        if not user_query:
            user_query = str(messages)
            
        # 1. Multi-Query 생성
        queries = generate_multi_query_3(user_query)
        # 원본 쿼리도 포함 (가중치 조절 가능)
        if user_query not in queries:
            queries.insert(0, user_query)
            
        # 2. 검색 및 점수 합산
        doc_scores = {}
        for q in queries:
            # 각 쿼리당 Top-10 검색
            results = search_e5(q, top_k=10)
            for res in results:
                doc_id = res["docid"]
                score = res["score"]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += score # 단순 합산 (Soft Voting)
        
        # 3. 정렬 및 Top-K 추출
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        topk_ids = [doc_id for doc_id, score in sorted_docs]
        
        # 4. 컨텍스트 구성
        context_parts = []
        for i, doc_id in enumerate(topk_ids):
            content = doc_map.get(doc_id, "")
            context_parts.append(f"[{i+1}] {content}")
        
        context = "\n\n".join(context_parts)
        
        # 5. 답변 생성 (Solar)
        answer = solar_client.generate_answer(messages, context)
        if not answer:
            answer = "죄송합니다. 답변을 생성할 수 없습니다."
            
        # 6. 결과 저장
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
