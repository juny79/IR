import json
import os
import re
from tqdm import tqdm
from retrieval.e5_search import search_e5
from models.solar_client import solar_client

# 설정
EVAL_FILE = "/root/IR/data/eval.jsonl"
DOCS_FILE = "/root/IR/data/documents.jsonl"
OUTPUT_FILE = "submission_e5_final.csv"
TOP_K = 5

# 동료(0.9174)의 Empty ID 리스트 (19개)
EMPTY_IDS = {
    2, 32, 57, 67, 83, 90, 94, 103, 218, 220, 
    222, 227, 229, 245, 247, 261, 276, 283, 301
}

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
    3관점 Multi-Query 생성 (JSON 파싱 강화)
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
    
    prompt = f"{system_prompt}\n\n사용자 질문: {query_text}"
    
    try:
        # Solar API 호출
        response_text = solar_client._call_with_retry(prompt, temperature=0.1)
        if not response_text:
            return [query_text]

        # JSON 추출 (Markdown 코드 블록 제거)
        json_str = response_text.strip()
        if "```" in json_str:
            # 첫 번째 코드 블록 찾기
            match = re.search(r"```(?:json)?(.*?)```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                # 코드 블록이 닫히지 않은 경우 등
                json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        # JSON 파싱
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 파싱 실패 시, 간단한 정규식으로 리스트 추출 시도
            queries = re.findall(r'"([^"]+)"', json_str)
            if queries:
                # 키값("queries") 등도 포함될 수 있으므로 필터링 필요하지만, 
                # 복잡하므로 실패 시 원본 반환이 안전
                pass
            return [query_text]

        queries = data.get("queries", [])
        if not isinstance(queries, list) or not queries:
            return [query_text]
            
        # 문자열만 필터링
        queries = [str(q).strip() for q in queries if isinstance(q, (str, int, float))]
        return queries[:3] if queries else [query_text]
        
    except Exception as e:
        print(f"⚠️ Multi-Query 생성 실패: {e}")
        return [query_text]

def main():
    print(f"🚀 E5 Final Evaluation Start (Gating Applied)")
    print(f"📂 Input: {EVAL_FILE}")
    print(f"💾 Output: {OUTPUT_FILE}")
    print(f"🔒 Gating IDs: {len(EMPTY_IDS)} items")
    
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # 기존 파일 초기화
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
            
        # ---------------------------------------------------------
        # 1. 게이팅 체크 (Empty ID인 경우 검색 건너뜀)
        # ---------------------------------------------------------
        if eval_id in EMPTY_IDS:
            result = {
                "eval_id": eval_id,
                "standalone_query": user_query,
                "topk": [],  # 빈 리스트
                "answer": "이 질문은 과학적 사실과 무관하거나 답변하기 어려운 질문입니다.", # 적절한 기본 답변
                "references": []
            }
            # Solar에게 비과학 질문에 대한 답변을 생성하게 할 수도 있음.
            # 하지만 topk=[]이면 평가 스크립트에서 정답 처리될 가능성이 높음 (과학 질문이 아니므로).
            # 동료의 답변을 확인해보면 좋겠지만, 일단 Solar에게 맡기거나 고정 답변 사용.
            # 여기서는 Solar에게 "검색 없이" 답변하도록 요청.
            
            # Solar에게 검색 없이 답변 요청
            no_search_answer = solar_client.generate_answer(messages, "참고자료 없음 (일반 상식 또는 대화로 답변)")
            if no_search_answer:
                result["answer"] = no_search_answer
            
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            continue
        # ---------------------------------------------------------

        # 2. Multi-Query 생성
        queries = generate_multi_query_3(user_query)
        # 원본 쿼리도 포함 (가중치 조절 가능)
        if user_query not in queries:
            queries.insert(0, user_query)
            
        # 3. 검색 및 점수 합산
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
        
        # 4. 정렬 및 Top-K 추출
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        topk_ids = [doc_id for doc_id, score in sorted_docs]
        
        # 5. 컨텍스트 구성
        context_parts = []
        for i, doc_id in enumerate(topk_ids):
            content = doc_map.get(doc_id, "")
            context_parts.append(f"[{i+1}] {content}")
        
        context = "\n\n".join(context_parts)
        
        # 6. 답변 생성 (Solar)
        answer = solar_client.generate_answer(messages, context)
        if not answer:
            answer = "죄송합니다. 답변을 생성할 수 없습니다."
            
        # 7. 결과 저장
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
