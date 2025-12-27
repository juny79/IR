import json
import os
from tqdm import tqdm
from retrieval.e5_search import search_e5
from models.solar_client import solar_client

# 설정
EVAL_FILE = "/root/IR/data/eval.jsonl"
DOCS_FILE = "/root/IR/data/documents.jsonl"
OUTPUT_FILE = "submission_e5_base.csv"
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

def main():
    print(f"🚀 E5 Base Evaluation Start")
    print(f"📂 Input: {EVAL_FILE}")
    print(f"💾 Output: {OUTPUT_FILE}")
    
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
        
        # 1. 검색 (E5)
        search_results = search_e5(user_query, top_k=TOP_K)
        topk_ids = [res["docid"] for res in search_results]
        
        # 2. 컨텍스트 구성
        context_parts = []
        for i, res in enumerate(search_results):
            doc_id = res["docid"]
            content = doc_map.get(doc_id, "")
            context_parts.append(f"[{i+1}] {content}")
        
        context = "\n\n".join(context_parts)
        
        # 3. 답변 생성 (Solar)
        answer = solar_client.generate_answer(messages, context)
        if not answer:
            answer = "죄송합니다. 답변을 생성할 수 없습니다."
            
        # 4. 결과 저장
        result = {
            "eval_id": eval_id,
            "standalone_query": user_query,
            "topk": topk_ids,
            "answer": answer,
            "references": [] # 제출 포맷에 맞춤
        }
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
