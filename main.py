import json
import os
from eval_rag import answer_question_optimized

def main():
    submission_file = os.getenv("SUBMISSION_FILE", "submission.csv")
    eval_file = os.getenv("EVAL_FILE", "./data/eval.jsonl")
    # 0이면 전체(220) 처리, 양수면 앞에서부터 N개만 처리(빠른 드라이런/프리필터용)
    try:
        eval_limit = int(os.getenv("EVAL_LIMIT", "0"))
    except Exception:
        eval_limit = 0

    # 이미 처리된 eval_id들 로드
    processed_ids = set()
    try:
        with open(submission_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data["eval_id"])
    except:
        pass
    
    with open(eval_file, "r", encoding="utf-8") as f, \
         open(submission_file, "a", encoding="utf-8") as of:
        
        for i, line in enumerate(f, 1):
            if eval_limit and i > eval_limit:
                break
            data = json.loads(line)
            
            # 이미 처리된 항목은 건너뛰기
            if data["eval_id"] in processed_ids:
                continue
            
            print(f"[{i}/220] ID {data['eval_id']} 처리 중...")
            
            try:
                result = answer_question_optimized(data["msg"])
                
                output = {
                    "eval_id": data["eval_id"],
                    "standalone_query": result["standalone_query"],
                    "topk": result["topk"],
                    "answer": result["answer"]
                }
                of.write(json.dumps(output, ensure_ascii=False) + "\n")
                of.flush()
                processed_ids.add(data["eval_id"])
            except Exception as e:
                print(f"오류 [ID {data['eval_id']}]: {str(e)[:100]}")
                continue

if __name__ == "__main__":
    main()