import json
from eval_rag import answer_question_optimized

def main():
    # 이미 처리된 eval_id들 로드
    processed_ids = set()
    try:
        with open("./submission.csv", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data["eval_id"])
    except:
        pass
    
    with open("./data/eval.jsonl", "r", encoding="utf-8") as f, \
         open("submission.csv", "a", encoding="utf-8") as of:
        
        for i, line in enumerate(f, 1):
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