# main.py
import json
import os
from eval_rag import answer_question_optimized

def run_evaluation(eval_path, output_path):
    """
    eval.jsonl을 읽어 RAG 시스템을 실행하고 submission.csv를 생성합니다.
    """
    if not os.path.exists(eval_path):
        print(f"에러: {eval_path} 경로에 파일이 없습니다.")
        return

    results = []
    with open(eval_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"총 {len(lines)}개의 평가 데이터 처리를 시작합니다.")

    with open(output_path, "w", encoding="utf-8") as of:
        for i, line in enumerate(lines):
            data = json.loads(line)
            eval_id = data["eval_id"]
            messages = data["msg"]
            
            print(f"[{i+1}/{len(lines)}] Eval ID {eval_id} 처리 중...")
            
            # 최적화된 RAG 로직 호출 (의도분석 -> 검색/앙상블 -> QA)
            response = answer_question_optimized(messages)
            
            # 제출 형식 구성 (JSON Lines 형태지만 확장자는 .csv 요구사항 반영)
            output = {
                "eval_id": eval_id,
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            
            of.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"최종 결과물이 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    # 데이터 경로 설정
    EVAL_DATA_PATH = "./data/eval.jsonl"
    OUTPUT_FILE_PATH = "submission.csv"
    
    run_evaluation(EVAL_DATA_PATH, OUTPUT_FILE_PATH)