import json
import os
from eval_rag import answer_question_optimized


def _selected_gemini_model_id() -> str:
    """Gemini LLM 모델 ID는 환경변수로 오버라이드되며, 없으면 기본값을 사용."""
    return os.getenv("GEMINI_MODEL_ID") or "models/gemini-3-flash-preview"


def _print_model_config_once():
    """실행 시점의 모델 설정을 한 번만 출력 (키/토큰 등 민감정보는 출력 금지)."""
    gemini_env = os.getenv("GEMINI_MODEL_ID")
    gemini_selected = _selected_gemini_model_id()
    print(f"[Model] Gemini model id: {gemini_selected} (GEMINI_MODEL_ID={'<unset>' if not gemini_env else gemini_env})")

    # Solar은 eval_rag에서 이미 사용되므로, 실제 인스턴스의 모델 문자열을 그대로 표시
    try:
        from models.solar_client import solar_client
        print(f"[Model] Solar model id: {getattr(solar_client, 'model', '<unknown>')}")
    except Exception as e:
        print(f"[Model] Solar model id: <unavailable> ({str(e)[:80]})")

def main():
    _print_model_config_once()
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