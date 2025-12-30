import json
import os
from tqdm import tqdm
from models.solar_client import SolarClient
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "submission_ab_solar_gap0.05.csv"
OUTPUT_FILE = "submission_solar_final_sota.csv"
DOC_PATH = "data/documents.jsonl"

EMPTY_IDS = {276, 261, 283, 32, 94, 90, 220, 245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218}

def generate_answer(solar, query, context):
    system_prompt = """당신은 친절한 AI 어시스턴트입니다. 주어진 문맥(Context)을 바탕으로 사용자의 질문에 답하세요.
문맥에 없는 내용은 답하지 마세요. 한국어로 답변하세요."""
    user_prompt = f"질문: {query}\n\n문맥:\n{context}"
    
    try:
        answer = solar._call_with_retry(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=1024
        )
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "답변을 생성할 수 없습니다."

def main():
    solar = SolarClient(model_name="solar-pro")
    
    print("Loading documents...")
    docs = {}
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs[obj["docid"]] = obj["content"]
            
    print(f"Processing {INPUT_FILE}...")
    
    # Load eval data for fallback queries
    eval_queries = {}
    with open("data/eval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            eval_queries[obj["eval_id"]] = obj["msg"][-1]["content"]

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for line in tqdm(lines):
            item = json.loads(line)
            eval_id = item["eval_id"]
            query = item.get("standalone_query") or eval_queries.get(eval_id, "")
            topk = item["topk"]
            
            if eval_id in EMPTY_IDS:
                item["answer"] = "검색이 필요하지 않은 질문입니다."
            else:
                # Use top 3 docs for context
                context = "\n".join([docs.get(docid, "") for docid in topk[:3]])
                item["answer"] = generate_answer(solar, query, context)
            
            # Ensure standalone_query is present
            item["standalone_query"] = query
            
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"✅ Final submission file created: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
