import json
import os
from models.openai_client import openai_client

def get_doc(doc_id):
    with open('/root/IR/data/documents.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if obj['docid'] == doc_id:
                return obj['content']
    return "NOT_FOUND"

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data[str(obj['eval_id'])] = obj
    return data

eval_data = load_jsonl('/root/IR/data/eval.jsonl')
v5 = load_jsonl('/root/IR/submission_bge_m3_sota_v5.csv')
best_9394 = load_jsonl('/root/IR/submission_best_9394.csv')

mismatch_ids = [3, 7, 31, 35, 37, 47, 69, 91, 96, 97, 102, 104, 205, 250, 252, 270, 271, 305]

results = []

for eid in mismatch_ids:
    eid_str = str(eid)
    query = eval_data[eid_str]['msg'][-1]['content']
    
    v5_top1_id = v5[eid_str]['topk'][0] if v5[eid_str]['topk'] else None
    best_top1_id = best_9394[eid_str]['topk'][0] if best_9394[eid_str]['topk'] else None
    
    if not v5_top1_id or not best_top1_id:
        continue
        
    v5_content = get_doc(v5_top1_id)
    best_content = get_doc(best_top1_id)
    
    system_prompt = """당신은 검색 결과의 정확성을 판별하는 전문가입니다.
사용자의 질문과 두 개의 검색 결과(A, B)가 주어집니다.
질문에 대해 더 정확하고 직접적인 정보를 담고 있는 결과를 선택하세요.
반드시 JSON 형식으로 {"winner": "A", "reason": "이유"} 또는 {"winner": "B", "reason": "이유"} 또는 {"winner": "TIE", "reason": "이유"}로 답변하세요."""

    user_prompt = f"""질문: {query}

결과 A (v5):
{v5_content[:1500]}

결과 B (Best 0.9394):
{best_content[:1500]}"""

    try:
        resp = openai_client._call_with_retry(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        decision = json.loads(resp)
        decision['eval_id'] = eid
        decision['v5_id'] = v5_top1_id
        decision['best_id'] = best_top1_id
        results.append(decision)
        print(f"ID {eid}: Winner {decision['winner']}")
    except Exception as e:
        print(f"ID {eid}: Error {e}")

with open('judge_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
