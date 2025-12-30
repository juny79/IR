import json
from models.openai_client import openai_client
from tqdm import tqdm

def llm_choose_best(messages, candidates, eid):
    system_prompt = """당신은 검색 결과의 정확도를 판별하는 전문가입니다.
사용자의 질문(대화 맥락 포함)과 2개의 검색 결과(Candidate)가 주어집니다.
질문에 대해 가장 정확하고, 직접적이며, 완결된 답변을 제공하는 문서를 하나만 골라주세요.
반드시 JSON 형식으로 {"best_candidate_index": 0} (0 또는 1) 로 답변하세요.
선택 이유를 아주 간략하게 "reason" 필드에 적어주세요."""

    user_prompt = f"ID: {eid}\n대화 맥락:\n{json.dumps(messages, ensure_ascii=False, indent=2)}\n\n"
    for i, (doc_id, doc_content) in enumerate(candidates):
        user_prompt += f"Candidate {i} (ID: {doc_id}):\n{doc_content[:2000]}\n\n"
    
    try:
        resp = openai_client._call_with_retry(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        parsed = json.loads(resp)
        return int(parsed.get("best_candidate_index", 0)), parsed.get("reason", "")
    except:
        return 0, "error"

with open('data/eval.jsonl', 'r') as f:
    eval_data = {json.loads(l)['eval_id']: json.loads(l)['msg'] for l in f if l.strip()}
with open('data/documents.jsonl', 'r') as f:
    docs = {json.loads(l)['docid']: json.loads(l)['content'] for l in f if l.strip()}

def load_full(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                d[obj['eval_id']] = obj
    return d

v1 = load_full('submission_surgical_v1.csv')
break_v2 = load_full('submission_final_0.95_break_v2.csv')
cand_d = load_full('submission_candidate_D_id271_id303.csv')
master = load_full('submission_final_0.95_master.csv')

# 1. Check [31, 43, 84, 214, 250] (v1 vs break_v2)
print("=== Comparing v1 vs break_v2 ===")
for eid in [31, 43, 84, 214, 250]:
    cands = [
        (v1[eid]['topk'][0], docs[v1[eid]['topk'][0]]),
        (break_v2[eid]['topk'][0], docs[break_v2[eid]['topk'][0]])
    ]
    idx, reason = llm_choose_best(eval_data[eid], cands, eid)
    print(f"ID {eid}: Winner={'v1' if idx==0 else 'break_v2'} | Reason: {reason}")

# 4. Check [26, 37, 85, 106, 215, 243, 246] (v1 vs sota)
print("\n=== Comparing v1 vs sota ===")
sota = load_full('submission_v9_sota.csv')
for eid in [26, 37, 85, 106, 215, 243, 246]:
    cands = [
        (v1[eid]['topk'][0], docs[v1[eid]['topk'][0]]),
        (sota[eid]['topk'][0], docs[sota[eid]['topk'][0]])
    ]
    idx, reason = llm_choose_best(eval_data[eid], cands, eid)
    print(f"ID {eid}: Winner={'v1' if idx==0 else 'sota'} | Reason: {reason}")
