import json
import os
from models.gemini_client import gemini_client

def load_docs():
    docs = {}
    with open('/root/IR/data/documents.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            docs[obj['docid']] = obj['content']
    return docs

def get_topk(file, qid):
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if obj['eval_id'] == qid: return obj['topk']
    return None

def gemini_rerank_topk(query, candidates):
    system_prompt = """당신은 한국어 과학 지식 검색 전문가입니다. 
사용자의 질문과 검색된 문서 후보(Candidate)들이 주어집니다.
질문에 대해 가장 정확하고, 직접적인 해답을 포함하고 있는 문서를 하나만 선택하세요.

선택 기준:
1. 질문의 핵심 의도에 완벽히 부합하는가?
2. 과학적 사실이 정확한가?
3. 질문에서 요구하는 구체적인 정보를 담고 있는가?

반드시 JSON 형식으로 {"best_index": 0} 와 같이 답변하세요."""

    candidate_text = ""
    for i, content in enumerate(candidates):
        candidate_text += f"Candidate {i}:\n{content[:2000]}\n\n"
        
    user_prompt = f"## 질문:\n{query}\n\n## 검색 후보:\n{candidate_text}"
    
    resp = gemini_client._call_with_retry(
        prompt=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=100,
        response_format={"type": "json_object"}
    )
    try:
        parsed = json.loads(resp)
        return int(parsed.get("best_index", 0))
    except:
        return 0

docs = load_docs()
test_ids = [37, 8, 106, 263]

for qid in test_ids:
    print(f"\n--- Testing ID {qid} ---")
    # Get top 10 from v9
    v9_topk = get_topk('submission_v9_sota.csv', qid)
    if not v9_topk:
        print(f"ID {qid} not found in v9")
        continue
        
    query = ""
    with open('/root/IR/data/eval.jsonl', 'r') as f:
        for line in f:
            obj = json.loads(line)
            if obj['eval_id'] == qid:
                query = obj['msg'][-1]['content']
                break
    
    candidates = [docs[docid] for docid in v9_topk[:10]]
    best_idx = gemini_rerank_topk(query, candidates)
    
    print(f"Query: {query}")
    print(f"v9 Rank 1: {v9_topk[0]}")
    print(f"Gemini Choice Index: {best_idx}")
    print(f"Gemini Choice DocID: {v9_topk[best_idx]}")
    print(f"Gemini Choice Content: {candidates[best_idx][:200]}...")
