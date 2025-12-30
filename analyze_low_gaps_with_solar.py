import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from models.solar_client import SolarClient

load_dotenv()
solar = SolarClient()

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            data[obj['eval_id']] = obj
    return data

docs_dict = {}
with open('/root/IR/data/documents.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        docs_dict[obj['docid']] = obj['content']

with open('v5_score_gaps.json', 'r') as f:
    gaps = json.load(f)

with open('solar_diff_analysis.json', 'r') as f:
    diffs = json.load(f)
diff_ids = {item['eval_id'] for item in diffs}

# Target queries: Low gap (< 0.05) and NOT in the 18 diffs
targets = [item for item in gaps if item['gap'] < 0.05 and item['eval_id'] not in diff_ids]

print(f"Analyzing {len(targets)} low-gap queries with Solar Pro2...")

results = []

for item in tqdm(targets):
    qid = item['eval_id']
    query = item['query']
    topk = item['topk'] # Top 5 from v5
    
    candidates = [docs_dict[docid] for docid in topk]
    
    system_prompt = """당신은 한국어 정보 검색 전문가입니다. 
질문(Query)과 여러 개의 문서 후보(Candidate)가 주어집니다.
각 문서를 꼼꼼히 읽고, 질문에 대해 가장 정확하고 직접적인 해답을 포함하고 있는 문서를 하나만 선택하세요.
만약 모든 문서가 부적절하다면 가장 근접한 것을 고르세요.

JSON 형식으로 {"best_index": 0} 와 같이 답변하세요. (0은 첫 번째 문서)"""

    candidate_text = ""
    for i, cand in enumerate(candidates):
        candidate_text += f"Candidate {i}:\n{cand}\n\n"
    
    user_prompt = f"Query: {query}\n\n{candidate_text}"
    
    try:
        response = solar._call_with_retry(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], 
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        choice = json.loads(response)['best_index']
        
        if choice != 0:
            print(f"\n[ID {qid}] Solar changed Rank 1: 0 -> {choice}")
            results.append({
                'eval_id': qid,
                'query': query,
                'old_rank1': topk[0],
                'new_rank1': topk[choice],
                'choice_index': choice
            })
    except Exception as e:
        print(f"Error on ID {qid}: {e}")

with open('solar_low_gap_improvements.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nDone. Found {len(results)} potential improvements.")
