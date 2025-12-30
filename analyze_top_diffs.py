import json
import os
from models.solar_client import SolarClient
from dotenv import load_dotenv

load_dotenv()

solar_client = SolarClient(model_name="solar-pro")

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if 'eval_id' in obj:
                    data[obj['eval_id']] = obj
                elif 'docid' in obj:
                    data[obj['docid']] = obj
    return data

def analyze():
    with open('submission_diffs.json', 'r', encoding='utf-8') as f:
        diffs = json.load(f)
    
    queries = load_jsonl('data/eval.jsonl')
    docs = load_jsonl('data/documents.jsonl')
    
    results = []
    
    system_prompt = """당신은 한국어 정보 검색 전문가입니다. 
 질문(Query)과 2개의 문서 후보(Candidate)가 주어집니다.
 대해 가장 정확하고, 직접적인 해답을 포함하고 있는 문서를 하나만 선택하세요.

 JSON 형식으로 {"best_index": 0} 와 같이 답변하세요. (0: Candidate 0, 1: Candidate 1)"""

    for d in diffs:
        eval_id = d['eval_id']
        query_obj = queries.get(eval_id)
        if not query_obj:
            continue
        
        # Fix: query is in msg[0]['content']
        query = query_obj['msg'][0]['content']
        doc0 = docs.get(d['v5_docid'], {}).get('content', 'N/A')
        doc1 = docs.get(d['best_docid'], {}).get('content', 'N/A')
        
        user_prompt = f"## 질문:\n{query}\n\n## 검색 후보:\nCandidate 0 (v5):\n{doc0}\n\nCandidate 1 (best):\n{doc1}"
        
        try:
            resp = solar_client._call_with_retry(
                prompt=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(resp)
            best_idx = parsed.get('best_index')
            results.append({
                'eval_id': eval_id,
                'query': query,
                'v5_docid': d['v5_docid'],
                'best_docid': d['best_docid'],
                'solar_choice': 'v5' if best_idx == 0 else 'best',
                'solar_raw': resp
            })
            print(f"ID {eval_id}: Solar chose {'v5' if best_idx == 0 else 'best'}")
        except Exception as e:
            print(f"Error on ID {eval_id}: {e}")
            
    with open('solar_diff_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    analyze()
