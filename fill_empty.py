import json
from retrieval.es_connector import es
from models.gemini_client import gemini_client
import re

def search(q, limit=50):
    res = es.search(index='test', query={'multi_match': {'query': q, 'fields': ['content']}}, size=limit)
    return [{'docid': hit['_source']['docid'], 'content': hit['_source']['content']} for hit in res['hits']['hits']]

def get_best(query, candidates):
    prompt = f"질문: {query}\n\n후보 문서들:\n"
    for i, c in enumerate(candidates):
        prompt += f"{i+1}. [ID: {c['docid']}] {c['content'][:600]}\n\n"
    prompt += "\n위 질문에 대해 가장 정확하고 완벽한 정답을 제공하는 문서의 번호만 답하세요. 만약 정답이 없으면 0을 답하세요."
    try:
        res = gemini_client._call_with_retry(prompt).strip()
        match = re.search(r'\d+', res)
        if match:
            return int(match.group(0))
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return None
    return None

def main():
    empty_ids = ['2', '32', '57', '64', '67', '83', '90', '94', '103', '218', '220', '222', '227', '229', '245', '247', '261', '276', '283', '301']
    
    with open('submission_v9_sota.csv', 'r') as f:
        v9_data = {str(obj['eval_id']): obj for line in f for obj in [json.loads(line)]}

    results = {}
    for eid in empty_ids:
        if eid not in v9_data: continue
        query = v9_data[eid]['standalone_query']
        print(f"Processing Empty ID {eid}: {query}")
        candidates = search(query, 50)
        best_idx = get_best(query, candidates)
        if best_idx and 0 < best_idx <= len(candidates):
            results[eid] = candidates[best_idx-1]['docid']
            print(f"  -> Selected Rank {best_idx}: {results[eid]}")
        else:
            print(f"  -> No change or failed (idx: {best_idx})")

    print("\n--- EMPTY IDS RECOMMENDATIONS ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
