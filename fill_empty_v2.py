import json
from retrieval.es_connector import es
from models.gemini_client import gemini_client
import re

def search(q, limit=50):
    res = es.search(index='test', query={'multi_match': {'query': q, 'fields': ['content']}}, size=limit)
    return [{'docid': hit['_source']['docid'], 'content': hit['_source']['content']} for hit in res['hits']['hits']]

def get_best(query, candidates):
    prompt = f"사용자 질문: {query}\n\n후보 문서들:\n"
    for i, c in enumerate(candidates):
        prompt += f"{i+1}. [ID: {c['docid']}] {c['content'][:600]}\n\n"
    prompt += "\n위 질문에 대해 (비록 질문이 일상 대화일지라도) 가장 관련이 있거나 키워드가 겹치는 문서를 하나만 선택하세요. 번호만 답하세요."
    try:
        res = gemini_client._call_with_retry(prompt).strip()
        match = re.search(r'\d+', res)
        if match:
            return int(match.group(0))
    except:
        return None
    return None

def main():
    # Queries retrieved from eval.jsonl
    queries = {
        "2": "이제 그만 얘기하자.",
        "32": "오늘 너무 즐거웠다!",
        "57": "우울한데 신나는 얘기 좀 해줄래?",
        "64": "너 모르는 것도 있어?",
        "67": "니가 대답을 잘해줘서 기분이 좋아!",
        "83": "너 정말 똑똑하구나?",
        "90": "안녕 반갑다",
        "94": "우울한데 신나는 얘기 좀 해줘!",
        "103": "너 뭘 잘해?",
        "218": "요새 너무 힘드네..",
        "220": "너는 누구야?",
        "222": "안녕 반가워",
        "227": "너는 누구니?",
        "229": "너 잘하는게 뭐야?",
        "245": "너 모르는 것도 있니?",
        "247": "너 정말 똑똑하다!",
        "261": "니가 대답을 잘해줘서 너무 신나!",
        "276": "요새 너무 힘들다.",
        "283": "이제 그만 얘기해!",
        "301": "오늘 너무 즐거웠어!"
    }

    results = {}
    for eid, query in queries.items():
        print(f"Processing Conversational ID {eid}: {query}")
        candidates = search(query, 50)
        best_idx = get_best(query, candidates)
        if best_idx and 0 < best_idx <= len(candidates):
            results[eid] = candidates[best_idx-1]['docid']
            print(f"  -> Selected Rank {best_idx}: {results[eid]}")
        else:
            # Fallback to Rank 1 if Gemini fails
            if candidates:
                results[eid] = candidates[0]['docid']
                print(f"  -> Fallback to Rank 1: {results[eid]}")

    print("\n--- CONVERSATIONAL IDS RECOMMENDATIONS ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
