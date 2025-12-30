import json
import os
from models.gemini_client import gemini_client
from retrieval.es_connector import es

def get_doc_content(docid):
    try:
        res = es.search(index="test", query={"term": {"docid": docid}}, _source=["content"])
        if res['hits']['hits']:
            return res['hits']['hits'][0]['_source']['content']
    except:
        pass
    return "Content not found."

def load_submission(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            data[str(obj['eval_id'])] = obj
    return data

def surgical_strike():
    v9 = load_submission('/root/IR/submission_v9_sota.csv')
    v3 = load_submission('/root/IR/submission_v3_final.csv')
    
    # Top-1이 다른 ID 추출
    diff_ids = []
    for eid in v9:
        if eid in v3:
            tk9 = v9[eid].get('topk', [])
            tk3 = v3[eid].get('topk', [])
            if (tk9 and tk3 and tk9[0] != tk3[0]) or (not tk9 and tk3) or (tk9 and not tk3):
                diff_ids.append(eid)
    
    print(f"Found {len(diff_ids)} differences to judge.")
    
    final_results = v9.copy() # v9을 베이스로 시작
    
    prompt_template = """당신은 과학 지식 검색 시스템의 정밀 평가관입니다.
사용자의 질문에 대해 가장 정확하고 직접적인 정보를 담고 있는 문서를 하나만 선택해야 합니다.

질문: {query}

[후보 문서 리스트]
{candidates_text}

위 질문에 대해 정답을 포함하고 있거나 가장 관련성이 높은 문서의 번호를 선택하세요.
반드시 선택한 문서의 번호(예: 1)만 출력하세요. 다른 설명은 절대 하지 마세요."""

    for eid in diff_ids:
        query = v9[eid].get('standalone_query') or v3[eid].get('standalone_query')
        tk9 = v9[eid].get('topk', [])[:5]
        tk3 = v3[eid].get('topk', [])[:5]
        
        # 후보군 합치기 (중복 제거)
        candidate_ids = list(dict.fromkeys(tk9 + tk3))
        if not candidate_ids:
            continue
            
        candidates_info = []
        for idx, cid in enumerate(candidate_ids, 1):
            content = get_doc_content(cid)
            candidates_info.append(f"{idx}. [ID: {cid}]\n내용: {content[:1000]}...")
        
        candidates_text = "\n\n".join(candidates_info)
        prompt = prompt_template.format(query=query, candidates_text=candidates_text)
        
        try:
            # Gemini-3-Flash를 판사로 사용
            response = gemini_client._call_with_retry(prompt).strip()
            
            # 숫자 추출
            import re
            match = re.search(r'\d+', response)
            if match:
                idx = int(match.group(0)) - 1
                if 0 <= idx < len(candidate_ids):
                    selected_id = candidate_ids[idx]
                    
                    # 새로운 Top-1으로 설정하고 나머지는 순서대로 유지
                    new_topk = [selected_id]
                    for cid in candidate_ids:
                        if cid != selected_id:
                            new_topk.append(cid)
                    
                    final_results[eid]['topk'] = new_topk[:5]
                    source = "v9" if selected_id == (tk9[0] if tk9 else "") else "v3"
                    print(f"ID {eid}: Selected #{idx+1} ({selected_id}) - Source: {source}")
                else:
                    print(f"ID {eid}: Gemini selected out of range index {idx+1}. Keeping v9.")
            else:
                print(f"ID {eid}: Gemini response had no number: {response}. Keeping v9.")
        except Exception as e:
            print(f"ID {eid}: Error during judging: {e}")

    # 결과 저장
    output_path = '/root/IR/submission_surgical_v1.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        for eid in sorted(final_results.keys(), key=lambda x: int(x)):
            f.write(json.dumps(final_results[eid], ensure_ascii=False) + '\n')
    
    print(f"Surgical strike complete. Saved to {output_path}")

if __name__ == "__main__":
    surgical_strike()
