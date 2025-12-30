import json
import os
import re
from models.gemini_client import gemini_client
from models.solar_client import solar_client
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

def get_llm_choice(model_type, query, candidates_info):
    prompt_template = """당신은 과학 지식 검색 시스템의 최종 평가관입니다.
사용자의 질문에 대해 가장 정확하고 직접적인 정답을 포함하고 있는 문서를 하나만 선택해야 합니다.

질문: {query}

[후보 문서 리스트]
{candidates_text}

위 질문에 대해 정답을 포함하고 있거나 가장 관련성이 높은 문서의 번호를 선택하세요.
반드시 선택한 문서의 번호(예: 1)만 출력하세요. 다른 설명은 절대 하지 마세요."""
    
    candidates_text = "\n\n".join([f"{i+1}. [ID: {c['id']}]\n내용: {c['content'][:1000]}..." for i, c in enumerate(candidates_info)])
    prompt = prompt_template.format(query=query, candidates_text=candidates_text)
    
    try:
        if model_type == "gemini":
            response = gemini_client._call_with_retry(prompt).strip()
        else:
            response = solar_client._call_with_retry(prompt).strip()
            
        match = re.search(r'\d+', response)
        if match:
            return int(match.group(0))
    except:
        pass
    return None

def consensus_rerank():
    # 0.9470 달성한 최신 파일 로드
    base_sub = load_submission('/root/IR/submission_surgical_v1.csv')
    v9 = load_submission('/root/IR/submission_v9_sota.csv')
    v3 = load_submission('/root/IR/submission_v3_final.csv')
    
    # 타겟 ID (v9과 v3가 달랐던 21개 핵심 질문)
    target_ids = ["26", "31", "37", "43", "44", "51", "65", "66", "74", "84", "85", "97", "98", "106", "107", "214", "215", "243", "246", "250", "252"]
    
    print(f"Starting Consensus Reranking for {len(target_ids)} hard questions...")
    
    final_results = base_sub.copy()
    
    for eid in target_ids:
        query = base_sub[eid].get('standalone_query') or v9[eid].get('standalone_query')
        
        # 후보군 확장: v9 Top-5 + v3 Top-5 (최대 10개)
        tk9 = v9[eid].get('topk', [])[:5]
        tk3 = v3[eid].get('topk', [])[:5]
        candidate_ids = list(dict.fromkeys(tk9 + tk3))
        
        candidates_info = []
        for cid in candidate_ids:
            candidates_info.append({'id': cid, 'content': get_doc_content(cid)})
            
        # Dual-LLM Judging
        gemini_choice = get_llm_choice("gemini", query, candidates_info)
        solar_choice = get_llm_choice("solar", query, candidates_info)
        
        final_choice_idx = None
        if gemini_choice and solar_choice and gemini_choice == solar_choice:
            # 두 모델이 합의한 경우
            final_choice_idx = gemini_choice - 1
            print(f"ID {eid}: Consensus reached on #{gemini_choice}")
        elif gemini_choice:
            # 합의 실패 시 Gemini 우선 (또는 문맥에 따라 판단)
            final_choice_idx = gemini_choice - 1
            print(f"ID {eid}: Consensus failed. Gemini: {gemini_choice}, Solar: {solar_choice}. Using Gemini.")
        elif solar_choice:
            final_choice_idx = solar_choice - 1
            print(f"ID {eid}: Gemini failed. Using Solar: {solar_choice}")
            
        if final_choice_idx is not None and 0 <= final_choice_idx < len(candidate_ids):
            selected_id = candidate_ids[final_choice_idx]
            
            # Top-1 교체 및 순서 재정렬
            new_topk = [selected_id]
            for cid in candidate_ids:
                if cid != selected_id:
                    new_topk.append(cid)
            
            final_results[eid]['topk'] = new_topk[:5]
        else:
            print(f"ID {eid}: Both LLMs failed or invalid choice. Keeping base.")

    # 최종 파일 저장
    output_path = '/root/IR/submission_final_0.95_break.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        for eid in sorted(final_results.keys(), key=lambda x: int(x)):
            f.write(json.dumps(final_results[eid], ensure_ascii=False) + '\n')
            
    print(f"\nFinal 0.95 Break submission saved to {output_path}")

if __name__ == "__main__":
    consensus_rerank()
