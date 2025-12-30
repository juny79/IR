import json
import os
from models.solar_client import solar_client

def judge_diffs(report_path, output_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    final_decisions = {}
    
    prompt_template = """질문: {query}

문서 A: {doc_a}
문서 B: {doc_b}

위 질문에 대해 더 정확하고 직접적인 답변을 포함하고 있는 문서를 선택하세요.
답변은 반드시 'A' 또는 'B' 한 글자만 출력하세요."""

    for item in report:
        eid = item['eval_id']
        query = item['query']
        doc_a = item['v9']['content']
        doc_b = item['v3']['content']
        
        prompt = prompt_template.format(query=query, doc_a=doc_a, doc_b=doc_b)
        
        try:
            # Solar Pro 2를 판사로 사용 (Gemini도 가능하지만 Solar가 빠름)
            decision = solar_client._call_with_retry(prompt).strip()
            # 'A' 또는 'B'만 추출
            if 'A' in decision and 'B' not in decision:
                choice = 'v9'
            elif 'B' in decision and 'A' not in decision:
                choice = 'v3'
            else:
                # 모호하면 v9 선택 (안전)
                choice = 'v9'
        except:
            choice = 'v9'
            
        final_decisions[eid] = choice
        print(f"ID {eid}: Selected {choice}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_decisions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    judge_diffs('/root/IR/judge_report.json', '/root/IR/judge_decisions.json')
