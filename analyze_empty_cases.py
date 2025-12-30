"""
submission_33의 Empty Case 21개를 분석하여
검색이 필요한 질문인지(False Negative) 판단하기 위한 스크립트
"""
import json

def load_data(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data[item['eval_id']] = item
    return data

# 데이터 로드
s33 = load_data('/root/IR/submission_33_ready_4_tk80_cp80_h200_w542.csv')
wrrf = load_data('/root/IR/submission_weighted_rrf.csv')
eval_data = load_data('/root/IR/data/eval.jsonl')

print("🔍 submission_33의 Empty Case (21개) 분석")
print("=" * 100)
print(f"{'ID':<5} | {'질문 내용 (원본)':<60} | {'wrrf 검색 결과 (Top1)':<40}")
print("-" * 100)

empty_ids = [eid for eid, item in s33.items() if not item.get('topk')]

for eid in empty_ids:
    # 원본 질문
    original_msg = eval_data[eid]['msg']
    last_content = original_msg[-1]['content'] if isinstance(original_msg, list) else str(original_msg)
    
    # wrrf 결과
    wrrf_item = wrrf.get(eid)
    wrrf_top1 = "없음"
    if wrrf_item and wrrf_item.get('topk'):
        # 답변의 첫 문장이나 top1 문서 내용을 가져오면 좋겠지만, 여기선 답변 앞부분만
        wrrf_answer = wrrf_item.get('answer', '')
        wrrf_top1 = wrrf_answer[:40].replace('\n', ' ') + "..."
    
    print(f"{eid:<5} | {last_content[:60]:<60} | {wrrf_top1:<40}")

print("=" * 100)
