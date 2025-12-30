"""
Phase 3-1: Solar-pro2 HyDE 소규모 평가
Solar-pro2로 HyDE를 생성하여 MAP 개선 효과 검증
"""

import json
from eval_rag import answer_question_optimized

# 평가 데이터 로드
with open('data/eval.jsonl', 'r', encoding='utf-8') as f:
    eval_data = [json.loads(line) for line in f]

# 소규모 평가: 처음 30개 질문만
sample_size = 30
eval_sample = eval_data[:sample_size]

print(f"=== Phase 3-1: Solar-pro2 HyDE 평가 시작 ===")
print(f"평가 샘플: {sample_size}개 질문\n")

results = []

for i, item in enumerate(eval_sample):
    messages = item['msg']
    eval_id = item['eval_id']
    
    try:
        result = answer_question_optimized(messages)
        
        results.append({
            'eval_id': eval_id,
            'topk': result['topk']
        })
        
        if (i + 1) % 5 == 0:
            print(f"진행: {i+1}/{sample_size} 완료")
    
    except Exception as e:
        print(f"❌ 오류 발생 (eval_id {eval_id}): {e}")
        results.append({
            'eval_id': eval_id,
            'topk': []
        })

print(f"\n{'='*60}")
print(f"Phase 3-1 완료")
print(f"{'='*60}")
print(f"처리 질문: {len(results)}개")
print(f"결과 저장: phase_3_1_results.json")

# 결과 저장
with open('phase_3_1_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n📊 다음 단계:")
print(f"1. 전체 220개 평가 실행: python3 main.py")
print(f"2. MAP 계산 및 리더보드 제출")
