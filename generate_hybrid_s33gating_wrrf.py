"""
🎯 하이브리드 전략: submission_33의 게이팅 + weighted_rrf의 검색

핵심:
1. submission_33의 21개 empty 케이스는 신뢰할 수 있음
   - Solar LLM이 is_science=false로 판정한 질문들
   - Ground truth topk=[]와 일치할 확률 높음

2. weighted_rrf의 가중치 RRF + Multi-Query는 좋은 전략
   - 검색 품질 향상 기대

3. 결합: submission_33의 게이팅 + weighted_rrf의 검색 로직
"""

import json
from pathlib import Path

def load_submission(filepath):
    """JSONL 형식 submission 로드"""
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                results[data['eval_id']] = data
    return results

# 파일 로드
s33 = load_submission('/root/IR/submission_33_ready_4_tk80_cp80_h200_w542.csv')
wrrf = load_submission('/root/IR/submission_weighted_rrf.csv')

print("📊 하이브리드 생성 중...")

# 결합 로직
hybrid_results = []
empty_from_33 = 0
search_from_wrrf = 0

for eval_id, data_wrrf in wrrf.items():
    data_33 = s33.get(eval_id)
    
    # submission_33에서 empty인 경우: 그 상태 유지 (신뢰도 높음)
    if data_33 and not data_33.get('topk', []):
        hybrid_results.append(data_33)
        empty_from_33 += 1
    else:
        # submission_33에서 non-empty인 경우: wrrf의 검색 결과 사용 (품질 향상)
        hybrid_results.append(data_wrrf)
        if data_wrrf.get('topk', []):
            search_from_wrrf += 1

# 정렬 (eval_id 순)
hybrid_results.sort(key=lambda x: x['eval_id'])

# 저장
output_path = '/root/IR/submission_hybrid_s33gating_wrrf_search.csv'
with open(output_path, 'w', encoding='utf-8') as f:
    for r in hybrid_results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"✅ 파일 생성: {output_path}")
print(f"\n📈 구성:")
print(f"   - submission_33의 empty 케이스: {empty_from_33}개 (그대로 유지)")
print(f"   - weighted_rrf의 검색 결과: {search_from_wrrf}개 (품질 향상)")
print(f"\n💡 기대 효과:")
print(f"   - submission_33 (0.8886)의 신뢰할 수 있는 게이팅 유지")
print(f"   - weighted_rrf의 더 나은 검색 품질 적용")
print(f"   - 예상 점수: 0.8886 ~ 0.8950")

# 통계
empty_hybrid = sum(1 for r in hybrid_results if not r.get('topk', []))
print(f"\n📊 최종 통계:")
print(f"   - Empty topk: {empty_hybrid}개 (s33 정책 유지)")
print(f"   - Non-empty topk: {220-empty_hybrid}개")
