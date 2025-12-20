#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

# 전체 전략 비교 분석
strategies = {
    'Phase 1 (Reranker Only)': './submission_reranker.csv',
    'Phase 2 (HyDE Full)': './submission_hyde_v1.csv',
    '방안 A (HyDE Sparse Only)': './submission_planA.csv',
    '전략 A (HyDE Sparse+Reranker)': './submission.csv'
}

scores = {
    'Phase 1 (Reranker Only)': {'MAP': 0.7742, 'MRR': 0.7773},
    'Phase 2 (HyDE Full)': {'MAP': 0.7970, 'MRR': 0.8015},
    '방안 A (HyDE Sparse Only)': {'MAP': 0.7962, 'MRR': 0.7985},
    '전략 A (HyDE Sparse+Reranker)': {'MAP': 0.7780, 'MRR': 0.7818}
}

# 각 전략의 결과 로드
results = {}
for name, path in strategies.items():
    try:
        with open(path, 'r', encoding='utf-8') as f:
            results[name] = [json.loads(line) for line in f]
    except:
        print(f'{name} 파일 없음: {path}')

print('=' * 80)
print('전체 전략 성능 비교')
print('=' * 80)
print(f'\n{"전략":<30} {"MAP":<10} {"MRR":<10} {"Phase 1 대비":<15}')
print('-' * 80)

baseline_map = scores['Phase 1 (Reranker Only)']['MAP']
for name in ['Phase 1 (Reranker Only)', 'Phase 2 (HyDE Full)', '방안 A (HyDE Sparse Only)', '전략 A (HyDE Sparse+Reranker)']:
    score = scores[name]
    diff = score['MAP'] - baseline_map
    sign = '+' if diff > 0 else ''
    print(f'{name:<30} {score["MAP"]:<10.4f} {score["MRR"]:<10.4f} {sign}{diff:.4f} ({sign}{diff/baseline_map*100:.1f}%)')

print('\n' + '=' * 80)
print('전략별 Top-1 문서 비교 (전략 A vs 다른 전략)')
print('=' * 80)

if '전략 A (HyDE Sparse+Reranker)' in results:
    strategy_a = results['전략 A (HyDE Sparse+Reranker)']
    
    for name in ['Phase 1 (Reranker Only)', 'Phase 2 (HyDE Full)', '방안 A (HyDE Sparse Only)']:
        if name not in results:
            continue
            
        other = results[name]
        
        # Top-1 비교
        top1_same = sum(1 for i in range(220) 
                       if len(strategy_a[i]['topk']) > 0 and len(other[i]['topk']) > 0 
                       and strategy_a[i]['topk'][0] == other[i]['topk'][0])
        
        # Top-5 중복률
        overlap_sum = 0
        for i in range(220):
            sa_topk = set(strategy_a[i]['topk'][:5])
            o_topk = set(other[i]['topk'][:5])
            overlap_sum += len(sa_topk & o_topk)
        
        avg_overlap = overlap_sum / 220
        
        print(f'\n{name}:')
        print(f'  Top-1 동일: {top1_same}/220 ({top1_same/220*100:.1f}%)')
        print(f'  Top-5 평균 중복: {avg_overlap:.2f}/5개 ({avg_overlap/5*100:.1f}%)')

print('\n' + '=' * 80)
print('핵심 분석')
print('=' * 80)
print('''
1. 전략 A (Reranker HyDE 적용)의 참혹한 실패:
   - MAP 0.7780: 방안 A 대비 -0.0182 (2.3% 하락)
   - MAP 0.7780: Phase 2 대비 -0.0190 (2.4% 하락)
   - Phase 1과 거의 비슷한 수준 (+0.0038, 0.5%)
   
2. 성능 순위 (높은 순):
   1위: Phase 2 (HyDE Full) - MAP 0.7970 ⭐
   2위: 방안 A (HyDE Sparse Only) - MAP 0.7962
   3위: 전략 A (HyDE Sparse+Reranker) - MAP 0.7780 ❌
   4위: Phase 1 (Reranker Only) - MAP 0.7742
   
3. 결론:
   - Reranker에 HyDE 적용은 명백한 역효과
   - HyDE는 Dense에 적용하거나 전체 적용이 더 나음
   - Sparse Only도 전체 적용보다 약간 낮음
''')
