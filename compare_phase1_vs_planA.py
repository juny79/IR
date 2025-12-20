#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

# Phase 1 (Reranker) vs 방안 A (HyDE Sparse Only) 비교
with open('./submission_reranker.csv', 'r', encoding='utf-8') as f:
    phase1_results = [json.loads(line) for line in f]

with open('./submission.csv', 'r', encoding='utf-8') as f:
    plan_a_results = [json.loads(line) for line in f]

print('=== Phase 1 (Reranker Only) vs 방안 A (HyDE Sparse Only) 비교 ===\n')
print('성능 점수:')
print('  Phase 1: MAP 0.7742, MRR 0.7773')
print('  방안 A:  MAP 0.7962, MRR 0.7985')
print('  차이:    MAP +0.0220, MRR +0.0212 (✅ 개선)\n')

# Top-K 문서 변화 분석
same_count = 0
partial_match = 0
total_different = 0
overlap_counts = []

for i in range(len(phase1_results)):
    p1_topk = set(phase1_results[i]['topk'][:5])
    pa_topk = set(plan_a_results[i]['topk'][:5])
    
    overlap = len(p1_topk & pa_topk)
    overlap_counts.append(overlap)
    
    if overlap == 5:
        same_count += 1
    elif overlap > 0:
        partial_match += 1
    else:
        total_different += 1

print(f'1. Top-5 문서 변화:')
print(f'   동일 (5/5): {same_count}개 ({same_count/220*100:.1f}%)')
print(f'   부분 변화 (1~4/5): {partial_match}개 ({partial_match/220*100:.1f}%)')
print(f'   완전 변화 (0/5): {total_different}개 ({total_different/220*100:.1f}%)')
print(f'   평균 중복: {sum(overlap_counts)/len(overlap_counts):.2f}/5개\n')

# 상위 랭킹 변화 분석 (Top-1만 비교)
top1_same = sum(1 for i in range(220) if len(phase1_results[i]['topk']) > 0 and len(plan_a_results[i]['topk']) > 0 and phase1_results[i]['topk'][0] == plan_a_results[i]['topk'][0])
print(f'2. Top-1 문서 변화:')
print(f'   동일: {top1_same}개 ({top1_same/220*100:.1f}%)')
print(f'   변경: {220-top1_same}개 ({(220-top1_same)/220*100:.1f}%)\n')

# 개선/악화 분석 (Top-1 기준)
print('3. 주요 분석 포인트:')
print(f'   - Phase 1 → 방안 A 전환 시 Top-1이 {220-top1_same}개 ({(220-top1_same)/220*100:.1f}%) 변경')
print(f'   - MAP는 +2.2% 개선 (0.7742 → 0.7962)')
print(f'   - 이는 HyDE가 Sparse Search에 긍정적 영향을 주었음을 의미\n')
