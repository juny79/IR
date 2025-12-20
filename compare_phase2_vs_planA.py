#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

# Phase 2 (HyDE 전체) vs 방안 A (HyDE Sparse Only) 비교
with open('./submission_hyde_v1.csv', 'r', encoding='utf-8') as f:
    phase2_results = [json.loads(line) for line in f]

with open('./submission.csv', 'r', encoding='utf-8') as f:
    plan_a_results = [json.loads(line) for line in f]

print('=== Phase 2 (HyDE 전체) vs 방안 A (HyDE Sparse Only) 비교 ===\n')

# Top-K 문서 변화 분석
same_count = 0
partial_match = 0
total_different = 0
overlap_counts = []

for i in range(len(phase2_results)):
    p2_topk = set(phase2_results[i]['topk'][:5])
    pa_topk = set(plan_a_results[i]['topk'][:5])
    
    overlap = len(p2_topk & pa_topk)
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
top1_same = sum(1 for i in range(220) if len(phase2_results[i]['topk']) > 0 and len(plan_a_results[i]['topk']) > 0 and phase2_results[i]['topk'][0] == plan_a_results[i]['topk'][0])
print(f'2. Top-1 문서 변화:')
print(f'   동일: {top1_same}개 ({top1_same/220*100:.1f}%)')
print(f'   변경: {220-top1_same}개 ({(220-top1_same)/220*100:.1f}%)\n')

# 답변 길이 비교
p2_lens = [len(r['answer']) for r in phase2_results]
pa_lens = [len(r['answer']) for r in plan_a_results]

print(f'3. 답변 길이:')
print(f'   Phase 2 평균: {sum(p2_lens)/len(p2_lens):.0f}자')
print(f'   방안 A 평균: {sum(pa_lens)/len(pa_lens):.0f}자\n')

# 변화가 큰 케이스 분석
print(f'4. Top-K 완전 변경 케이스 샘플 (처음 5개):')
count = 0
for i in range(len(phase2_results)):
    if count >= 5:
        break
    
    p2_topk = phase2_results[i]['topk'][:5]
    pa_topk = plan_a_results[i]['topk'][:5]
    
    overlap = len(set(p2_topk) & set(pa_topk))
    
    if overlap == 0:
        count += 1
        print(f'\n   [{count}] ID {phase2_results[i]["eval_id"]}: {phase2_results[i]["standalone_query"][:40]}...')
        print(f'       Phase 2 Top-3: {p2_topk[:3]}')
        print(f'       방안 A Top-3: {pa_topk[:3]}')
