#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

# Phase 1 (원본 쿼리 Reranker) vs 전략 A (HyDE 쿼리 Reranker) 비교
with open('./submission_reranker.csv', 'r', encoding='utf-8') as f:
    phase1 = [json.loads(line) for line in f]

with open('./submission.csv', 'r', encoding='utf-8') as f:
    strategy_a = [json.loads(line) for line in f]

print('=' * 80)
print('Reranker HyDE 적용 역효과 분석')
print('=' * 80)
print(f'\nPhase 1 (원본 쿼리): MAP 0.7742')
print(f'전략 A (HyDE 쿼리): MAP 0.7780 (+0.0038, +0.5%)')
print(f'\n→ Reranker에 HyDE 적용이 거의 효과 없음\n')

# Top-1 변화 분석
top1_better = 0  # 전략 A가 더 나은 경우 (추정)
top1_worse = 0   # 전략 A가 더 나쁜 경우 (추정)
top1_same = 0

examples_worse = []
examples_better = []

for i in range(220):
    p1_top1 = phase1[i]['topk'][0] if len(phase1[i]['topk']) > 0 else None
    sa_top1 = strategy_a[i]['topk'][0] if len(strategy_a[i]['topk']) > 0 else None
    
    if p1_top1 == sa_top1:
        top1_same += 1
    else:
        # Phase 1의 Top-1이 전략 A에서 순위 하락
        if p1_top1 in strategy_a[i]['topk']:
            p1_rank_in_sa = strategy_a[i]['topk'].index(p1_top1) + 1
        else:
            p1_rank_in_sa = 99
        
        # 전략 A의 Top-1이 Phase 1에서 순위
        if sa_top1 in phase1[i]['topk']:
            sa_rank_in_p1 = phase1[i]['topk'].index(sa_top1) + 1
        else:
            sa_rank_in_p1 = 99
        
        # Phase 1의 원래 Top-1이 하락하고, 전략 A의 새 Top-1이 Phase 1에서 낮았다면 악화
        if p1_rank_in_sa > 1 and sa_rank_in_p1 > 1:
            top1_worse += 1
            if len(examples_worse) < 5:
                examples_worse.append({
                    'eval_id': phase1[i]['eval_id'],
                    'query': phase1[i]['standalone_query'],
                    'p1_top1': p1_top1,
                    'sa_top1': sa_top1,
                    'p1_rank_in_sa': p1_rank_in_sa,
                    'sa_rank_in_p1': sa_rank_in_p1
                })
        elif sa_rank_in_p1 > p1_rank_in_sa:
            top1_better += 1
            if len(examples_better) < 5:
                examples_better.append({
                    'eval_id': phase1[i]['eval_id'],
                    'query': phase1[i]['standalone_query'],
                    'p1_top1': p1_top1,
                    'sa_top1': sa_top1,
                    'p1_rank_in_sa': p1_rank_in_sa,
                    'sa_rank_in_p1': sa_rank_in_p1
                })

print('Top-1 변화 패턴:')
print(f'  동일: {top1_same}/220 ({top1_same/220*100:.1f}%)')
print(f'  변경: {220-top1_same}/220 ({(220-top1_same)/220*100:.1f}%)')
print(f'    - 추정 개선: {top1_better}개')
print(f'    - 추정 악화: {top1_worse}개')

print('\n' + '=' * 80)
print('악화 사례 샘플 (Phase 1이 더 나았던 경우)')
print('=' * 80)

for ex in examples_worse:
    print(f'\n[ID {ex["eval_id"]}] {ex["query"][:40]}...')
    print(f'  Phase 1 Top-1: {ex["p1_top1"][:40]}...')
    print(f'  전략 A Top-1: {ex["sa_top1"][:40]}...')
    print(f'  → Phase 1의 Top-1이 전략 A에서 {ex["p1_rank_in_sa"]}위로 하락')
    print(f'  → 전략 A의 Top-1은 Phase 1에서 {ex["sa_rank_in_p1"]}위였음')

print('\n' + '=' * 80)
print('개선 사례 샘플 (전략 A가 더 나은 경우)')
print('=' * 80)

for ex in examples_better:
    print(f'\n[ID {ex["eval_id"]}] {ex["query"][:40]}...')
    print(f'  Phase 1 Top-1: {ex["p1_top1"][:40]}...')
    print(f'  전략 A Top-1: {ex["sa_top1"][:40]}...')
    print(f'  → 전략 A의 Top-1이 Phase 1에서 {ex["sa_rank_in_p1"]}위였음 (상승)')
