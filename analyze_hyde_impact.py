#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

# submission.csv 분석
with open('./submission.csv', 'r', encoding='utf-8') as f:
    results = [json.loads(line) for line in f]

# submission_reranker.csv (Phase 1) 분석
with open('./submission_reranker.csv', 'r', encoding='utf-8') as f:
    reranker_results = [json.loads(line) for line in f]

print('=== Phase 1 (Reranker) vs Phase 2 (HyDE+Reranker) 비교 ===\n')

# standalone_query 길이 비교
reranker_query_lens = [len(r['standalone_query']) for r in reranker_results]
hyde_query_lens = [len(r['standalone_query']) for r in results]

print(f'1. Standalone Query 길이:')
print(f'   Phase 1 평균: {sum(reranker_query_lens)/len(reranker_query_lens):.1f}자')
print(f'   Phase 2 평균: {sum(hyde_query_lens)/len(hyde_query_lens):.1f}자')
print(f'   (Phase 2는 standalone_query만 저장, HyDE는 내부에서만 사용)\n')

# Top-K 문서 변화 분석
same_count = 0
partial_match = 0
total_different = 0

for i in range(len(results)):
    r1_topk = set(reranker_results[i]['topk'][:5])
    r2_topk = set(results[i]['topk'][:5])
    
    overlap = len(r1_topk & r2_topk)
    
    if overlap == 5:
        same_count += 1
    elif overlap > 0:
        partial_match += 1
    else:
        total_different += 1

print(f'2. Top-5 문서 변화:')
print(f'   동일 (5/5): {same_count}개 ({same_count/220*100:.1f}%)')
print(f'   부분 변화 (1~4/5): {partial_match}개 ({partial_match/220*100:.1f}%)')
print(f'   완전 변화 (0/5): {total_different}개 ({total_different/220*100:.1f}%)\n')

# 답변 길이 비교
reranker_answer_lens = [len(r['answer']) for r in reranker_results]
hyde_answer_lens = [len(r['answer']) for r in results]

print(f'3. 답변 길이:')
print(f'   Phase 1 평균: {sum(reranker_answer_lens)/len(reranker_answer_lens):.0f}자')
print(f'   Phase 2 평균: {sum(hyde_answer_lens)/len(hyde_answer_lens):.0f}자\n')

# 샘플 비교
print(f'4. Top-K 변화 샘플 (처음 5개 부분 변화 케이스):')
count = 0
for i in range(len(results)):
    if count >= 5:
        break
    
    r1_topk = reranker_results[i]['topk'][:5]
    r2_topk = results[i]['topk'][:5]
    
    overlap = len(set(r1_topk) & set(r2_topk))
    
    if 0 < overlap < 5:
        count += 1
        print(f'\n   [{count}] ID {results[i]["eval_id"]}: {results[i]["standalone_query"][:40]}...')
        print(f'       Phase 1: {r1_topk[:2]}...')
        print(f'       Phase 2: {r2_topk[:2]}...')
        print(f'       중복: {overlap}/5개')
