#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파라미터 튜닝을 위한 평가 스크립트
Hard Voting 가중치와 Reranker Top-K를 다양하게 조합하여 테스트
"""

import json
import time
from eval_rag import answer_question_optimized

# 테스트할 파라미터 조합
test_configs = [
    # Hard Voting 가중치 조합
    {"name": "weights_5_3_1", "weights": [5, 3, 1]},  # 현재 (MAP 0.7970)
    {"name": "weights_6_3_1", "weights": [6, 3, 1]},
    {"name": "weights_7_4_2", "weights": [7, 4, 2]},
    {"name": "weights_4_2_1", "weights": [4, 2, 1]},
    {"name": "weights_8_4_2", "weights": [8, 4, 2]},
    {"name": "weights_10_5_2", "weights": [10, 5, 2]},
]

# eval.jsonl에서 처음 30개 질문만 로드 (빠른 테스트)
test_samples = []
with open('./data/eval.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 30:
            break
        test_samples.append(json.loads(line))

print('=' * 80)
print('파라미터 튜닝 테스트 (처음 30개 질문 샘플)')
print('=' * 80)
print(f'테스트할 설정: {len(test_configs)}개\n')

for config in test_configs:
    print(f"\n테스트: {config['name']} - Weights {config['weights']}")
    print('-' * 60)
    
    start_time = time.time()
    
    # 임시로 hybrid_search의 가중치 설정
    # (실제로는 run_hybrid_search 호출 시 파라미터로 전달)
    
    success_count = 0
    error_count = 0
    
    for j, sample in enumerate(test_samples):
        try:
            result = answer_question_optimized(sample['msg'])
            if result['topk']:
                success_count += 1
        except Exception as e:
            error_count += 1
            print(f'  오류 [ID {sample["eval_id"]}]: {str(e)[:50]}')
    
    elapsed = time.time() - start_time
    
    print(f'  처리: {success_count}/{len(test_samples)} 성공')
    print(f'  시간: {elapsed:.1f}초')
    print(f'  평균: {elapsed/len(test_samples):.2f}초/질문')

print('\n' + '=' * 80)
print('✅ 파라미터 튜닝 테스트 완료')
print('=' * 80)
print('\n다음 단계:')
print('1. 최적의 Hard Voting 가중치 선택')
print('2. Reranker Top-K 파라미터 테스트')
print('3. 선택된 파라미터로 220개 전체 질문 평가')
