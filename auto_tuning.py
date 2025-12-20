#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 파라미터 튜닝 자동화
3가지 Hard Voting 가중치 조합으로 자동 평가
"""

import json
import subprocess
import os

# 테스트할 가중치 조합
weight_configs = [
    ([5, 3, 1], "weights_5_3_1_baseline"),   # 현재 (MAP 0.7970)
    ([6, 3, 1], "weights_6_3_1"),
    ([7, 4, 2], "weights_7_4_2"),
]

print('=' * 80)
print('Phase 2 파라미터 튜닝 - Hard Voting 가중치')
print('=' * 80)
print(f'현재 MAP (baseline): 0.7970 [5, 3, 1]')
print(f'테스트 설정: {len(weight_configs)}개\n')

# 현재 기본 submission을 백업
if os.path.exists('./submission.csv'):
    os.system('mv ./submission.csv ./submission_phase2_baseline.csv')

# 각 설정으로 평가 실행
results = {}

for weights, name in weight_configs:
    print(f"\n{'=' * 80}")
    print(f'테스트: {name} - Weights {weights}')
    print('=' * 80)
    
    # 전역 변수로 가중치 설정
    with open('/tmp/tuning_params.txt', 'w') as f:
        f.write(f'{weights[0]},{weights[1]},{weights[2]}')
    
    # main.py 실행 (220개 전체)
    exit_code = os.system('cd /root/IR && python main.py > /tmp/tuning_log.txt 2>&1')
    
    if exit_code == 0 and os.path.exists('./submission.csv'):
        line_count = int(os.popen('wc -l ./submission.csv').read().split()[0])
        
        if line_count == 220:
            # 파일 저장
            os.system(f'cp ./submission.csv ./submission_{name}.csv')
            print(f'✅ 완료: 220개 질문 처리')
            results[name] = 'pending'
        else:
            print(f'❌ 오류: {line_count}개만 처리')
            results[name] = 'failed'
    else:
        print(f'❌ 실행 오류')
        results[name] = 'failed'

print('\n' + '=' * 80)
print('파라미터 튜닝 평가 완료')
print('=' * 80)
print('\n생성된 파일:')
for name in results:
    print(f'  - submission_{name}.csv')

print('\n다음 단계:')
print('1. 리더보드에서 3개 결과의 MAP 점수 확인')
print('2. 최고 점수 선택')
print('3. 최종 제출')
