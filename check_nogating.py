#!/usr/bin/env python3
"""
Phase 4D-NoGating 결과 리더보드 제출 및 조회
"""

import subprocess
import json
import os
import time

# submission_nogating.csv를 리더보드에 제출하고 결과 조회
submission_file = "/root/IR/submission_nogating.csv"

print("="*80)
print("Phase 4D-NoGating 결과 조회")
print("="*80)

if os.path.exists(submission_file):
    print(f"\n✅ 제출 파일 확인: {submission_file}")
    print(f"   파일 크기: {os.path.getsize(submission_file)} bytes")
    
    # 첫 3줄 확인
    with open(submission_file, 'r') as f:
        lines = f.readlines()[:3]
    print(f"\n📊 파일 샘플 (처음 3줄):")
    for i, line in enumerate(lines, 1):
        data = json.loads(line)
        print(f"   {i}. eval_id={data['eval_id']}, topk={len(data['topk'])}개")
else:
    print(f"❌ 파일 없음: {submission_file}")

print("\n" + "="*80)
print("다음 단계: 리더보드에 제출하여 MAP/MRR 점수 확인")
print("="*80)
