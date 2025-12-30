#!/usr/bin/env python3
import json
from models.llm_client import llm_client

with open('data/eval.jsonl', 'r') as f:
    eval_data = [json.loads(line) for line in f]

science_count = 0
non_science_count = 0

print("="*70)
print("평가 데이터셋 분류 분석 (샘플: 50개)")
print("="*70)

for i, item in enumerate(eval_data[:50]):
    query = item['msg'][0]['content']
    analysis = llm_client.analyze_query(item['msg'])
    
    if analysis.tool_calls:
        science_count += 1
        label = "SCIENCE"
    else:
        non_science_count += 1
        label = "NON-SCIENCE"
    
    if i < 10:
        print(f"{i+1}. [{label}] {query[:60]}")

print("="*70)
print(f"\n샘플 통계 (n=50):")
print(f"  과학 질문: {science_count}개 ({science_count/50*100:.1f}%)")
print(f"  비과학 질문: {non_science_count}개 ({non_science_count/50*100:.1f}%)")

est_science = int(science_count/50*220)
est_non_science = int(non_science_count/50*220)
print(f"\n추정 (전체 220개):")
print(f"  과학: {est_science}개 (검색/랭킹 필요)")
print(f"  비과학: {est_non_science}개 (topk=[] -> MAP=1)")

print(f"\n분석:")
print(f"  - {est_non_science}개 쿼리는 topk=[] (검색 결과 없음)")
print(f"  - 이들의 MAP = 1.0 (정답 판정)")
print(f"  - 따라서 전체 MAP은 과학 쿼리에 크게 좌우됨")
print(f"\n결론:")
print(f"  게이팅 정책이 너무 공격적이면 검색 결과를 차단하는 쿼리가 많아짐")
print(f"  -> 리더보드 MAP이 하락할 수 있음")
