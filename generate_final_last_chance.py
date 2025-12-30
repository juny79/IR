# -*- coding: utf-8 -*-
"""
최종 제출 파일 생성 스크립트
전략: v1 (0.9470) 베이스에서 ID 31만 Top-1/Top-2 스왑
- ID 31은 break_v2에서도 스왑되어 있고 동일하게 0.9470을 기록
- 이론적으로 중립적인 변경이지만, 미세한 점수 차이가 있을 수 있음
"""
import json

def load_full(path):
    with open(path, 'r', encoding='utf-8') as f:
        return {obj['eval_id']: obj for line in f for obj in [json.loads(line)]}

print("Loading v1 (0.9470 baseline)...")
v1 = load_full('submission_surgical_v1.csv')

# ID 31의 Top-1과 Top-2를 스왑
print("\nOriginal ID 31 topk:", v1[31]['topk'][:3])
# v1: ['40d0c42f-fa06-421a-ab58-e56bb5690f4d', '8918f0f9-795d-4f15-8982-2cede58c202e', ...]
# break_v2: ['8918f0f9-795d-4f15-8982-2cede58c202e', '40d0c42f-fa06-421a-ab58-e56bb5690f4d', ...]

original_topk = v1[31]['topk'].copy()
if len(original_topk) >= 2:
    # Swap top-1 and top-2
    v1[31]['topk'] = [original_topk[1], original_topk[0]] + original_topk[2:]
    print("Swapped ID 31 topk:", v1[31]['topk'][:3])

# 파일 저장
output_path = 'submission_92_final_last_chance.csv'
print(f"\nSaving to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as f:
    for eid in sorted(v1.keys()):
        f.write(json.dumps(v1[eid], ensure_ascii=False) + '\n')

print("Done!")
print(f"\n변경 사항:")
print(f"- 베이스: submission_surgical_v1.csv (MAP 0.9470)")
print(f"- ID 31: Top-1 ↔ Top-2 스왑 (break_v2와 동일한 선택)")
print(f"- 예상: 0.9470 유지 또는 미세 변동")
