#!/usr/bin/env python3
"""
Phase 4D 기반 다양한 조건 비교
1. Phase 4D-NoGating: 게이팅 정책 제거
2. Phase 4D-TopK60: TOP_K = 60
3. Phase 4D-TopK70: TOP_K = 70
4. Phase 4D-Weight[5,5,2]: 가중치 조정
"""

import json
import os
from pathlib import Path

configs = {
    "4D-NoGating": {
        "desc": "게이팅 정책 제거 (모든 질문 검색)",
        "settings": {
            "VOTING_WEIGHTS": "[5, 4, 2]",
            "TOP_K_RETRIEVE": 50,
            "USE_GATING": False
        }
    },
    "4D-TopK60": {
        "desc": "TOP_K = 60으로 더 넓은 후보군",
        "settings": {
            "VOTING_WEIGHTS": "[5, 4, 2]",
            "TOP_K_RETRIEVE": 60,
            "USE_GATING": True
        }
    },
    "4D-TopK70": {
        "desc": "TOP_K = 70으로 더욱 넓은 후보군",
        "settings": {
            "VOTING_WEIGHTS": "[5, 4, 2]",
            "TOP_K_RETRIEVE": 70,
            "USE_GATING": True
        }
    },
    "4D-Weight[5,5,2]": {
        "desc": "가중치 [5,5,2]: 2위 강조",
        "settings": {
            "VOTING_WEIGHTS": "[5, 5, 2]",
            "TOP_K_RETRIEVE": 50,
            "USE_GATING": True
        }
    },
}

print("="*80)
print("Phase 4D 기반 최적화 전략")
print("="*80)
print("\n기준선:")
print("  - Phase 4D: MAP 0.8424, MRR 0.8500")
print("  - Phase 6B-1 (게이팅): MAP 0.8083, MRR 0.8106 (-3.41%)\n")

print("테스트 항목:")
for idx, (name, config) in enumerate(configs.items(), 1):
    print(f"\n{idx}. {name}")
    print(f"   {config['desc']}")
    print(f"   설정: {config['settings']}")

print("\n" + "="*80)
print("실행 순서:")
print("  1. eval_rag.py 수정하여 각 설정 적용")
print("  2. main.py로 평가 실행")
print("  3. 결과 비교 분석")
print("="*80)
