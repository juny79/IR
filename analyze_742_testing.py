#!/usr/bin/env python3
"""
[7,4,2] Hard Voting Weight Tuning - 결과 분석 및 비교
=========================================================
"""

print("="*80)
print("🧪 Hard Voting 가중치 진화 과정")
print("="*80)

results = {
    "Baseline [5,3,1] (Phase 2)": {
        "map": 0.7970,
        "mrr": 0.8015,
        "description": "HyDE Full 구현"
    },
    "Optimized [6,3,1]": {
        "map": 0.8470,
        "mrr": 0.8500,
        "improvement": "+6.3%",
        "description": "Rank 1 우대 (+20%)"
    },
    "Testing [7,4,2]": {
        "map": "?",
        "mrr": "?",
        "description": "강한 Rank 차등화 (모든 가중치 +1)",
        "status": "대기 중..."
    }
}

print("\n📊 진행 상황")
print("-" * 80)
for label, data in results.items():
    map_val = data["map"]
    improvement = f" ({data.get('improvement', '')})" if "improvement" in data else ""
    status = f" [{data.get('status', '')}]" if "status" in data else ""
    print(f"  {label:25} | MAP: {map_val!s:8} | MRR: {data['mrr']!s:8}{improvement}{status}")
    print(f"    └─ {data['description']}")

print("\n" + "="*80)
print("🔍 [7,4,2] vs [6,3,1] 비교 분석")
print("="*80)

print("\n📈 가중치 변화:")
print("  [6,3,1]  →  [7,4,2]")
print("  - Rank 1: 6 → 7 (+1, +16.7%)")
print("  - Rank 2: 3 → 4 (+1, +33.3%)")
print("  - Rank 3: 1 → 2 (+1, +100%)")

print("\n💡 [7,4,2]의 특징:")
print("  ✅ Rank 간 차등화 더욱 강화")
print("  ✅ Rank 2와 Rank 3의 가중치가 더욱 상향")
print("  ✅ 상위 문서들의 우대 정도가 극대화")
print("  ⚠️  Rank 2,3의 문서들이 과도하게 우대될 가능성")

print("\n🎯 기대되는 결과:")
print("  시나리오 1: [7,4,2] > 0.85  → 계속 가중치 튜닝 ([8,5,2] 등)")
print("  시나리오 2: [7,4,2] = 0.84~0.85  → [6,3,1] 최적 가중치일 수 있음")
print("  시나리오 3: [7,4,2] < 0.84  → 다중 임베딩 전략으로 전환")

print("\n📋 다음 단계:")
print("  1. [7,4,2] 결과 대기")
print("  2. 결과에 따라 다음 최적화 결정")
print("  3. 최상의 가중치로 최종 제출")

print("\n" + "="*80)
