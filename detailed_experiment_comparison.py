#!/usr/bin/env python3
"""
📋 실험별 상세 비교 테이블
==========================
"""

def print_table(headers, rows, col_widths=None):
    """간단한 텍스트 테이블 출력"""
    if col_widths is None:
        col_widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    
    # 헤더
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    
    print(separator)
    print(header_row)
    print(separator)
    
    # 행
    for row in rows:
        row_str = "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + " |"
        print(row_str)
    
    print(separator)

# 실험 데이터
data = {
    "Phase": ["Baseline", "Phase 1", "Phase 2", "Phase 2-A", "Strategy A", "Param [6,3,1]", "Testing [7,4,2]"],
    "MAP": [0.6629, 0.7742, 0.7970, 0.7962, 0.7780, 0.8470, "?"],
    "MRR": [0.6712, 0.7850, 0.8015, 0.7995, 0.7810, 0.8500, "?"],
    "개선도": ["기준", "+16.8%", "+2.9%", "-0.1%", "-2.4%", "+6.3%", "?"],
}

print("\n" + "="*100)
print("📊 MAP 값 비교 테이블")
print("="*100)
print_table(["Phase", "MAP", "MRR", "개선도"],
            [[data["Phase"][i], data["MAP"][i], data["MRR"][i], data["개선도"][i]] for i in range(len(data["Phase"]))])

# 컴포넌트 비교 테이블
components_data = [
    ["Baseline", "원본", "원본", "[5,3,1]", "❌", "-", 0.6629],
    ["Phase 1", "원본", "원본", "[5,3,1]", "✅", "원본", 0.7742],
    ["Phase 2", "HyDE", "HyDE", "[5,3,1]", "✅", "원본", 0.7970],
    ["Param [6,3,1]", "HyDE", "HyDE", "[6,3,1]", "✅", "원본", 0.8470],
    ["Testing [7,4,2]", "HyDE", "HyDE", "[7,4,2]", "✅", "원본", "?"],
]

print("\n" + "="*100)
print("🔧 컴포넌트 비교")
print("="*100)
print_table(["Phase", "Sparse 쿼리", "Dense 쿼리", "Hard Voting", "Reranker", "Reranker 쿼리", "MAP"],
            components_data)

# 가중치 상세 분석
weights_data = [
    ["1위", 5, 6, 7, "+1 (+20%)", "+2 (+40%)"],
    ["2위", 3, 3, 4, "0 (0%)", "+1 (+33%)"],
    ["3위", 1, 1, 2, "0 (0%)", "+1 (+100%)"],
    ["합계", 9, 10, 13, "+1 (+11%)", "+4 (+44%)"],
]

print("\n" + "="*100)
print("⚖️  Hard Voting 가중치 상세 비교")
print("="*100)
print_table(["Rank", "[5,3,1]", "[6,3,1]", "[7,4,2]", "[6,3,1] vs [5,3,1]", "[7,4,2] vs [5,3,1]"],
            weights_data)

# 각 단계별 구체적 변화
print("\n" + "="*100)
print("🔄 각 단계별 구체적 변화")
print("="*100)

changes = [
    {
        "from": "Baseline → Phase 1",
        "change": "Reranker 도입",
        "map_change": "+0.1113 (+16.8%)",
        "reason": "Hard Voting Top-20 + Reranker 정확성 재순위"
    },
    {
        "from": "Phase 1 → Phase 2",
        "change": "HyDE 쿼리 확장",
        "map_change": "+0.0228 (+2.9%)",
        "reason": "Sparse/Dense 모두 HyDE 적용, 검색 신호 풍부화"
    },
    {
        "from": "Phase 2 → Param [6,3,1]",
        "change": "Hard Voting [5,3,1] → [6,3,1]",
        "map_change": "+0.0500 (+6.3%)",
        "reason": "Rank 1 우대로 상위 문서 신뢰도 증가 → Reranker 입력 품질 향상"
    },
    {
        "from": "Param [6,3,1] → Testing [7,4,2]",
        "change": "모든 가중치 +1씩 상향",
        "map_change": "?",
        "reason": "강한 Rank 차등화, Rank 2,3도 대폭 상향"
    }
]

for i, change in enumerate(changes, 1):
    print(f"\n[{i}] {change['from']}")
    print(f"    변화: {change['change']}")
    print(f"    MAP: {change['map_change']}")
    print(f"    분석: {change['reason']}")

# 실패한 실험 교훈
print("\n" + "="*100)
print("❌ 실패한 실험에서의 교훈")
print("="*100)

failures = [
    {
        "name": "Phase 2-A: HyDE Sparse Only",
        "result": "MAP 0.7962 (-0.0008 vs Phase 2)",
        "lesson": "Sparse/Dense 일관성 중요\nHyDE를 일관되게 적용해야 시너지 발생"
    },
    {
        "name": "Strategy A: Reranker에도 HyDE",
        "result": "MAP 0.7780 (-0.0190 vs Phase 2)",
        "lesson": "Reranker는 정확한 관련성 판단이 핵심\nHyDE 확장 쿼리는 관련성 판단을 오히려 방해"
    }
]

for failure in failures:
    print(f"\n🔴 {failure['name']}")
    print(f"   결과: {failure['result']}")
    print(f"   교훈: {failure['lesson']}")

# 최종 성능 요약
print("\n" + "="*100)
print("🎯 최종 성능 요약")
print("="*100)

summary = f"""
Baseline (0.6629) 대비:
├─ Phase 1: +16.8% (0.7742)
├─ Phase 2: +20.4% (0.7970)  
├─ Param [6,3,1]: +27.7% (0.8470) ⭐⭐⭐
└─ Target: 0.95 (17.8% 추가 필요)

단계별 누적 효과:
├─ Reranker: +1,113 MAP 포인트
├─ HyDE: +228 MAP 포인트  
├─ 파라미터튜닝: +500 MAP 포인트
└─ 총 누적: +1,841 MAP 포인트 (+27.7%)

컴포넌트별 영향도:
1. Reranker: +1113 (60.4%)
2. Parameter Tuning: +500 (27.1%)
3. HyDE: +228 (12.4%)
"""

print(summary)

print("="*100)
