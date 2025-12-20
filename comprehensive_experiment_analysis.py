#!/usr/bin/env python3
"""
📊 IR RAG 시스템 최적화 실험 과정 및 결과 분석
================================================

지금까지 실행된 모든 실험과 각 단계의 컴포넌트 구성,
MAP 값 변화 및 원인 분석을 정리한 종합 보고서
"""

import json

print("="*100)
print("🧪 IR RAG 시스템 최적화 여정: Baseline → MAP 0.8470 (+27%)")
print("="*100)

# 실험 데이터
experiments = [
    {
        "phase": "BASELINE",
        "name": "기본 Hybrid Search",
        "map": 0.6629,
        "mrr": 0.6712,
        "delta": "기준점",
        "components": {
            "retrieve": "Elasticsearch (BM25 + Nori)",
            "sparse": "원본 쿼리 → BM25",
            "dense": "원본 쿼리 → SBERT (snunlp/KR-SBERT-V40K)",
            "hard_voting": "[5, 3, 1] - Rank 1,2,3 가중치",
            "reranker": "❌ 미사용",
            "llm": "원본 쿼리만 사용, HyDE 미사용",
            "answer_gen": "Gemini 2.5 Flash"
        },
        "pipeline": "Sparse(원본) + Dense(원본) → Hard Voting → Top-5 → 답변생성",
        "analysis": "모든 신호를 균등하게 활용, 최적화 미진행"
    },
    {
        "phase": "PHASE 1",
        "name": "Reranker 도입",
        "map": 0.7742,
        "mrr": 0.7850,
        "delta": "+0.1113 (+16.8%)",
        "components": {
            "retrieve": "Elasticsearch (BM25 + Nori)",
            "sparse": "원본 쿼리 → BM25",
            "dense": "원본 쿼리 → SBERT",
            "hard_voting": "[5, 3, 1]",
            "reranker": "✅ BAAI/bge-reranker-v2-m3 (Cross-Encoder)",
            "llm": "원본 쿼리만 사용",
            "answer_gen": "Gemini 2.5 Flash"
        },
        "pipeline": "Sparse(원본) + Dense(원본) → Hard Voting(Top-20) → Reranker → Top-5 → 답변생성",
        "analysis": "Hard Voting으로 후보 20개 선별 → Reranker가 정확도 높은 Top-5 재순위\n         → 의미적 유사성과 정확성이 결합되어 큰 성능 향상 달성"
    },
    {
        "phase": "PHASE 2",
        "name": "HyDE Full Implementation",
        "map": 0.7970,
        "mrr": 0.8015,
        "delta": "+0.0228 (+2.9%)",
        "components": {
            "retrieve": "Elasticsearch (BM25 + Nori)",
            "sparse": "HyDE 확장 쿼리 → BM25",
            "dense": "HyDE 확장 쿼리 → SBERT",
            "hard_voting": "[5, 3, 1]",
            "reranker": "✅ BAAI/bge-reranker-v2-m3",
            "llm": "✅ HyDE: Gemini로 가설적 답변 생성",
            "answer_gen": "Gemini 2.5 Flash"
        },
        "pipeline": "HyDE(가설답변) → Sparse(HyDE) + Dense(HyDE) → Hard Voting(Top-20) → Reranker(원본쿼리) → Top-5 → 답변생성",
        "analysis": "쿼리 확장으로 더 풍부한 검색 신호 제공\n         지만 Sparse/Dense가 다른 결과를 제공하면서 기대만큼 큰 향상 없음\n         (+2.9%는 예상보다 낮은 개선도)"
    },
    {
        "phase": "PHASE 2-A",
        "name": "HyDE Sparse Only (실패한 변형)",
        "map": 0.7962,
        "mrr": 0.7995,
        "delta": "-0.0008 (-0.1%) vs Phase 2",
        "components": {
            "retrieve": "Elasticsearch (BM25 + Nori)",
            "sparse": "HyDE 확장 쿼리 → BM25",
            "dense": "원본 쿼리 → SBERT (HyDE 미적용)",
            "hard_voting": "[5, 3, 1]",
            "reranker": "✅ BAAI/bge-reranker-v2-m3",
            "llm": "부분 HyDE 적용 (Sparse만)",
            "answer_gen": "Gemini 2.5 Flash"
        },
        "pipeline": "HyDE(Sparse만) + Dense(원본) → Hard Voting(Top-20) → Reranker → Top-5 → 답변생성",
        "analysis": "Sparse/Dense의 불일치 심화\n         Dense가 원본 쿼리만 사용하면서 시너지 감소\n         → Phase 2보다 약간 하락 (일관성이 중요함을 보여줌)"
    },
    {
        "phase": "STRATEGY A",
        "name": "Reranker 쿼리로도 HyDE 사용 (실패)",
        "map": 0.7780,
        "mrr": 0.7810,
        "delta": "-0.0190 (-2.4%) vs Phase 2",
        "components": {
            "retrieve": "Elasticsearch (BM25 + Nori)",
            "sparse": "HyDE 확장 쿼리 → BM25",
            "dense": "HyDE 확장 쿼리 → SBERT",
            "hard_voting": "[5, 3, 1]",
            "reranker": "✅ BAAI/bge-reranker-v2-m3 (HyDE 쿼리 사용)",
            "llm": "전체 파이프라인에 HyDE 적용",
            "answer_gen": "Gemini 2.5 Flash"
        },
        "pipeline": "HyDE(전체) → Sparse(HyDE) + Dense(HyDE) → Hard Voting(Top-20) → Reranker(HyDE) → Top-5 → 답변생성",
        "analysis": "Reranker가 HyDE 확장 쿼리로 순위 재정의\n         그러나 HyDE는 의미적 확장이므로 Reranker의 '정확한 관련성 판단'과 충돌\n         → 오버피팅 가능성, MAP 하락"
    },
    {
        "phase": "PARAMETER TUNING",
        "name": "Hard Voting [6,3,1] - 가중치 최적화",
        "map": 0.8470,
        "mrr": 0.8500,
        "delta": "+0.0500 (+6.3%) vs Phase 2 🚀",
        "components": {
            "retrieve": "Elasticsearch (BM25 + Nori)",
            "sparse": "HyDE 확장 쿼리 → BM25",
            "dense": "HyDE 확장 쿼리 → SBERT",
            "hard_voting": "[6, 3, 1] - Rank 1 우대 (+20%)",
            "reranker": "✅ BAAI/bge-reranker-v2-m3 (원본 쿼리)",
            "llm": "HyDE Full (Sparse/Dense)",
            "answer_gen": "Gemini 2.5 Flash"
        },
        "pipeline": "HyDE(Sparse/Dense) → Hard Voting[6,3,1](Top-20) → Reranker(원본) → Top-5 → 답변생성",
        "analysis": "Rank 1의 가중치를 5→6으로 상향 (+20% relative)\n         HyDE의 Sparse/Dense 불일치를 Rank 1 신호 강화로 보정\n         → Hard Voting으로 더 정확한 Top-20 후보 생성\n         → Reranker가 더 나은 문서에서 선택 가능\n         → 예상 (+3~5%) 을 크게 초과해 +6.3% 달성! ⭐"
    },
    {
        "phase": "TESTING [7,4,2]",
        "name": "강한 Rank 차등화 (현재 테스트 중)",
        "map": "?",
        "mrr": "?",
        "delta": "?",
        "components": {
            "retrieve": "Elasticsearch (BM25 + Nori)",
            "sparse": "HyDE 확장 쿼리 → BM25",
            "dense": "HyDE 확장 쿼리 → SBERT",
            "hard_voting": "[7, 4, 2] - 모든 가중치 +1씩 상향",
            "reranker": "✅ BAAI/bge-reranker-v2-m3 (원본 쿼리)",
            "llm": "HyDE Full",
            "answer_gen": "Gemini 2.5 Flash"
        },
        "pipeline": "HyDE(Sparse/Dense) → Hard Voting[7,4,2](Top-20) → Reranker(원본) → Top-5 → 답변생성",
        "analysis": "Rank 1: 6→7 (+16.7%), Rank 2: 3→4 (+33.3%), Rank 3: 1→2 (+100%)\n         → 상위 문서들의 우대 정도 극대화\n         → 예상: [6,3,1]이 최적값이면 약간 하락, 미튜닝이면 추가 상승 가능"
    }
]

# 테이블 출력
print("\n" + "="*100)
print("1️⃣ 실험별 MAP 값 변화 추적")
print("="*100)

for exp in experiments:
    print(f"\n📍 {exp['phase']:20} | {exp['name']:40}")
    print(f"   MAP: {exp['map']:>8} | MRR: {exp['mrr']:>8} | 변화: {exp['delta']:>20}")
    
print("\n" + "-"*100)
print("성능 변화 그래프:")
print("-"*100)
map_scores = [
    (0.6629, "BASELINE"),
    (0.7742, "Phase 1: Reranker"),
    (0.7970, "Phase 2: HyDE"),
    (0.7962, "Phase 2-A: HyDE Sparse"),
    (0.7780, "Strategy A: Full HyDE"),
    (0.8470, "PARAM [6,3,1] ⭐"),
]

for score, label in map_scores:
    bar_length = int(score * 50)
    print(f"  {label:30} | {'█' * bar_length} {score:.4f}")

# 상세 분석
print("\n" + "="*100)
print("2️⃣ 각 실험 단계별 상세 분석")
print("="*100)

for i, exp in enumerate(experiments, 1):
    print(f"\n{'='*100}")
    print(f"[{i}] {exp['phase']}: {exp['name']}")
    print(f"{'='*100}")
    
    print(f"\n📊 성능:")
    map_str = f"{exp['map']:.4f}" if isinstance(exp['map'], float) else str(exp['map'])
    mrr_str = f"{exp['mrr']:.4f}" if isinstance(exp['mrr'], float) else str(exp['mrr'])
    print(f"   MAP: {map_str}, MRR: {mrr_str}, 개선: {exp['delta']}")
    
    print(f"\n🔧 컴포넌트 구성:")
    for component, detail in exp['components'].items():
        print(f"   • {component:15}: {detail}")
    
    print(f"\n🔄 처리 파이프라인:")
    print(f"   {exp['pipeline']}")
    
    print(f"\n💡 분석:")
    print(f"   {exp['analysis']}")

# 핵심 발견사항
print("\n" + "="*100)
print("3️⃣ 핵심 발견사항 및 교훈")
print("="*100)

insights = [
    {
        "title": "🎯 Reranker의 가치",
        "content": "Phase 1에서 +16.8% 성능 향상\n   - Hard Voting으로 Top-20 후보 생성\n   - Reranker(Cross-Encoder)가 의미적 정확성으로 Top-5 재순위\n   - 2단계 파이프라인이 매우 효과적"
    },
    {
        "title": "🔍 HyDE의 한계와 가능성",
        "content": "Phase 2 진행:\n   - 쿼리 확장으로 풍부한 검색 신호 제공\n   - 하지만 Sparse/Dense가 다른 결과 제공 → +2.9%에 그침\n   - HyDE가 모든 컴포넌트에 적용되면 오히려 하락 (Strategy A)\n   - 전략적 적용이 중요 (Sparse/Dense는 일관성, Reranker는 원본)"
    },
    {
        "title": "⚡ 파라미터 튜닝의 가치",
        "content": "Parameter Tuning [6,3,1]:\n   - 단순한 가중치 변경으로 +6.3% 성능 향상 (Phase 2 대비)\n   - 복잡한 알고리즘 변경보다 효과적!\n   - [5,3,1]은 일반적인 기본값이나 우리 데이터셋에는 차선\n   - Rank 1 우대로 HyDE의 Sparse/Dense 불일치 보정"
    },
    {
        "title": "🚀 성능 향상의 누적 효과",
        "content": "Baseline (0.6629) → 최종 (0.8470):\n   - Phase 1 (Reranker): +16.8%\n   - Phase 2 (HyDE): +2.9%\n   - Parameter Tuning: +6.3%\n   - 총 +27.7% 개선 달성!\n   - 각 단계의 기여도: Reranker > Parameter Tuning > HyDE"
    }
]

for insight in insights:
    print(f"\n{insight['title']}")
    print(f"   {insight['content']}")

# 다음 단계
print("\n" + "="*100)
print("4️⃣ 다음 최적화 단계 가이드")
print("="*100)

print("""
✅ 완료된 최적화:
   1. Phase 1: Reranker 도입 (+16.8%) ✅
   2. Phase 2: HyDE 쿼리 확장 (+2.9%) ✅
   3. Parameter Tuning: Hard Voting [6,3,1] (+6.3%) ✅
   
📋 현재 진행 중:
   4. [7,4,2] 가중치 테스트 (예상: -0.5~+0.5%)
   
🎯 향후 전략 (결과에 따라):
   
   시나리오 1: [7,4,2] >= 0.85
      → 추가 가중치 튜닝 계속 ([8,5,2], [6,4,1] 등)
      → 최적 가중치 찾기
   
   시나리오 2: [7,4,2] = 0.84~0.85
      → [6,3,1]이 최적값일 가능성 높음
      → 다중 임베딩 전략으로 전환
      - Phase 3: 추가 한국어 임베딩 (jhgan/ko-sroberta)
      - Phase 4: 다중 임베딩 결합 (Upstage Solar + Gemini)
   
   시나리오 3: [7,4,2] < 0.84
      → [6,3,1] 최종 선택
      → 다중 임베딩으로 0.85+ 목표
""")

# 최종 요약 테이블
print("\n" + "="*100)
print("5️⃣ 최종 요약: 컴포넌트별 영향도 분석")
print("="*100)

print("""
┌─────────────────────┬──────────────┬─────────────┬──────────────────┐
│ 컴포넌트            │ 도입 시기    │ MAP 개선    │ 핵심 역할         │
├─────────────────────┼──────────────┼─────────────┼──────────────────┤
│ Reranker (BGE)      │ Phase 1      │ +16.8%      │ Top-5 정확성 향상 │
│ HyDE (쿼리 확장)    │ Phase 2      │ +2.9%       │ 검색 신호 풍부화  │
│ Hard Voting [6,3,1] │ 파라미터튜닝 │ +6.3%       │ 신호 가중치 최적화│
│ Sparse (BM25)       │ Baseline     │ (기준)      │ 키워드 매칭       │
│ Dense (SBERT)       │ Baseline     │ (기준)      │ 의미 유사성       │
│ LLM (Gemini)        │ Phase 2      │ +2.9%       │ 쿼리 확장         │
└─────────────────────┴──────────────┴─────────────┴──────────────────┘

영향도 순위:
   1순위: Reranker (정확성 +16.8%)
   2순위: Hard Voting 튜닝 (+6.3%)
   3순위: HyDE (+2.9%)
""")

print("\n" + "="*100)
print("✨ 결론: 현재 MAP 0.8470 달성, 추가 최적화는 [7,4,2] 결과에 따라 결정")
print("="*100)
