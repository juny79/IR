# Phase 4C 분석 및 MAP 상향 전략

## 현재 성능 현황

| Phase | 설정 | MAP | MRR | 상태 |
|-------|------|-----|-----|------|
| Phase 2 | Gemini + [6,3,1] + TopK=50 | **0.8470** | - | ✅ 최고점 |
| Phase 2.1 | Gemini + [7,3,1] + TopK=40 | 0.8333 | - | -1.6% |
| Phase 3 | Solar + [5,3,1] + TopK=50 | 0.7992 | - | ❌ -5.6% |
| Phase 4A | Solar + Gemini only + [5,3,1] + TopK=40 | 0.8212 | - | -3.0% |
| Phase 4B | Solar + SBERT+Gemini + [5,3,1] + TopK=50 | **0.8303** | **0.8333** | -2.0% |
| **Phase 4C** | **Solar + SBERT+Gemini + [6,3,1] + TopK=50** | **예상 0.84-0.85** | - | **+1-2% 예상** |

## Phase 4B 실패 원인 분석

### 1. Solar HyDE vs Gemini HyDE (-1.3%)
```
실험 주장: Solar + Gemini embedding = MAP 0.8985
우리 결과: Solar + SBERT+Gemini = MAP 0.8303

근본 원인:
- 실험은 5개 embedding 앙상블 (SBERT + Upstage + Upstage_HyDE + Gemini + Gemini_HyDE)
- 우리는 2개 embedding (SBERT + Gemini)만 사용
- 또한 Gemini HyDE (0.8470)가 Solar HyDE (0.8303)보다 +2.0% 우수

결론: 이 문서 컬렉션에는 Gemini HyDE가 더 적합
```

### 2. 가중치 [5,3,1] vs [6,3,1] (-0.5% 추정)
```
Phase 2: [6,3,1] = 0.8470 (검증된 최고점)
Phase 4B: [5,3,1] = 0.8303

Rank 1 신뢰도가 높을 때 [6,3,1]이 더 효과적
```

## Phase 4C 전략 (현재 실행 중)

**설정:**
- HyDE: Solar Pro 2 (유지)
- Embedding: SBERT + Gemini (유지)
- Weight: [6, 3, 1] (개선)
- TopK: 50 (유지)

**기대 효과:**
- Phase 4B (0.8303) + [6,3,1] 개선 = **0.84-0.85** (+1.0-2.0%)
- 여전히 Phase 2 (0.8470)보다는 낮을 가능성

## 차선책: Phase 2로 복귀 또는 추가 최적화

### 옵션 A: Phase 2 완전 복귀
```python
HyDE: Gemini (Solar 제거)
Embedding: SBERT only (Gemini 제거)
Weight: [6, 3, 1]
TopK: 50
예상: 0.8470 (확실한 성능)
```

### 옵션 B: Gemini HyDE + SBERT + Gemini embedding + [6,3,1]
```python
HyDE: Gemini (최고 성능)
Embedding: SBERT + Gemini (새로운 조합)
Weight: [6, 3, 1]
TopK: 50
예상: 0.85-0.87 (Phase 2 < X < 실험)
```

### 옵션 C: 다른 TopK 값 세밀 조정
```
Phase 2.1에서 TopK=40이 0.8333을 달성
Phase 4B에서 TopK=50이 0.8303을 달성
→ TopK=45가 최적일 수 있음

테스트: TopK=45, 55 (각 30분)
예상: +0.5-1.0% 개선 가능
```

### 옵션 D: Query Expansion
```
질문을 자동으로 확장하여 검색 쿼리 풍부화
예: "광합성이란?" → "광합성 정의 과정 메커니즘 에너지"

구현: retrieval/query_expansion.py 작성
시간: 1-2시간 + 1시간 평가
예상: +1-2% 개선
```

## MAP 0.95 달성을 위한 로드맵

### 현재 최고점: Phase 2 (0.8470) = 89.2% 도달

### 남은 5.3% 증가 필요

**방안:**
1. **단기 (30분-1시간):**
   - ✅ Phase 4C [6,3,1] 재시도 (진행 중)
   - 예상: 0.84-0.85 (+1-2%)

2. **중기 (1-2시간):**
   - Gemini HyDE + Multi-embedding 조합
   - 예상: 0.85-0.87 (+3-4%)
   
3. **장기 (2-3시간):**
   - Query Expansion 구현
   - Reranker 프롬프트 최적화
   - HyDE 프롬프트 튜닝
   - 예상: 0.88-0.90 (+5-6%)

## 즉시 조치

1. Phase 4C 완료 대기 (20분)
2. 결과 분석
3. Phase 4C 성공 시:
   - 리더보드 제출
   - 옵션 B 또는 D 실행
4. Phase 4C 실패 시:
   - Phase 2 복귀
   - TopK 세밀 조정
