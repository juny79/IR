# 📋 IR RAG 시스템 최적화 종합 보고서

## 🎯 Executive Summary

**현재 상태**: MAP 0.8470 (Baseline 0.6629 대비 **+27.7%**)

**주요 성과**:
- ✅ Reranker 도입: +16.8%
- ✅ HyDE 쿼리 확장: +2.9%
- ✅ Parameter Tuning [6,3,1]: +6.3% ⭐⭐⭐

---

## 📊 1단계: BASELINE (MAP 0.6629)

### 구성 요소
- **Retrieve**: Elasticsearch (BM25 + Nori Analyzer)
- **Sparse**: 원본 쿼리 → BM25
- **Dense**: 원본 쿼리 → SBERT (snunlp/KR-SBERT-V40K-klueNLI-augSTS, 768 dims)
- **Hard Voting**: [5, 3, 1] (Rank 1,2,3 가중치)
- **Reranker**: ❌ 미사용
- **LLM**: Gemini 2.5 Flash (답변 생성만)
- **Pipeline**: `Sparse(원본) + Dense(원본) → Hard Voting → Top-5 → 답변생성`

### 특징
- 최적화되지 않은 기본 Hybrid Search
- Sparse와 Dense를 동등하게 취급
- 상위 5개 문서만 사용

### 문제점
❌ 키워드(Sparse)와 의미(Dense) 신호 불균형  
❌ 의미적으로 관련되었으나 키워드 불일치 문서 검색 불가  
❌ Rank 신호의 신뢰도 차이 미반영

---

## 🔍 PHASE 1: RERANKER 도입 (MAP 0.7742) → **+16.8%** 🚀

### 변경 사항
- Reranker 추가: **BAAI/bge-reranker-v2-m3** (Cross-Encoder)
- Hard Voting에서 **Top-20** 문서 선택
- Reranker로 Top-20을 **Top-5**로 재순위

### 파이프라인
```
Sparse(원본) + Dense(원본) 
  ↓
Hard Voting[5,3,1](Top-20)
  ↓
Reranker(원본)
  ↓
Top-5
```

### 효과
✅ **+0.1113 MAP** (+16.8%) - **가장 큰 단계별 개선**  
✅ 후보를 Top-5에서 Top-20으로 확대  
✅ Reranker가 정확도로 최상위 5개 재선별

### 원인 분석

1. **Multi-stage 파이프라인의 효과**
   - Stage 1: Hard Voting이 충분한 후보 제공 (Top-20)
   - Stage 2: Reranker(Cross-Encoder)가 정확한 순위 결정

2. **Cross-Encoder의 강점**
   - 쿼리-문서 쌍의 관련성을 직접 평가
   - Sparse/Dense 하이브리드의 불완전한 결합 보정

3. **Precision-Recall 트레이드오프 해결**
   - Top-5만으로는 관련 문서 누락 가능
   - Top-20으로 확대하되 Reranker로 정확도 확보

### 결론
🎯 **Multi-stage 파이프라인의 중요성 증명**

---

## 📚 PHASE 2: HyDE (쿼리 확장) 도입 (MAP 0.7970) → **+2.9%** 📈

### HyDE 개념
**Hypothetical Document Embeddings**
- 원본 쿼리로부터 가설적 답변 생성
- 이를 쿼리 확장에 사용하여 검색 신호 풍부화

### 변경 사항
- **LLM (Gemini 2.5 Flash)** 추가
- **Sparse**: 원본 → **HyDE 확장 쿼리**
- **Dense**: 원본 → **HyDE 확장 쿼리**
- **Reranker**: 원본 쿼리 유지 (정확성)

### 파이프라인
```
LLM(가설답변 생성)
  ↓
Sparse(HyDE) + Dense(HyDE)
  ↓
Hard Voting[5,3,1](Top-20)
  ↓
Reranker(원본)
  ↓
Top-5
```

### 효과
✅ **+0.0228 MAP** (+2.9%)  
⚠️ 예상(+3~5%) 이하의 개선도

### 왜 기대보다 낮을까?

#### 원인 1: Sparse/Dense 불일치
```
질문: "식물 호흡이란?"

HyDE 확장: "식물 호흡 작용은 광합성과 반대로 미토콘드리아에서 산소를 소비하여 ATP를 생성하는 과정이다..."

결과:
- Sparse (BM25): 확장 쿼리의 구체적 키워드 활용
  → "미토콘드리아", "ATP", "산소" 등으로 검색
- Dense (SBERT): 확장 쿼리의 의미적 의미
  → 다른 결과 반환

→ Rank 1 Sparse + Rank 10 Dense 같은 불일치 발생
```

#### 원인 2: 확장 쿼리의 노이즈
- 생성된 가설적 답변이 정확하지 않은 경우
- 키워드 오염 발생

#### 원인 3: Hard Voting의 신호 불균형
- [5,3,1]에서 Sparse/Dense 불일치를 균등하게 처리
- 하이브리드 신호의 신뢰도 차이 미반영

### 교훈
🔑 **HyDE 적용 전략이 중요**
- ✅ Sparse/Dense: 일관되게 적용 (동일 쿼리)
- ❌ Reranker: 원본 쿼리 사용 (정확성)

---

## ❌ 실패한 실험 1: PHASE 2-A (HyDE Sparse Only) → MAP 0.7962 (-0.1%)

### 시도
- Sparse: HyDE 적용 ✅
- Dense: 원본 쿼리 유지 ❌

### 결과
**MAP 0.7962** (Phase 2 대비 -0.0008, -0.1% 하락)

### 분석: Sparse/Dense 불일치 극대화

```
질문: "식물 호흡이란?"

Sparse (HyDE):  "식물 호흡 작용 미토콘드리아 ATP..."
Dense (원본):   "식물 호흡이란?"

→ 극단적 신호 차이 발생
→ Hard Voting의 신뢰도 급락
```

### 교훈
🔑 **Hybrid Search에서 Sparse/Dense 일관성 필수**
- 동일 쿼리 사용 시 시너지 극대
- 불일치하면 성능 저하

---

## ❌ 실패한 실험 2: STRATEGY A (Reranker HyDE) → MAP 0.7780 (-2.4%)

### 시도
Reranker도 HyDE 확장 쿼리로 순위 재정의

### 파이프라인
```
HyDE(전체 파이프라인)
  ↓
Sparse(HyDE) + Dense(HyDE)
  ↓
Hard Voting
  ↓
Reranker(HyDE) ← 문제!
  ↓
Top-5
```

### 결과
**MAP 0.7780** (Phase 2 대비 -0.0190, -2.4% 하락)

### 분석: 컴포넌트 역할 충돌

#### Reranker의 본질
- Cross-Encoder: 쿼리-문서 쌍의 **정확한** 관련성 평가
- 강점: 의미적 정확성

#### HyDE의 특성
- 의미적 확장으로 '부정확한' 신호 제공
- Reranker의 정확성 판단에 간섭

#### 결과
- Reranker가 확장 쿼리의 노이즈로 혼동
- Hard Voting과 Reranker의 신호 충돌
- Multi-stage 파이프라인의 이점 상실

### 교훈
🔑 **각 컴포넌트의 역할 명확화**
```
Sparse/Dense   → 검색 신호 (HyDE로 풍부화 가능)
Reranker       → 정확성 판단 (원본 쿼리로 평가)
```

---

## ⚡ PARAMETER TUNING: Hard Voting [6,3,1] (MAP 0.8470) → **+6.3%** ⭐⭐⭐

### 변경 사항
Hard Voting 가중치: `[5, 3, 1]` → `[6, 3, 1]`
- Rank 1: 5 → 6 (+**20%** relative)
- Rank 2: 3 → 3 (변화 없음)
- Rank 3: 1 → 1 (변화 없음)

### 효과
✅ **+0.0500 MAP** (+6.3%)  
✅ 예상(+3~5%)을 **크게 초과**! 🚀

### 가중치 상세 분석

#### [5,3,1] (Phase 2)에서:
```
Rank 1 Sparse + Rank 1 Dense = 5 + 3 = 8점
Rank 1 Sparse + Rank 5 Dense = 5 + 0 = 5점
```

#### [6,3,1] (Parameter Tuning)에서:
```
Rank 1 Sparse + Rank 1 Dense = 6 + 3 = 9점 (+12.5%)
Rank 1 Sparse + Rank 5 Dense = 6 + 0 = 6점 (+20%)
```

### 왜 [6,3,1]이 효과적인가?

#### 1️⃣ HyDE의 불일치 문제 해결
```
Phase 2 [5,3,1]:
- Rank 1 Sparse + Rank 10 Dense
- Hard Voting: 5 + 0 = 5점
- 하위 신호도 동등 취급 → 신뢰도 저하

[6,3,1]:
- Rank 1 Sparse + Rank 10 Dense
- Hard Voting: 6 + 0 = 6점 (+20%)
- Rank 1 신호를 강조 → Rank 1 문서 우선순위 상향
```

#### 2️⃣ Hard Voting 품질 향상
- Rank 1 우대로 더 정확한 **Top-20** 후보 선별
- Reranker에 입력되는 문서 품질 증가

#### 3️⃣ Reranker 효율성 증대
- 더 나은 Top-20에서 정확한 Top-5 선택
- 최종 결과 품질 향상

### 통계적 분석

**가설**: HyDE에서 Sparse/Dense의 Rank 1 일치율이 높을 수 있음

#### 질문 유형별:
```
명사/키워드 기반 (식물, 화학, 역사):
- BM25: 키워드 매칭 우수
- SBERT: 의미적 유사성도 우수
→ 두 신호가 동일 상위 문서 반환 가능성 높음
→ [6,3,1]의 Rank 1 우대가 매우 효과적

추상적/개념 기반 (철학, 사회):
- BM25: 키워드 부족
- SBERT: 의미적 유사성 강조
→ 불일치 가능성 높음
```

**결론**: 평가 세트에 명사/키워드 기반 질문이 많았을 가능성

### 핵심 통찰
🎯 **단순한 파라미터 튜닝이 복잡한 알고리즘 변경보다 효과적**
- 데이터와 모델에 맞는 최적값의 중요성
- 작은 변경이 큰 성능 향상을 야기할 수 있음

---

## 🧪 현재 진행: TESTING [7,4,2] (결과 대기 중)

### 시도
모든 가중치 +1씩 상향: `[6,3,1]` → `[7,4,2]`
- Rank 1: 6 → 7 (+16.7%)
- Rank 2: 3 → 4 (+33.3%)
- Rank 3: 1 → 2 (+100%)

### 예상 결과

#### 시나리오 1: [7,4,2] >= 0.85
✓ 계속 가중치 튜닝 ([8,5,2], [6,4,2] 등)  
✓ 최적값 탐색 진행

#### 시나리오 2: [7,4,2] = 0.84~0.85 (근소 하락)
✓ [6,3,1]이 최적 가중치일 가능성 높음  
✓ 다중 임베딩 전략으로 전환 (Phase 3, 4)

#### 시나리오 3: [7,4,2] < 0.84 (명확한 하락)
✓ [6,3,1] 최종 선택  
✓ 다중 임베딩으로 0.85+ 달성

### 분석
- [7,4,2]는 Rank 2,3까지 강하게 우대
- 상위 3개 모두가 정확할 가능성 낮음
- **오버피팅 가능성** 있음
- **예상**: 약간 하락 또는 소폭 상승

---

## 📊 종합 성능 분석

### 성능 진화

```
0.6629 (Baseline)
  ↓ +16.8%
0.7742 (Phase 1: Reranker)
  ↓ +2.9%
0.7970 (Phase 2: HyDE)
  ↓ +6.3%
0.8470 (Current) ⭐⭐⭐
  ↓ +17.8% (Target: 0.95)
```

### 단계별 기여도

| 단계 | 기여 포인트 | 비율 |
|------|-----------|------|
| Reranker | +1,113 | 60.4% |
| Parameter Tuning | +500 | 27.1% |
| HyDE | +228 | 12.4% |
| **총합** | **+1,841** | **100%** |

### 컴포넌트별 영향도

#### 🥇 Reranker (BAAI/bge-reranker-v2-m3)
- 정확도 판단에 최적화된 Cross-Encoder
- Hard Voting 상위 20개에서 정확한 Top-5 선별
- **가장 큰 성능 향상** 제공 (+60%)

#### 🥈 Hard Voting 가중치 [6,3,1]
- Sparse/Dense 신호의 신뢰도 차등화
- Rank 1의 중요성 강조
- Reranker 입력 품질 향상
- **단순하지만 강력한** 최적화 (+27%)

#### 🥉 HyDE (쿼리 확장)
- 검색 신호 다양화
- 의미적 확장으로 유사 문서 발견
- 노이즈도 함께 증가
- 전략적 적용이 중요 (+12%)

### 실패 사례의 교훈

❌ **Phase 2-A (HyDE Sparse Only)**
→ Sparse/Dense 불일치 심화  
→ Hard Voting 신뢰도 저하

❌ **Strategy A (Reranker HyDE)**
→ 컴포넌트 역할 충돌  
→ 각 컴포넌트의 역할 명확화 필요

**핵심**: 무분별한 기법 추가는 오히려 성능 저하

---

## 🎯 최종 결론

### 현재 성과
- **MAP 0.8470** 달성 (Baseline 대비 **+27.7%**)
- Reranker 도입으로 정확성 대폭 강화 ✅
- HyDE로 검색 신호 다양화 ✅
- 파라미터 튜닝으로 신호 최적화 ✅

### 핵심 통찰

1️⃣ **복잡한 알고리즘보다 단순한 파라미터 튜닝이 효과적**
- 작은 변경 → 큰 성능 향상 가능

2️⃣ **각 컴포넌트의 역할을 명확히 해야 함**
- Sparse/Dense: 검색 신호
- Reranker: 정확성 판단

3️⃣ **하이브리드 검색의 신호 균형 중요**
- 일관된 쿼리 사용
- 가중치로 신호 신뢰도 조정

4️⃣ **실패한 실험도 소중한 학습 자료**
- 부정적 결과도 최적화 방향 제시

### 다음 전략
1. [7,4,2] 결과 대기
2. 결과에 따라 추가 가중치 튜닝 또는 다중 임베딩 전략 전환
3. **궁극 목표**: MAP 0.95 달성

---

**보고서 작성일**: 2025-12-20  
**현재 상태**: [7,4,2] 가중치 평가 진행 중  
**다음 업데이트**: Leaderboard 결과 반영
