# Phase 5 회복 및 MAP 0.95 달성 전략

## 📊 현재 상황
- **Phase 4D (Best)**: MAP 0.8424, MRR 0.8500
- **Phase 5 (RRF)**: MAP 0.8159, MRR 0.8182 ❌ (-3.14%)
- **목표**: MAP 0.95 (현재 대비 +12.8% 필요)

---

## 🔍 Phase 5 실패 원인 분석

### 1. RRF 알고리즘의 구조적 한계
```python
# RRF 점수 계산: 1/(k+rank)
k=60일 때:
- rank 1: 1/61 = 0.0164
- rank 2: 1/62 = 0.0161
- rank 3: 1/63 = 0.0159
- 차이: 0.0003, 0.0002 → 너무 작음!

# Hard Voting [5,4,2]:
- rank 1: 5점
- rank 2: 4점
- rank 3: 2점
- 차이: 1점, 2점 → 명확한 차등
```

### 2. MAP 메트릭 특성
- MAP는 상위 rank의 정확도에 매우 민감
- rank 1-3의 차별화가 중요
- RRF는 rank 차이를 충분히 반영 못함

---

## 🚀 단계적 복구 및 최적화 계획

### Phase 6A: Phase 4D 복원 + 미세 조정 (우선순위 1)
**목표**: MAP 0.845-0.850 (+0.3-0.6%)
**소요시간**: 각 30분 × 3회 = 1.5시간

1. **Phase 4D 설정으로 복귀**
   ```python
   USE_RRF = False  # Hard Voting 사용
   VOTING_WEIGHTS = [5, 4, 2]
   TOP_K_RETRIEVE = 50
   ```

2. **Voting Weights 미세 조정**
   - **실험 A**: [6, 4, 2] - rank 1 강조
   - **실험 B**: [5, 4, 3] - rank 3 강화
   - **실험 C**: [7, 4, 2] - rank 1 극대화

3. **예상 결과**
   - [6,4,2]: MAP 0.845-0.848 (rank 1 정확도 개선)
   - [5,4,3]: MAP 0.843-0.846 (rank 3 품질 향상)
   - [7,4,2]: MAP 0.848-0.852 (rank 1 최우선)

---

### Phase 6B: TopK 확장 실험 (우선순위 2)
**목표**: MAP 0.850-0.860 (+0.6-1.6%)
**소요시간**: 각 30분 × 3회 = 1.5시간

**로드맵 3단계 적용**: "Solar Pro 2의 저렴한 비용을 믿고 Reranker가 검토할 후보군을 대폭 확대"

1. **TopK 확장 테스트**
   - 현재: TOP_K_RETRIEVE = 50
   - **실험 A**: 60 (+20%)
   - **실험 B**: 70 (+40%)
   - **실험 C**: 80 (+60%)

2. **논리적 근거**
   - Reranker(BGE-reranker-v2-m3)는 강력함
   - 더 많은 후보 → Reranker가 숨은 관련 문서 발견 가능
   - Solar Pro 2 비용이 저렴하므로 TopK 증가 부담 적음

3. **예상 결과**
   - TopK 60: MAP 0.850-0.855 (후보 다양성 증가)
   - TopK 70: MAP 0.855-0.860 (최적 균형점)
   - TopK 80: MAP 0.853-0.858 (오버헤드 발생 가능)

---

### Phase 6C: Reranker 전략 최적화 (우선순위 3)
**목표**: MAP 0.860-0.875 (+1.6-3.1%)
**소요시간**: 각 45분 × 2회 = 1.5시간

1. **Reranker Top-K 조정**
   ```python
   # 현재: 20개 후보 → Reranker → Top-5
   # 실험: 30개, 40개 후보로 확대
   ```

2. **Reranker 쿼리 전략**
   - **실험 A**: 원본 쿼리 + HyDE 조합
   - **실험 B**: Solar 쿼리 확장 (더 상세한 쿼리)

3. **예상 결과**
   - 30개 후보: MAP 0.860-0.865
   - 40개 후보: MAP 0.865-0.870
   - 쿼리 확장: MAP 0.870-0.875

---

### Phase 6D: Solar Pro 2 HyDE 최적화 (우선순위 4)
**목표**: MAP 0.875-0.900 (+3.1-5.8%)
**소요시간**: 각 1시간 × 3회 = 3시간

1. **HyDE 프롬프트 개선**
   - 현재: "200-300자 가설 답변 생성"
   - **실험**: 더 구체적인 프롬프트 엔지니어링
   ```python
   """
   다음 질문에 대한 정확하고 상세한 답변을 작성하세요.
   답변은 핵심 개념, 정의, 예시를 포함해야 하며,
   검색에 유용한 키워드를 최대한 많이 포함해야 합니다.
   """
   ```

2. **HyDE 길이 실험**
   - 현재: max_tokens=300
   - **실험 A**: 400 (더 상세)
   - **실험 B**: 500 (매우 상세)

3. **HyDE Temperature 조정**
   - 현재: temperature=0.3
   - **실험 A**: 0.2 (더 일관성)
   - **실험 B**: 0.4 (더 다양성)

---

### Phase 6E: Multi-Query 전략 (우선순위 5)
**목표**: MAP 0.900-0.920 (+5.8-7.8%)
**소요시간**: 2시간

1. **쿼리 확장 전략**
   ```python
   # 1개 쿼리 → 3개 변형 쿼리 생성
   original: "광합성이란?"
   variant1: "광합성의 정의와 원리"
   variant2: "식물의 광합성 과정"
   variant3: "엽록소와 광합성의 관계"
   ```

2. **Multi-Query Fusion**
   - 각 변형 쿼리로 검색 → 결과 융합
   - Hard Voting으로 최종 순위 결정

---

### Phase 6F: Ensemble 전략 (우선순위 6)
**목표**: MAP 0.920-0.950 (+7.8-10.6%)
**소요시간**: 3시간

1. **Multiple Reranker Ensemble**
   - 현재: BGE-reranker-v2-m3 단일
   - **추가**: cross-encoder/ms-marco-MiniLM-L-12-v2
   - 2개 Reranker 점수 평균

2. **Embedding Ensemble**
   - SBERT + Gemini (현재)
   - **추가**: Upstage embedding 인덱싱
   - 3개 embedding 결과 융합

---

### Phase 6G: Groundedness Check (우선순위 7)
**목표**: MAP 0.950-0.970 (+10.6-12.8%)
**소요시간**: 4시간

**로드맵 4단계 적용**: "Upstage Groundedness Check를 통해 답변의 신뢰도 확인"

1. **Upstage Groundedness Check API 통합**
   ```python
   def check_groundedness(answer, context):
       # Upstage API 호출
       # 답변이 context에 기반했는지 확인
       pass
   ```

2. **저품질 답변 필터링**
   - Groundedness 점수 < 0.7 → 재검색

3. **최종 품질 검증**
   - 모든 답변에 대해 신뢰도 확인
   - MAP 극대화

---

## 📋 실행 우선순위

### 즉시 실행 (Phase 6A)
1. **Hard Voting 복원**: USE_RRF = False
2. **Weights [6,4,2] 테스트**: 30분
3. **Weights [7,4,2] 테스트**: 30분
4. **최적 가중치 선정**

### 단기 목표 (Phase 6B-C, 3시간)
- TopK 확장: 50 → 60 → 70
- Reranker 후보 증가: 20 → 30 → 40
- **예상 MAP: 0.860-0.875**

### 중기 목표 (Phase 6D-E, 5시간)
- HyDE 프롬프트 최적화
- Multi-Query 전략 구현
- **예상 MAP: 0.900-0.920**

### 장기 목표 (Phase 6F-G, 7시간)
- Ensemble 전략
- Groundedness Check
- **목표 MAP: 0.950+**

---

## 🎯 최종 목표 달성 로드맵

```
Phase 4D: 0.8424 (현재 최고점)
    ↓ Phase 6A (1.5h): Weights 최적화
Phase 6A: 0.8500 (+0.9%)
    ↓ Phase 6B (1.5h): TopK 확장
Phase 6B: 0.8600 (+2.1%)
    ↓ Phase 6C (1.5h): Reranker 최적화
Phase 6C: 0.8750 (+3.9%)
    ↓ Phase 6D (3h): HyDE 최적화
Phase 6D: 0.9000 (+6.8%)
    ↓ Phase 6E (2h): Multi-Query
Phase 6E: 0.9200 (+9.2%)
    ↓ Phase 6F (3h): Ensemble
Phase 6F: 0.9400 (+11.6%)
    ↓ Phase 6G (4h): Groundedness
Phase 6G: 0.9500+ (+12.8%+) 🎯 목표 달성!
```

**총 소요 시간**: 약 16.5시간
**예상 성공률**: 80-90%

---

## 💡 핵심 전략

1. **Hard Voting 유지**: RRF는 MAP에 부적합
2. **점진적 개선**: 각 단계마다 검증하며 진행
3. **Solar Pro 2 활용**: 저렴한 비용으로 TopK/쿼리 확장
4. **Reranker 신뢰**: BGE-reranker-v2-m3는 강력함
5. **캐싱 활용**: Gemini API 비용 절감 완료

---

## 📌 다음 액션

**지금 당장**: Phase 6A 실행
```bash
# 1. Hard Voting 복원
USE_RRF = False
VOTING_WEIGHTS = [6, 4, 2]

# 2. 평가 실행
python3 main.py

# 3. 결과 확인 후 [7,4,2] 테스트
```
