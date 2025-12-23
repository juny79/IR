# Phase 7: Gemini 3 Pro 분석 기반 종합 최적화 보고서

## 📋 개요

**날짜**: 2025-12-22  
**목표**: Gemini 3 Pro의 종합 분석 결과를 바탕으로 MAP 점수를 결정적으로 끌어올리기  
**기준선**: Phase 4D (MAP 0.8424)

---

## 🎯 Gemini 3 Pro 제안 분석

### 1. 게이팅(Gating) 전략: "MAP 0점 방지"

**문제점**:
- 일상 대화를 과학 질문으로 오판 시 → MAP 0점
- Phase 6B-1에서 게이팅 정책 실패 (MAP 0.8083)

**Gemini 3 Pro 제안**:
- 이중 검증(Self-Correction) 도입
- 전략적 "Top-K 비우기"
- Confidence 스코어 기반 판단

### 2. 검색(Retrieval) 고도화: "HyDE의 양면성 제어"

**문제점**:
- HyDE 답변이 너무 길면 검색 노이즈 발생
- 단일 쿼리만 사용하여 재현율(Recall) 부족

**Gemini 3 Pro 제안**:
- HyDE 답변 길이 제한 (300자 → 100자)
- 멀티 쿼리 생성 (3가지 핵심 키워드 조합)
- 임베딩 가중치 차등화

### 3. 앙상블 및 재순위화(Reranking) 최적화

**문제점**:
- TOP_K=60으로는 Reranker가 정답을 놓칠 수 있음
- Hard Voting의 계단식 점수 차이

**Gemini 3 Pro 제안**:
- TOP_K를 100으로 확대
- RRF(Reciprocal Rank Fusion) 재시도
- Reranker 후보군 확대

---

## 🔧 Phase 7 구현 내용

### Phase 7A: TOP_K 100 확대

**목적**: Reranker 후보군 증가 → 정답 발견 확률 향상

**변경사항**:
```python
# eval_rag.py
TOP_K_RETRIEVE = 100  # 60 → 100

# hybrid_search.py
candidates_with_scores = hard_vote_results(
    sparse_res, 
    dense_results,
    top_k=50,  # Reranker 후보 20 → 50
    weights=voting_weights
)
```

**근거**:
- BGE-Reranker-v2-m3는 성능이 우수하므로 더 넓은 범위 필요
- Phase 4D (TOP_K=50)에서 0.8424 달성
- Phase 4D-TopK60 테스트 진행 중

**예상 효과**: +0.01~0.02 MAP

---

### Phase 7B: HyDE 길이 제한

**목적**: 가설 답변의 노이즈 감소 → 임베딩 품질 향상

**변경사항**:
```python
# eval_rag.py
HYDE_MAX_LENGTH = 100  # 300자 → 100자

# solar_client.py
prompt = f"""당신은 과학 백과사전 전문가입니다. 다음 질문에 대한 핵심 답변을 작성하세요.

요구사항:
1. 100자 이내로 핵심만 작성
2. 전문 용어와 핵심 개념만 포함
3. 백과사전 스타일의 간결한 문장
4. 불필요한 설명 제외

질문: {query}

핵심 답변:"""
```

**근거**:
- 300자 HyDE는 부연 설명이 많아 Dense 검색 시 노이즈 발생
- 100자로 제한하면 핵심 개념에 집중 가능
- Sparse 검색(BM25)에는 여전히 충분한 키워드 제공

**예상 효과**: +0.005~0.01 MAP

---

### Phase 7C: 이중 게이팅 검증 (Cross-check)

**목적**: "일상 대화를 과학 질문으로 오판" 방지 → MAP 0점 회피

**변경사항**:
```python
# eval_rag.py
USE_DOUBLE_CHECK = True

# solar_client.py
def verify_science_query(self, query):
    """
    Gemini가 1차로 "과학 질문"이라고 판단한 후,
    Solar Pro 2가 2차 검증
    
    Returns:
        {
            "is_science": bool,
            "confidence": "high/medium/low",
            "reason": str
        }
    """
```

**프로세스**:
1. Gemini: 1차 판단 (tool_calls 있으면 과학 질문)
2. Solar Pro 2: 2차 검증 (정말 검색이 필요한가?)
3. 확신도가 낮으면 → topk=[] (MAP 점수 확보)

**근거**:
- Phase 6B-1 실패 원인: 게이팅 정책으로 topk=[] 반환했으나 MAP 0.8083
- 대회 룰: 비과학 질문에 topk 반환 시 0점, 비우면 1점
- 이중 검증으로 오판률 감소

**예상 효과**: 오판 케이스 5~10% 감소 → +0.005~0.01 MAP

---

### Phase 7D: 멀티 쿼리 생성

**목적**: Sparse 검색(BM25)의 재현율(Recall) 향상

**변경사항**:
```python
# eval_rag.py
USE_MULTI_QUERY = True

# solar_client.py
def generate_multi_query(self, query):
    """
    3가지 핵심 키워드 조합 생성
    
    예시:
    - 원본: "광합성이란?"
    - 멀티: ["광합성 엽록체 과정", "식물 광합성 반응", "CO2 산소 광합성"]
    """

# hybrid_search.py
# 각 멀티 쿼리로 추가 Sparse 검색 수행
for mq in multi_queries:
    additional_res = sparse_retrieve(mq, top_k_retrieve // 2)
    additional_sparse_results.append(additional_res)
```

**근거**:
- 단일 쿼리는 키워드 제한적
- 3가지 조합으로 다양한 관련 문서 발견
- Sparse 검색은 빠르므로 추가 비용 적음

**예상 효과**: +0.01~0.015 MAP

---

## 📊 Phase 7 종합 설정

| 설정 항목 | Phase 4D | Phase 7 | 변경 내용 |
|---------|----------|---------|-----------|
| TOP_K | 50 | 100 | ×2 |
| HyDE 길이 | 300자 | 100자 | ÷3 |
| 게이팅 검증 | Gemini만 | Gemini + Solar | 이중 검증 |
| 쿼리 확장 | HyDE 1개 | HyDE 1개 + 멀티 3개 | +3 쿼리 |
| Reranker 후보 | 20 | 50 | ×2.5 |
| VOTING_WEIGHTS | [5, 4, 2] | [5, 4, 2] | 동일 |
| 임베딩 | SBERT + Gemini | SBERT + Gemini | 동일 |

---

## 🔍 예상 MAP 점수

### 보수적 추정
- Phase 4D 기준: 0.8424
- Phase 7A (TOP_K 100): +0.01 → 0.8524
- Phase 7B (HyDE 100자): +0.005 → 0.8574
- Phase 7C (이중 검증): +0.005 → 0.8624
- Phase 7D (멀티 쿼리): +0.01 → 0.8724

**최종 예상**: **MAP 0.87~0.88**

### 낙관적 추정
- 각 최적화가 시너지 효과 발생
- TOP_K 100에서 이전에 놓친 정답 발견
- 멀티 쿼리로 재현율 대폭 향상

**최종 예상**: **MAP 0.88~0.90**

---

## ⚠️ 잠재적 리스크

### 1. API 호출 증가
- **문제**: 멀티 쿼리 3개 + 이중 검증 = API 호출 4배
- **영향**: 평가 시간 20분 → 40~60분
- **대응**: 캐싱으로 2차 평가 시 80% 감소

### 2. HyDE 길이 제한 부작용
- **문제**: 100자가 너무 짧으면 정보 손실
- **영향**: Dense 검색 품질 저하
- **대응**: Sparse 검색(BM25)은 여전히 HyDE 활용, Reranker가 최종 보정

### 3. 이중 검증 오류
- **문제**: Solar가 과학 질문을 비과학으로 오판
- **영향**: 정답을 찾을 수 없음
- **대응**: 확신도가 "low"면 검색 진행

---

## 📈 성능 모니터링

### 평가 중 확인사항
1. **처리 시간**: 평균 5~10초/문항 (220문항 → 20~40분)
2. **캐시 적중률**: HyDE 80%, Gemini 95%
3. **이중 검증 비율**: 전체의 5~10%가 Solar에 의해 비과학으로 재판단
4. **멀티 쿼리 효과**: Sparse 검색 결과 다양성 증가

### 평가 완료 후 분석
```python
# Phase 7 결과 분석
import json

with open('submission.csv') as f:
    results = [json.loads(line) for line in f]

# 통계
total = len(results)
with_topk = sum(1 for r in results if r['topk'])
without_topk = total - with_topk

print(f"전체: {total}")
print(f"topk 반환: {with_topk} ({with_topk/total*100:.1f}%)")
print(f"topk 비움: {without_topk} ({without_topk/total*100:.1f}%)")
```

---

## 🎓 학습 내용

### Gemini 3 Pro 분석의 핵심 통찰

1. **게이팅은 양날의 검**: 
   - 과학 질문을 놓치면 MAP 0점
   - 비과학 질문에 검색하면 MAP 0점
   - → 이중 검증으로 안전장치 마련

2. **HyDE의 적정 길이**:
   - 너무 짧으면 정보 부족
   - 너무 길면 노이즈 발생
   - → 100자가 최적 균형점

3. **재현율(Recall)의 중요성**:
   - MAP은 관련 문서가 상단에 있어야 점수 상승
   - 정답이 검색 결과에 없으면 Reranker도 무용지물
   - → 멀티 쿼리로 재현율 향상

4. **Reranker의 잠재력**:
   - BGE-Reranker-v2-m3는 매우 강력
   - 더 많은 후보를 주면 더 좋은 결과
   - → TOP_K 100으로 확대

---

## 📝 다음 단계

### Phase 7 평가 완료 후
1. ✅ submission.csv 생성 확인
2. ✅ 리더보드 제출
3. ✅ MAP 점수 확인
4. ✅ Phase 4D (0.8424) 대비 개선도 분석

### 추가 최적화 (필요 시)
- Phase 7E: 임베딩 가중치 차등화 ([5, 4, 3] → [5, 3, 4])
- Phase 7F: RRF k 파라미터 튜닝 (60 → 80)
- Phase 7G: Reranker 배치 크기 최적화

---

## 🏆 결론

**Phase 7은 Gemini 3 Pro의 전문적 분석을 바탕으로 설계된 종합 최적화입니다.**

4가지 핵심 최적화:
- ⭐ **7A**: TOP_K 100 (더 넓은 후보군)
- ⭐ **7B**: HyDE 100자 (노이즈 감소)
- ⭐ **7C**: 이중 게이팅 (오판 방지)
- ⭐ **7D**: 멀티 쿼리 (재현율 향상)

**예상 MAP**: 0.87~0.90  
**기준 대비**: +0.045~0.08 (Phase 4D 0.8424 → Phase 7 0.87~0.90)

---

**평가 진행 중**: `python main.py`  
**로그 파일**: `phase_7_evaluation.log`  
**예상 완료**: 40~60분 후
