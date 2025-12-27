# IR 시스템 코드 파일 - 한눈에 보기

## 📋 핵심 파일 요약표

| 파일명 | 위치 | 역할 | 주요 함수 | 수정 빈도 |
|--------|------|------|----------|----------|
| **main.py** | `/root/IR/` | 평가 루프 실행 | `main()` | 낮음 |
| **eval_rag.py** | `/root/IR/` | 파이프라인 오케스트레이션 | `answer_question_optimized()` | **높음** ⭐ |
| **llm_client.py** | `/root/IR/models/` | 쿼리 분석 (Gemini) | `analyze_query()` | 낮음 |
| **solar_client.py** | `/root/IR/models/` | HyDE + 답변생성 | `generate_hypothetical_answer()`, `generate_answer()` | 낮음 |
| **embedding_client.py** | `/root/IR/models/` | 멀티 임베딩 (SBERT+Gemini) | `get_query_embedding()` | 낮음 |
| **hybrid_search.py** | `/root/IR/retrieval/` | 하이브리드 검색 | `run_hybrid_search()`, `hard_vote_results()` | 중간 |
| **es_connector.py** | `/root/IR/retrieval/` | Elasticsearch 연동 | `search_sparse()`, `search_dense()`, `get_document()` | 낮음 |
| **reranker.py** | `/root/IR/retrieval/` | 재순위화 | `rerank_documents()` | 낮음 |

---

## 🔧 eval_rag.py 설정값 (가장 자주 수정)

```python
# Phase 4D-TopK60 (현재 설정)
VOTING_WEIGHTS = [5, 4, 2]          # Hard Voting 가중치
USE_MULTI_EMBEDDING = True           # SBERT + Gemini 조합
TOP_K_RETRIEVE = 60                  # 검색 후보군 (50→60 증가)
USE_RRF = False                      # False = Hard Voting 사용
RRF_K = 60                           # (USE_RRF=False면 미사용)
USE_GATING = True                    # True = 비과학 필터링
```

**각 설정값의 의미**:
- **VOTING_WEIGHTS**: Sparse(1위), SBERT(2위), Gemini(3위)에 부여하는 가중치
- **USE_MULTI_EMBEDDING**: False면 SBERT만, True면 SBERT+Gemini 조합
- **TOP_K_RETRIEVE**: 재순위화 전 후보군 수 (크면 느림, 작으면 정확도 감소)
- **USE_RRF**: True면 RRF(순위만 사용), False면 Hard Voting(점수 사용)
- **USE_GATING**: True면 비과학 질문 필터링 (topk=[])

---

## 🔀 데이터 흐름 (간단 버전)

```
main.py (루프)
  ↓
eval_rag.py (설정)
  ├─ llm_client: 쿼리 분류
  ├─ solar_client: HyDE 확장 (캐싱)
  ├─ hybrid_search: 하이브리드 검색
  │  ├─ es_connector: Sparse/Dense 검색
  │  ├─ embedding_client: 임베딩 생성
  │  ├─ Hard Voting 융합
  │  └─ reranker: 최종 순위
  ├─ es_connector: 문서 내용 조회
  └─ solar_client: 답변 생성
  ↓
submission.csv (결과)
```

---

## 💾 캐싱 구조

```
cache/ (캐시 파일 저장 위치)
├─ hyde_cache.pkl
│  └─ key: query 해시값
│  └─ value: hypothetical_answer
│  └─ 효과: 80% 비용/시간 절감
│
└─ query_embeddings.pkl
   └─ key: query MD5 해시
   └─ value: 768차원 벡터
   └─ 효과: 34,893배 속도 향상
```

---

## 📈 각 단계별 처리 시간

| 단계 | 초회 | 캐시 | 주요 함수 |
|------|------|------|----------|
| 1. 쿼리 분석 | 1-2초 | - | `llm_client.analyze_query()` |
| 2. HyDE 확장 | 10-20초 | <100ms | `solar_client.generate_hypothetical_answer()` |
| 3. Sparse 검색 | 100-200ms | - | `es_connector.search_sparse()` |
| 4. Dense 검색 (SBERT) | 100-200ms | - | `embedding_client.get_query_embedding()` |
| 5. Dense 검색 (Gemini) | 1-2초 | <10ms | `embedding_client.get_query_embedding(gemini=True)` |
| 6. Hard Voting | 50-100ms | - | `hard_vote_results()` |
| 7. Reranker | 300-500ms | - | `reranker.rerank_documents()` |
| 8. 문서 조회 | 30-50ms | - | `es_connector.get_document()` |
| 9. 답변 생성 | 3-5초 | - | `solar_client.generate_answer()` |
| **총 시간** | **15-25초** | **<600ms** | - |

---

## 🎯 최고 성능 설정 (Phase 4D)

```
설정: [5,4,2], TopK=50, 게이팅=OFF
결과: MAP 0.8424, MRR 0.8500

파일: eval_rag.py
├─ VOTING_WEIGHTS = [5, 4, 2]
├─ TOP_K_RETRIEVE = 50
├─ USE_MULTI_EMBEDDING = True
├─ USE_GATING = False
└─ 결과: submission_17.csv (또는 최신)
```

---

## ❌ 실패했던 설정들

| Phase | 설정 | MAP | 실패 이유 |
|-------|------|-----|----------|
| Phase 3 | Solar 단독 | 0.7992 | 단일 임베딩 부족 |
| Phase 5 | RRF 알고리즘 | 0.8159 | 순위만으로는 부족 |
| Phase 6A | [6,4,2] 가중치 | 0.8265 | 1위 과도 강조 |
| Phase 6B-1 | 게이팅 ON | 0.8083 | 비과학 분류 오류 |

---

## 🚀 수정하려면 어디를?

### 설정값 변경
→ **eval_rag.py**의 상단 설정값 수정 (VOTING_WEIGHTS, TOP_K 등)

### 쿼리 분석 로직 변경
→ **models/llm_client.py** 수정

### Solar Pro 2 HyDE 커스터마이징
→ **models/solar_client.py** 수정

### 검색 알고리즘 변경
→ **retrieval/hybrid_search.py** 수정

### Elasticsearch 연동 변경
→ **retrieval/es_connector.py** 수정

### 재순위화 모델 변경
→ **retrieval/reranker.py** 수정

---

## 📊 최종 시스템 특징

✅ **장점**:
- 멀티 임베딩 조합 (SBERT + Gemini)
- Hard Voting으로 여러 검색 결과 효과적 융합
- Solar Pro 2 HyDE로 쿼리 품질 향상
- 캐싱으로 빠른 재실험 가능
- Reranker로 최종 순위 정제

❌ **한계**:
- Solar Pro 2 고정으로 다른 LLM 미시도 불가
- MAP 0.86 이상 달성 어려움
- 문서 색인 누락 시 검색 불가능

---

**가장 중요한 파일**: eval_rag.py (설정) + hybrid_search.py (검색)  
**가장 자주 수정되는 파일**: eval_rag.py  
**가장 효과적인 최적화**: 멀티 임베딩 조합 + Hard Voting [5,4,2]
