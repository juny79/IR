# IR 시스템 코드 구조 및 역할 분석

## 📊 시스템 아키텍처 개요

```
Input (평가 데이터)
    ↓
main.py (메인 실행)
    ↓
eval_rag.py (통합 오케스트레이션)
    ├─→ models/llm_client.py (쿼리 분석)
    ├─→ models/solar_client.py (HyDE 확장)
    ├─→ retrieval/hybrid_search.py (하이브리드 검색)
    │   ├─→ retrieval/es_connector.py (Elasticsearch 연동)
    │   ├─→ models/embedding_client.py (다중 임베딩)
    │   └─→ retrieval/reranker.py (재순위화)
    └─→ submission.csv (최종 결과)
```

---

## 1️⃣ 핵심 실행 파일

### 📌 main.py - 전체 평가 루프 실행
**파일**: `/root/IR/main.py`

**역할**:
- 평가 데이터셋 (`data/eval.jsonl`) 읽기
- 각 질문에 대해 `eval_rag.py`의 `answer_question_optimized()` 호출
- 처리 결과를 `submission.csv`에 실시간 저장
- 이미 처리된 데이터는 건너뛰기 (재시작 안전성)

**주요 기능**:
```python
for i, line in enumerate(f, 1):
    data = json.loads(line)
    if data["eval_id"] in processed_ids:
        continue
    result = answer_question_optimized(data["msg"])
    # 결과를 submission.csv에 저장
```

**입력**: `data/eval.jsonl` (220개 질문)  
**출력**: `submission.csv` (평가 결과)

---

### 📌 eval_rag.py - 통합 오케스트레이션
**파일**: `/root/IR/eval_rag.py`

**역할**:
- 각 질문에 대한 전체 파이프라인 조율
- 설정값 정의 (VOTING_WEIGHTS, TOP_K 등)
- 질문 분석 → HyDE 생성 → 하이브리드 검색 → 답변 생성

**주요 설정값** (Phase 4D-TopK60 기준):
```python
VOTING_WEIGHTS = [5, 4, 2]          # Hard Voting 가중치
USE_MULTI_EMBEDDING = True           # SBERT + Gemini 조합
TOP_K_RETRIEVE = 60                  # TOP_K 증가
USE_RRF = False                      # Hard Voting 사용
USE_GATING = True                    # 게이팅 정책 유지
```

**처리 흐름**:
1. `llm_client.analyze_query()` - 쿼리 분석 및 분류
2. `solar_client.generate_hypothetical_answer()` - HyDE 확장
3. `run_hybrid_search()` - 하이브리드 검색 및 재순위화
4. `solar_client.generate_answer()` - 최종 답변 생성

**입력**: 사용자 메시지 (질문)  
**출력**: 
```python
{
    "standalone_query": "확장된 쿼리",
    "topk": [문서5개],
    "answer": "생성된 답변"
}
```

---

## 2️⃣ 모델 & 클라이언트 (models/)

### 🔵 models/llm_client.py - LLM 기반 쿼리 분석
**역할**:
- Gemini API를 통해 사용자 쿼리 분석
- 과학 질문 vs 비과학 질문 분류
- tool_calls 판단으로 게이팅 정책 결정

**주요 메서드**:
```python
def analyze_query(messages):
    """
    쿼리를 분석하고 tool_calls 생성 여부 판단
    - tool_calls 있음 → 과학 질문 (검색 필요)
    - tool_calls 없음 → 비과학 질문 (일상 대화)
    """
```

**API**: Gemini 2.5 Flash  
**기능**:
- 쿼리 intent 분류
- tool_calls 생성 (과학 질문만)
- 일상 대화 응답

---

### 🟠 models/solar_client.py - Solar Pro 2 HyDE
**역할**:
- Upstage Solar Pro 2 API를 통한 HyDE 쿼리 확장
- 최종 답변 생성
- 캐싱으로 비용 및 시간 절감

**주요 메서드**:
```python
def generate_hypothetical_answer(query):
    """
    Solar Pro 2로 쿼리 확장 (HyDE)
    예: "DNA의 구조는?" → "DNA는 뉴클레오타이드로..."
    """

def generate_answer(messages, context):
    """
    검색된 문서를 바탕으로 최종 답변 생성
    """
```

**기능**:
- ✅ HyDE 쿼리 확장 (캐싱 적용)
- ✅ 최종 답변 생성
- ✅ Pickle 기반 캐싱 (80% 비용 절감)

**캐싱 효과**:
- 첫 실행: Upstage API 호출 (10-20초)
- 캐시 히트: 즉시 반환 (<100ms)

---

### 🟢 models/embedding_client.py - 다중 임베딩
**역할**:
- SBERT + Gemini 두 임베딩 모델 관리
- 임베딩 캐싱으로 속도/비용 최적화

**주요 메서드**:
```python
def get_query_embedding(query, use_gemini_only=False):
    """
    SBERT 또는 Gemini로 쿼리 임베딩
    - SBERT: 로컬 모델 (빠름)
    - Gemini: API 기반 (정확함, 캐싱 적용)
    """
```

**임베딩 모델**:
1. **SBERT**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
   - 768 차원
   - 한국어 특화
   - 로컬 실행 (빠름)

2. **Gemini**: `text-embedding-004`
   - 768 차원
   - API 기반
   - 캐싱으로 34,893배 속도 향상

---

## 3️⃣ 검색 & 재순위화 (retrieval/)

### 🔴 retrieval/es_connector.py - Elasticsearch 연동
**역할**:
- Elasticsearch에 접근하여 문서 검색
- Sparse (BM25) 검색 수행
- 검색된 문서 내용 조회

**주요 메서드**:
```python
def search_sparse(query, top_k):
    """
    BM25 알고리즘으로 sparse 검색
    Solar HyDE 확장 쿼리 사용
    """

def search_dense(embedding, top_k):
    """
    임베딩 기반 dense 검색
    SBERT 또는 Gemini 임베딩 사용
    """

def get_document(doc_id):
    """
    특정 문서의 내용 조회
    """
```

**Elasticsearch 설정**:
- 인덱스: `test`
- 문서 수: 4,272개
- 필드: `docid`, `content`, `embeddings_sbert`, `embeddings_gemini` 등

---

### 🟡 retrieval/hybrid_search.py - 하이브리드 검색 & 융합
**역할**:
- Sparse (BM25) + Dense (임베딩) 검색 결합
- Hard Voting으로 두 검색 결과 융합
- Reranker로 최종 순위 조정

**주요 메서드**:
```python
def run_hybrid_search(
    original_query,
    sparse_query,
    reranker_query,
    voting_weights=[5, 4, 2],
    use_multi_embedding=True,
    top_k_retrieve=50
):
    """
    1. Sparse 검색: sparse_query (Solar HyDE)
    2. Dense 검색: original_query (SBERT + Gemini)
    3. Hard Voting 융합: voting_weights=[5,4,2]
    4. Reranker: 원본 쿼리로 재순위화
    """
```

**검색 흐름**:
```
Sparse Search (BM25)          Dense Search (SBERT)        Dense Search (Gemini)
      ↓                               ↓                              ↓
    결과 Top50                      결과 Top50                     결과 Top50
      ↓                               ↓                              ↓
    Hard Voting [5, 4, 2]  (투표로 점수 계산)
      ↓
    상위 50개 문서
      ↓
    Reranker (BAAI/bge-reranker-v2-m3)
      ↓
    최종 순위 Top5
```

---

### 🟣 retrieval/reranker.py - 재순위화
**역할**:
- BAAI/bge-reranker-v2-m3로 최종 문서 순위 조정
- 쿼리와 문서의 관련성을 정교하게 재계산

**주요 메서드**:
```python
def rerank_documents(query, documents, top_k):
    """
    BAAI Reranker로 문서 재순위화
    각 문서의 관련성 점수 재계산
    """
```

**Reranker 특징**:
- 768 차원 BERT 기반
- CrossEncoder 방식 (쿼리-문서 쌍 학습)
- 0-1 범위의 관련성 점수

---

## 4️⃣ 데이터 & 결과

### 📂 data/eval.jsonl - 평가 데이터셋
**구조**:
```json
{
  "eval_id": 78,
  "msg": [{"role": "user", "content": "질문 내용"}]
}
```

**특징**:
- 220개 질문
- 과학 질문: ~84% (184개)
- 비과학 질문: ~16% (36개)

---

### 📝 submission.csv - 최종 결과
**구조**:
```json
{
  "eval_id": 78,
  "standalone_query": "확장된 쿼리",
  "topk": ["doc_id_1", "doc_id_2", ...],
  "answer": "생성된 답변 텍스트"
}
```

**생성 방식**:
- 220줄 (각 질문당 1줄)
- 실시간으로 행 추가 (중단/재시작 안전)
- JSON Lines 형식

---

## 5️⃣ 현재 설정값 (Phase 4D-TopK60)

| 설정 | 값 | 역할 |
|------|-----|------|
| **VOTING_WEIGHTS** | [5, 4, 2] | Hard Voting에서 1,2,3위 가중치 |
| **USE_MULTI_EMBEDDING** | True | SBERT + Gemini 조합 사용 |
| **TOP_K_RETRIEVE** | 60 | 검색 후보 수 (증가됨) |
| **USE_RRF** | False | Hard Voting 사용 (RRF 아님) |
| **USE_GATING** | True | 비과학 질문 필터링 (게이팅) |

---

## 6️⃣ 성능 메트릭

### 각 모듈의 처리 시간

| 모듈 | 처음 | 캐시 히트 | 설명 |
|------|------|----------|------|
| Solar HyDE | 10-20초 | <100ms | 80% 비용 절감 |
| Gemini 임베딩 | 1-2초 | <10ms | 34,893배 속도 향상 |
| SBERT 임베딩 | <100ms | <50ms | 로컬 모델 |
| Sparse 검색 | 100-200ms | - | BM25 |
| Dense 검색 | 200-300ms | - | 2개 모델 |
| Reranker | 300-500ms | - | Top50 문서 |
| **총 시간** | **15-25초** | **<500ms** | 캐싱 적용 시 |

---

## 7️⃣ 코드 변형 파일들

### 대체 설정 파일
```
eval_rag.py (현재 사용 - Phase 4D-TopK60)
├─ eval_rag_no_gating.py (게이팅 OFF 버전)
├─ eval_rag_topk60.py (TopK60 버전)
└─ eval_rag_weight552.py (가중치 [5,5,2] 버전)
```

각 파일은 특정 파라미터 조합을 테스트할 때 사용됨.

---

## 8️⃣ 전체 데이터 흐름

```
평가 질문 (data/eval.jsonl)
    ↓
main.py [실행 루프]
    ↓
eval_rag.py [파이프라인 오케스트레이션]
    ├─ step 1: llm_client.analyze_query()
    │   └─ Gemini로 쿼리 분석 (tool_calls 판단)
    │
    ├─ step 2: solar_client.generate_hypothetical_answer()
    │   └─ Solar Pro 2로 HyDE 확장 (캐싱)
    │
    ├─ step 3: run_hybrid_search()
    │   ├─ es_connector.search_sparse() [BM25]
    │   ├─ embedding_client.get_query_embedding() [SBERT]
    │   ├─ embedding_client.get_query_embedding() [Gemini]
    │   ├─ Hard Voting [5,4,2] 융합
    │   └─ reranker.rerank_documents() [최종 순위]
    │
    ├─ step 4: es_connector.get_document()
    │   └─ Top-3 문서 내용 조회
    │
    └─ step 5: solar_client.generate_answer()
        └─ Solar Pro 2로 최종 답변 생성
    ↓
submission.csv [결과 저장]
    ↓
리더보드 제출 [MAP/MRR 평가]
```

---

## 9️⃣ 최종 요약

### 핵심 파일 체크리스트

| 파일 | 목적 | 수정 빈도 |
|------|------|----------|
| **main.py** | 전체 실행 루프 | 낮음 |
| **eval_rag.py** | 설정 + 파이프라인 | **높음** ⭐ |
| **models/llm_client.py** | 쿼리 분석 | 낮음 |
| **models/solar_client.py** | HyDE + 답변 생성 | 낮음 |
| **models/embedding_client.py** | 다중 임베딩 | 낮음 |
| **retrieval/hybrid_search.py** | 하이브리드 검색 | 중간 |
| **retrieval/es_connector.py** | ES 연동 | 낮음 |
| **retrieval/reranker.py** | 재순위화 | 낮음 |

**가장 자주 수정되는 파일**: `eval_rag.py` (설정값 변경)

---

## 🔟 현재 최고 성능 설정

**Phase 4D** (MAP 0.8424):
- Solar HyDE ✅
- SBERT + Gemini 조합 ✅
- Hard Voting [5,4,2] ✅
- TOP_K=50 ✅

**Phase 4D-TopK60** (테스트 중):
- 위와 동일하되 TOP_K=60 증가

**제출 파일**:
- `submission_nogating.csv`: 게이팅 OFF 버전
- `submission_topk60.csv`: TopK60 버전
