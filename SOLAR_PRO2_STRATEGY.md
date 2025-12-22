# Solar Pro 2 중심 최적화 전략

## 실험 결과 핵심 인사이트

### 발견 1: SBERT가 병목
| Embedding | LLM | MAP | 평가 |
|-----------|-----|-----|------|
| **SBERT** | Solar Pro 2 | 0.5053 | ❌ 최악 |
| **Gemini** | Solar Pro 2 | 0.8985 | ✅ 우수 |
| **Upstage** | Solar Pro 2 | 0.8970 | ✅ 우수 |
| **Gemini** | Gemini | 0.8985 | ✅ 우수 (현재와 동일) |

**결론:** Solar Pro 2는 우수한 LLM이지만, SBERT embedding과 궁합이 나쁨

### 발견 2: Multi-embedding의 위력
- **단일 Gemini**: MAP 0.8985
- **Multi-embedding** (SBERT + Upstage + Upstage_HyDE + Gemini + Gemini_HyDE): MAP **0.9121** (+1.5%)

### 발견 3: 최적 설정
- **TopK**: 40 (MAP 0.9061) > 45 (0.8970) > 50 (0.8955)
- **Voting**: [5:3:1] (MAP 0.9424) - CSV 하드보팅 기준
- **LLM**: Solar Pro 2
- **검색**: BM25 + Multi-embedding

## Phase 4: Solar Pro 2 기반 재구축

### 전략 A: Gemini Embedding 단독 (추천)
**예상 MAP: 0.89-0.90** (+5.4% to +6.2%)

#### 구현 단계
1. **Gemini Embedding 인덱싱** (2-3시간)
   - API: `text-embedding-004` (768 dims)
   - 4,272 documents 재인덱싱
   
2. **Solar Pro 2 HyDE 활성화**
   - eval_rag.py에서 Solar client 사용
   
3. **설정 최적화**
   - TopK: 40
   - Voting: [5:3:1]

#### 예상 결과
```
Phase 2: Gemini LLM + SBERT = 0.8470
Phase 4A: Solar Pro 2 + Gemini Embedding = 0.89-0.90
개선: +5-6% (목표 0.95에 94-95% 도달)
```

### 전략 B: Multi-Embedding 앙상블 (최고 성능)
**예상 MAP: 0.91-0.92** (+7.4% to +8.6%)

#### 구현 단계
1. **다중 임베딩 인덱싱** (4-5시간)
   - Gemini: text-embedding-004
   - Upstage: solar-embedding-1-large
   - SBERT: 기존 유지 (snunlp/KR-SBERT-V40K-klueNLI-augSTS)

2. **하이브리드 검색 통합**
   - BM25 (sparse)
   - SBERT dense
   - Gemini dense
   - Upstage dense

3. **Solar Pro 2 적용**
   - HyDE 생성
   - 최종 답변 생성

4. **Hard Voting [5:3:1]**
   - TopK=40 per method
   - 각 방법의 Top-10 결과 통합

#### 예상 결과
```
실험: Solar Pro 2 + Multi-embedding = 0.9121
우리: Solar Pro 2 + Multi-embedding = 0.91-0.92
개선: +7-9% (목표 0.95에 96-97% 도달)
```

## 구현 우선순위

### 🔴 Phase 4A: Gemini Embedding (우선)
**시간:** 2-3시간  
**위험:** 낮음  
**예상 MAP:** 0.89-0.90  
**근거:** 실험에서 검증됨 (Solar + Gemini = 0.8985)

**장점:**
- ✅ 단일 변수 변경 (SBERT → Gemini)
- ✅ 구현 간단
- ✅ 실험 결과 명확
- ✅ 5-6% 성능 향상 보장

**단점:**
- ⚠️ 최고점(0.95)까지 0.05-0.06 gap 남음

### 🟡 Phase 4B: Multi-Embedding (차선)
**시간:** 4-5시간  
**위험:** 중간  
**예상 MAP:** 0.91-0.92  
**근거:** 실험에서 0.9121 달성

**장점:**
- ✅ 최고 성능 (실험 검증)
- ✅ 목표(0.95)에 96-97% 접근
- ✅ 다양한 semantic 표현

**단점:**
- ⚠️ 구현 복잡
- ⚠️ 인덱싱 시간 오래 걸림
- ⚠️ 시스템 자원 많이 사용

## 즉시 실행 계획

### 1단계: Gemini Embedding 인덱싱 시작 (지금 시작, 2-3시간)

```bash
# 1. Gemini embedding API 테스트
python3 -c "
from models.embedding_client import get_gemini_embedding
test_text = '광합성은 식물이 빛을 이용하여 포도당을 만드는 과정입니다.'
embedding = get_gemini_embedding(test_text)
print(f'Gemini embedding dimension: {len(embedding)}')
print(f'First 5 values: {embedding[:5]}')
"

# 2. Elasticsearch에 gemini_embedding 필드 추가
python3 scripts/add_gemini_embedding_field.py

# 3. 전체 문서 재인덱싱 (2-3시간 소요)
python3 scripts/index_gemini_embeddings.py
```

### 2단계: Solar Pro 2 활성화 (인덱싱 중 병행 가능)

```python
# eval_rag.py 수정
from models.solar_client import solar_client

# HyDE 생성
hyde_answer = solar_client.generate_hypothetical_answer(standalone_query)

# 검색 (Gemini embedding 사용)
results = hybrid_search(
    query=hyde_answer,
    top_k=40,
    use_gemini_embedding=True  # 새로 추가
)

# 최종 답변 (Solar Pro 2 사용)
final_answer = solar_client.generate_answer(question, context)
```

### 3단계: 설정 최적화

```python
VOTING_WEIGHTS = [5, 3, 1]  # 실험 최적값
TOP_K_RETRIEVE = 40  # 실험 최적값
USE_SOLAR_PRO2 = True
USE_GEMINI_EMBEDDING = True
```

### 4단계: 평가 및 제출 (인덱싱 완료 후)

```bash
# Phase 4A 평가
python3 main.py  # → submission_11.csv
# 예상: MAP 0.89-0.90
```

## Phase 4A vs 4B 비교

| 항목 | Phase 4A (Gemini) | Phase 4B (Multi) |
|------|-------------------|------------------|
| 인덱싱 시간 | 2-3시간 | 4-5시간 |
| 구현 복잡도 | 낮음 | 높음 |
| 예상 MAP | 0.89-0.90 | 0.91-0.92 |
| 목표 도달률 | 94-95% | 96-97% |
| 위험도 | 낮음 | 중간 |
| **권장 순서** | **1순위** | 2순위 |

## 실행 전략

### 오늘 (토요일)
1. **Phase 4A 구현** (2-3시간 인덱싱 + 1시간 코드 수정)
2. **평가 및 제출** (30분)
3. **결과 확인** → MAP 0.89+ 달성시 성공

### 내일 (일요일) - Phase 4A 성공시
1. **Phase 4B 추가 구현** (Upstage embedding 추가)
2. **Multi-embedding 앙상블**
3. **최종 평가** → MAP 0.91-0.92 목표

### 내일 (일요일) - Phase 4A 실패시
1. **원인 분석**
2. **Upstage embedding 시도**
3. **또는 Phase 2로 복귀**

## 기대 효과

### Phase 4A 성공시
```
현재: MAP 0.8470 (Phase 2)
목표: MAP 0.89-0.90 (Phase 4A)
개선: +5.0% to +6.2%
순위: 상위권 진입 예상
```

### Phase 4B 성공시
```
현재: MAP 0.8470 (Phase 2)
목표: MAP 0.91-0.92 (Phase 4B)
개선: +7.4% to +8.6%
순위: 최상위권 예상 (목표 0.95의 96-97% 도달)
```

## 위험 관리

### 백업 계획
1. **Phase 2 설정 보존**: eval_rag.py.backup 생성
2. **SBERT 인덱스 유지**: 기존 인덱스 삭제하지 않음
3. **단계별 검증**: 각 단계마다 테스트 후 진행

### 실패시 대응
1. Gemini embedding 문제 → Upstage embedding 시도
2. Solar Pro 2 문제 → Gemini LLM으로 복귀
3. 전체 실패 → Phase 2로 롤백

## 즉시 시작

**지금 바로 실행할 명령:**

```bash
# Gemini embedding 인덱싱 스크립트 생성 및 실행
# 2-3시간 소요, 백그라운드 실행
```

**이것이 Solar Pro 2를 활용한 올바른 전략입니다.**
- ❌ Solar + SBERT = 0.5053 (실패)
- ✅ Solar + Gemini = 0.8985 (성공)
- ✅ Solar + Multi-embedding = 0.9121 (최고)
