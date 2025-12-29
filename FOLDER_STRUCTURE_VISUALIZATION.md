# 📂 프로젝트 폴더 구조 상세도

## 🌳 전체 디렉토리 트리

```
/root/IR/
│
├── 📁 finetune/                                    # 파인튜닝 파이프라인
│   ├── 🔵 1_generate_qa.py                        # Stage 1: QA 생성
│   ├── 🟢 2_mine_negatives_v3.py                  # Stage 2: Hard Negative Mining
│   ├── 🟡 3_run_train_v3.sh                       # Stage 3: BGE-M3 학습
│   ├── 📊 1_generate_qa.log                       # QA 생성 로그
│   ├── 📊 3_run_train.log                         # v1 학습 로그 (268 steps)
│   └── 📊 train_v2.log                            # v2 학습 로그 (402 steps)
│
├── 📁 data/                                        # 데이터 디렉토리
│   ├── 📄 corpus.jsonl                            # 원본 문서 (4,272개)
│   ├── 📄 synthetic_qa_solar.jsonl                # 생성 QA (12,816개)
│   ├── 📄 train_data_v3.jsonl                     # 학습 데이터 (12,816개)
│   ├── 📄 test.jsonl                              # 평가 질문 (220개)
│   └── ...
│
├── 📁 finetuned_bge_m3/                            # v1 파인튜닝 모델
│   ├── 🏆 model.safetensors                       # 2.27GB 모델 가중치
│   ├── ⚙️ config.json                             # 모델 설정
│   ├── 📝 tokenizer_config.json                   # 토크나이저 설정
│   ├── 📝 tokenizer.json                          # 토크나이저
│   ├── 📝 special_tokens_map.json                 # 특수 토큰
│   └── 📝 training_args.bin                       # 학습 인자
│
├── 📁 finetuned_bge_m3_v2/                         # v2 파인튜닝 모델 (402 steps)
│   ├── 🏆 model.safetensors                       # 2.27GB
│   └── ... (동일 구조)
│
├── 📁 finetuned_bge_m3_v3/                         # v3 파인튜닝 모델 (최종, 12K)
│   ├── 🏆 model.safetensors                       # 2.27GB
│   └── ... (동일 구조)
│
├── 📄 eval_rag.py                                  # 메인 평가 스크립트
├── 📄 eval_rag_finetuned.py                        # 파인튜닝 모델 평가
├── 📄 eval_finetuned_v9.log                        # v9 평가 로그
├── 📄 eval_rag_finetuned.log                       # 파인튜닝 평가 로그
│
├── 📄 submission_surgical_v1.csv                   # 현재 최고 (MAP 0.9470)
├── 📄 submission_54_bge_m3_sota.csv                # v1 평가 (206KB)
├── 📄 submission_55_bge_m3_sota.csv                # v2 평가 (175KB)
├── 📄 submission_56_bge_m3_sota_v3.csv             # v3 평가 (178KB)
├── 📄 submission_57_bge_m3_sota_v4.csv             # 파라미터 조정 (183KB)
├── 📄 submission_58_bge_m3_sota_v5.csv             # 파라미터 조정 (176KB)
├── 📄 submission_59_bge_m3_sota_v6.csv             # 파라미터 조정 (179KB)
├── 📄 submission_60_bge_m3_sota_v7.csv             # 파라미터 조정 (188KB)
├── 📄 submission_61_bge_m3_solar_sota.csv          # Solar 통합 (309KB)
├── 📄 submission_88_ready_bge_m3_*.csv             # 최종 제출 (107KB)
├── 📄 submission_bge_m3_finetuned.csv              # 기본 평가 (415KB)
├── 📄 submission_bge_m3_finetuned_v9.csv           # v9 평가 (391KB)
└── ... (20+ 더 많은 submission 파일)
│
├── 📄 SYNTHETIC_FINETUNING_COMPREHENSIVE_REPORT.md # 종합 보고서
├── 📄 FINETUNING_WORKFLOW_SUMMARY.md              # 워크플로우 요약
├── 📄 LEADERBOARD_SUBMISSION_HISTORY.md           # 리더보드 이력
│
└── ... (기타 분석 및 실험 파일)
```

---

## 🔍 주요 디렉토리 설명

### 1. `/finetune/` - 파인튜닝 파이프라인
**목적**: 합성 데이터 생성 및 모델 학습 자동화

```
finetune/
├── 1_generate_qa.py          # Solar Pro 2로 QA 생성
├── 2_mine_negatives_v3.py    # BM25+Dense+Reranker로 Hard Negatives
└── 3_run_train_v3.sh         # BGE-M3 Contrastive Learning
```

**워크플로우**:
```
Documents → QA Generation → Hard Negative Mining → Model Training
```

---

### 2. `/data/` - 데이터 디렉토리
**목적**: 원본 문서, 생성 데이터, 학습 데이터 저장

```
data/
├── corpus.jsonl              # 4,272 documents
├── synthetic_qa_solar.jsonl  # 12,816 QA pairs (3 Q per doc)
├── train_data_v3.jsonl       # 12,816 samples (1 pos + 7 neg)
└── test.jsonl                # 220 evaluation queries
```

**데이터 변환**:
```
4,272 docs → 12,816 QA → 102,528 doc-query pairs
```

---

### 3. `/finetuned_bge_m3_*` - 파인튜닝 모델
**목적**: 학습된 임베딩 모델 저장

```
finetuned_bge_m3_v3/
├── model.safetensors         # 2.27GB XLM-RoBERTa weights
├── config.json               # Model configuration
├── tokenizer*.json           # Tokenizer files
└── training_args.bin         # Training arguments
```

**모델 버전**:
- **v1**: 4,272 samples, 2 epochs, 268 steps (초기)
- **v2**: 4,272 samples, 2+ epochs, 402 steps (개선)
- **v3**: 12,816 samples, 5 epochs, ~1000+ steps (최종)

---

### 4. `/submission_*` - 제출 파일
**목적**: 리더보드 평가 결과 저장

```
submission_*.csv 패턴:
├── submission_54-61_bge_m3_*.csv    # v1-v3 평가 (8개)
├── submission_88_*.csv              # 최종 제출
├── submission_bge_m3_finetuned*.csv # 다양한 평가 (2개)
└── ... (총 20+ 파일)
```

**제출 전략**:
- 각 파일은 서로 다른 파라미터 조합 테스트
- Hard Voting: [6,3,1], [7,4,2], [5,3,1] 등
- HyDE: Full, Sparse Only, None
- Reranker: Top-5, Top-10, Top-20

---

## 📊 파일 크기 및 통계

### 모델 파일
```
finetuned_bge_m3/           2.27GB
finetuned_bge_m3_v2/        2.27GB
finetuned_bge_m3_v3/        2.27GB
─────────────────────────────────
총 모델 크기:               6.81GB
```

### 데이터 파일
```
corpus.jsonl                ~10MB   (4,272 docs)
synthetic_qa_solar.jsonl    ~15MB   (12,816 QA)
train_data_v3.jsonl         ~150MB  (12,816 samples × 8 docs)
─────────────────────────────────
총 데이터 크기:             ~175MB
```

### 제출 파일
```
submission_*.csv            48KB ~ 440KB (평균 ~180KB)
총 20+ 파일                 ~4MB
```

---

## 🔢 데이터 규모 요약

| 항목 | 수량 | 크기 |
|------|------|------|
| **원본 문서** | 4,272개 | ~10MB |
| **생성 QA** | 12,816개 | ~15MB |
| **학습 샘플** | 12,816개 | ~150MB |
| **파인튜닝 모델** | 3개 | 6.81GB |
| **제출 파일** | 20+ | ~4MB |
| **총 디스크 사용량** | - | ~7.5GB |

---

## 🚀 실행 순서

### 1단계: 환경 설정
```bash
cd /root/IR
pip install -r requirements.txt
```

### 2단계: QA 생성
```bash
cd finetune
python 1_generate_qa.py
# → data/synthetic_qa_solar.jsonl 생성
```

### 3단계: Hard Negative Mining
```bash
python 2_mine_negatives_v3.py
# → data/train_data_v3.jsonl 생성
```

### 4단계: 모델 학습
```bash
bash 3_run_train_v3.sh
# → finetuned_bge_m3_v3/ 생성
```

### 5단계: 평가
```bash
cd ..
python eval_rag_finetuned.py
# → submission_*.csv 생성
```

---

## 📁 주요 파일 상세

### `finetune/1_generate_qa.py`
**목적**: Solar Pro 2 API로 문서당 3개 질문 생성

**입력**:
- `data/corpus.jsonl` (4,272 docs)

**출력**:
- `data/synthetic_qa_solar.jsonl` (12,816 QA pairs)

**프로세스**:
```python
for each document:
    context = document[:1000]  # 1000자 제한
    questions = solar_pro_2.generate(
        prompt="문서를 읽고 3개의 질문 생성",
        context=context
    )
    save_qa_pair(docid, questions, content)
```

---

### `finetune/2_mine_negatives_v3.py`
**목적**: Hybrid Retrieval로 Hard Negatives 7개 추출

**입력**:
- `data/synthetic_qa_solar.jsonl` (12,816 QA pairs)

**출력**:
- `data/train_data_v3.jsonl` (12,816 samples)

**프로세스**:
```python
for each qa_pair:
    # 1. BM25 Sparse Search
    bm25_candidates = elasticsearch.search(query, top_k=50)
    
    # 2. Dense Search
    dense_candidates = faiss.search(query_embedding, top_k=50)
    
    # 3. Pool Merge
    pool = merge_and_dedupe(bm25_candidates, dense_candidates)
    
    # 4. Reranker
    reranked = bge_reranker.rerank(query, pool)
    hard_negatives = reranked[:7]
    
    save_training_sample(query, positive_doc, hard_negatives)
```

---

### `finetune/3_run_train_v3.sh`
**목적**: BGE-M3 Contrastive Learning 실행

**입력**:
- `data/train_data_v3.jsonl` (12,816 samples)
- Base Model: `BAAI/bge-m3`

**출력**:
- `finetuned_bge_m3_v3/` (2.27GB model)

**하이퍼파라미터**:
```bash
--num_train_epochs 5
--per_device_train_batch_size 2
--gradient_accumulation_steps 16  # effective batch = 32
--learning_rate 1e-5
--temperature 0.02
--fp16
```

---

### `eval_rag_finetuned.py`
**목적**: 파인튜닝 모델로 평가 및 제출 파일 생성

**입력**:
- `finetuned_bge_m3_v3/` (학습된 모델)
- `data/test.jsonl` (220 queries)

**출력**:
- `submission_*.csv` (220 rows)

**프로세스**:
```python
# 1. Load fine-tuned model
model = load_finetuned_bge_m3("finetuned_bge_m3_v3")

# 2. Build index
index = build_faiss_index(corpus, model)

# 3. Evaluate
for query in test_queries:
    # HyDE expansion
    hyde_query = gemini_hyde(query)
    
    # Sparse + Dense retrieval
    bm25_results = bm25_search(hyde_query)
    dense_results = faiss_search(hyde_query, model, index)
    
    # Hard Voting
    voted = hard_vote(bm25_results, dense_results, weights=[6,3,1])
    
    # Reranker
    final = rerank(query, voted[:20], top_k=5)
    
    save_submission(query_id, final)
```

---

## 🎯 파일 역할 매핑

| 파일 | 역할 | 입력 | 출력 |
|------|------|------|------|
| `1_generate_qa.py` | QA 생성 | corpus.jsonl | synthetic_qa_solar.jsonl |
| `2_mine_negatives_v3.py` | Hard Negative | synthetic_qa_solar.jsonl | train_data_v3.jsonl |
| `3_run_train_v3.sh` | 모델 학습 | train_data_v3.jsonl | finetuned_bge_m3_v3/ |
| `eval_rag_finetuned.py` | 평가 | test.jsonl + model | submission_*.csv |

---

## 💡 파일 명명 규칙

### Submission 파일
```
submission_{번호}_{모델}_{버전}_{특징}.csv

예시:
- submission_54_bge_m3_sota.csv          # 54번 제출, bge_m3, sota 설정
- submission_56_bge_m3_sota_v3.csv       # v3 모델 사용
- submission_61_bge_m3_solar_sota.csv    # Solar 통합
- submission_88_ready_bge_m3_*.csv       # 최종 제출 (88번)
```

### 모델 디렉토리
```
finetuned_bge_m3_{버전}/

예시:
- finetuned_bge_m3/           # v1 (초기)
- finetuned_bge_m3_v2/        # v2 (개선)
- finetuned_bge_m3_v3/        # v3 (최종)
```

### 데이터 파일
```
{목적}_{버전}.jsonl

예시:
- corpus.jsonl                # 원본 (버전 없음)
- synthetic_qa_solar.jsonl    # Solar로 생성
- train_data_v3.jsonl         # v3 학습 데이터
```

---

## 📚 관련 문서

- **종합 보고서**: [SYNTHETIC_FINETUNING_COMPREHENSIVE_REPORT.md](SYNTHETIC_FINETUNING_COMPREHENSIVE_REPORT.md)
- **워크플로우 요약**: [FINETUNING_WORKFLOW_SUMMARY.md](FINETUNING_WORKFLOW_SUMMARY.md)
- **리더보드 이력**: [LEADERBOARD_SUBMISSION_HISTORY.md](LEADERBOARD_SUBMISSION_HISTORY.md)

---

**작성일**: 2025년 12월 29일  
**버전**: v1.0  
**문서 유형**: 폴더 구조 시각화
