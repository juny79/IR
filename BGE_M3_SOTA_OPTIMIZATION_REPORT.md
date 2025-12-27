# BAAI/BGE-M3 기반 검색 파이프라인 최적화 종합 보고서

## 1. 개요 (Overview)
본 프로젝트는 과학 상식 질의응답 시스템의 성능을 극대화하기 위해 `BAAI/bge-m3` 임베딩 모델과 `BAAI/bge-reranker-v2-m3` 교차 인코더(Cross-Encoder)를 중심으로 한 새로운 검색 파이프라인을 구축하고 최적화하는 과정을 담고 있습니다.

- **핵심 목표**: 리더보드 MAP(Mean Average Precision) 점수 0.94 돌파 및 안정적인 검색 성능 확보
- **주요 도구**: BGE-M3 (Embedding), BGE-Reranker-v2-m3 (Reranker), Gemini API (HyDE & Reranking), Elasticsearch (Vector DB)

---

## 2. 단계별 실험 과정 및 전략

### Phase 1: BGE-M3 기반 파이프라인 구축 (Foundation)
기존의 임베딩 모델을 `BAAI/bge-m3`로 교체하고, 검색 후보군을 정교하게 재정렬하기 위한 Cross-Encoder를 도입했습니다.
- **전략**: Dense Retrieval (BGE-M3) + Sparse Retrieval (BM25) 하이브리드 검색
- **최적화**: `TOP_K_RETRIEVE`를 100으로 확대하여 Reranker가 정답을 발견할 확률(Recall)을 극대화함.

### Phase 2: 검색 고도화 및 SOTA 달성 (The Milestone)
단순 검색을 넘어 쿼리의 품질을 높이고 순위 결합 알고리즘을 정교화했습니다.
- **Multi-Query & HyDE**: Gemini를 활용해 질문의 의도를 다각화하고 가설 답변(HyDE)을 생성하여 검색 정확도를 높임.
- **RRF (Reciprocal Rank Fusion)**: 여러 검색 결과의 순위를 효과적으로 결합.
- **결과 (v9)**: **MAP 0.9409 / MRR 0.9424** 달성 (현재 프로젝트의 최고점 SOTA).

### Phase 3: LLM 기반 Reranking 실험 (v16)
Cross-Encoder의 한계를 극복하기 위해 Gemini API를 직접 Reranker로 사용하는 실험을 진행했습니다.
- **전략**: Top 5 후보군에 대해 Gemini가 직접 관련성을 평가하여 순위 재조정.
- **이슈 해결**: Gemini API의 `finish_reason: 2` (Safety/Empty)로 인한 무한 재시도 현상(Retry Storm)을 감지하고, 즉시 Fallback 로직을 적용하여 파이프라인 안정성 확보.
- **결과**: MAP 0.931~0.935 예측 (v9 대비 변동성이 커서 단독 제출로는 위험군으로 분류).

### Phase 4: 보수적 스왑 전략 (v17)
v9의 안정성을 유지하면서 v16(Gemini)의 통찰력을 부분적으로 수용하는 전략을 취했습니다.
- **전략**: Cross-Encoder의 점수 차이(Margin)가 큰 경우에만 v9의 Top1을 v16의 결과로 교체.
- **실행**: `safe3` (2개 교체), `attack5` (5개 교체) 변형 생성.
- **결과**: 리더보드 점수 변동 없음 (교체된 문서들이 정답 순위에 영향을 주지 못함).

### Phase 5: 최종 Union-Rerank (v18)
모든 성공적인 실험 결과물들을 하나로 모으는 최종 통합 전략을 실행했습니다.
- **전략**: v9, v12, v15, v16, BGE-M3 v7 등 주요 제출 파일의 Top 10 후보군을 모두 합친(Union) 후, Cross-Encoder로 최종 재정렬.
- **목적**: 각 모델이 놓친 정답을 합집합 내에서 다시 찾아내어 순위를 최적화.
- **결과**: **MAP 0.9348 / MRR 0.9364**.

---

## 3. 주요 지표 변화 (Metrics Summary)

| 실험 단계 | 주요 전략 | MAP | MRR | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 4D** | 초기 베이스라인 | 0.8424 | - | 기준점 |
| **v9 (SOTA)** | **BGE-M3 + RRF + HyDE** | **0.9409** | **0.9424** | **최고 성능** |
| **v16** | Gemini Reranking | 0.931* | - | 예측치, 변동성 높음 |
| **v17 (Safe)** | Margin-based Swap | 0.9409 | 0.9424 | v9과 동일 점수 |
| **v18 (Final)** | **Union + Rerank** | **0.9348** | **0.9364** | **최종 통합본** |

---

## 4. 결론 및 시사점

1.  **BGE-M3의 강력한 성능**: `BAAI/bge-m3`와 `bge-reranker-v2-m3`의 조합은 0.94 이상의 높은 MAP를 달성하는 데 핵심적인 역할을 했습니다.
2.  **앙상블의 한계 효용**: 이미 0.94 수준에 도달한 SOTA 모델에서 추가적인 Union-Rerank나 LLM 교체는 정답 순위를 뒤흔드는 노이즈로 작용할 가능성이 큼을 확인했습니다.
3.  **안정성 확보**: Gemini API 연동 시 발생할 수 있는 예외 상황(Retry Storm)에 대한 방어 로직을 구축하여, 향후 어떤 LLM을 도입하더라도 견고한 파이프라인을 유지할 수 있게 되었습니다.

**최종 권고**: 현재의 `v9` (MAP 0.9409) 세팅이 가장 신뢰할 수 있는 최적의 조합이며, 향후 성능 향상을 위해서는 데이터 증강(Data Augmentation)이나 도메인 특화 미세 조정(Fine-tuning) 단계로의 전환이 필요합니다.
