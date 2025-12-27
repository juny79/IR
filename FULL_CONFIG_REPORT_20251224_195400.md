# Full Config Report (20251224_195400)

## Secrets (presence only)

- GEMINI_API_KEY: set
- UPSTAGE_API_KEY: set
- ES_PASSWORD: set
- OPENAI_API_KEY: set

## Effective Model IDs

- Gemini LLM effective: `models/gemini-3-flash-preview` (GEMINI_MODEL_ID env: <unset>)
- Solar chat effective: `solar-pro` (SOLAR_MODEL_ID env: <unset>)
- EmbeddingClient import: `ok`

## Embedding Models (as coded)

- SBERT: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- Gemini embedding: `models/text-embedding-004`
- Upstage embedding (query): `solar-embedding-1-large-query`
- Upstage embedding (passage): `solar-embedding-1-large-passage`

## Runtime Toggles / Hyperparameters

- USE_RRF: <unset>
- RRF_K: <unset>
- TOP_K_RETRIEVE: <unset>
- CANDIDATE_POOL_SIZE: <unset>
- USE_MULTI_EMBEDDING: <unset>
- USE_GEMINI_ONLY: <unset>
- VOTING_WEIGHTS_JSON: <unset>
- DENSE_EMBEDDING_FIELDS: <unset>
- DENSE_K_PER_FIELD: <unset>
- DENSE_K_PER_FIELD_MAP: <unset>
- USE_SOLAR_ANALYZER: <unset>
- USE_GATING: <unset>
- NO_SEARCH_CONFIDENCE_THRESHOLD: <unset>
- ENABLE_INTENT_NO_SEARCH: <unset>
- NO_SEARCH_STRICT_THRESHOLD: <unset>
- NO_SEARCH_HEURISTIC_THRESHOLD: <unset>
- NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF: <unset>
- HYDE_MAX_LENGTH: <unset>
- USE_MULTI_QUERY: <unset>
- MULTI_QUERY_COUNT: <unset>
- GEMINI_MODEL_ID: <unset>
- SOLAR_MODEL_ID: <unset>

## Effective Defaults (from eval_rag.py)

- VOTING_WEIGHTS: `[5, 4, 2]`
- USE_MULTI_EMBEDDING: `True`
- USE_GEMINI_ONLY: `False`
- TOP_K_RETRIEVE: `80`
- USE_RRF: `False`
- RRF_K: `60`
- USE_GATING: `False`
- HYDE_MAX_LENGTH: `200`
- USE_SOLAR_ANALYZER: `True`
- USE_MULTI_QUERY: `False`
- NO_SEARCH_CONFIDENCE_THRESHOLD: `0.0`
- CANDIDATE_POOL_SIZE: `80`
- ENABLE_INTENT_NO_SEARCH: `False`
- NO_SEARCH_STRICT_THRESHOLD: `0.0`
- NO_SEARCH_HEURISTIC_THRESHOLD: `0.85`
- NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF: `0.1`

## Health Checks

- RUN_HEALTH_CHECKS: `true`
- SBERT query: ✅ dim=768 l2_norm=18.6174 all_zero=False
- Gemini embedding retrieval_query: ✅ dim=768 l2_norm=1.0000 all_zero=False
- Gemini embedding retrieval_document: ✅ dim=768 l2_norm=1.0000 all_zero=False
- Upstage embedding query: ✅ dim=4096 l2_norm=1.0000 all_zero=False
- Upstage embedding passage: ✅ dim=4096 l2_norm=1.0002 all_zero=False
- Solar chat ping (HyDE): ✅ response_chars=200
- Gemini chat ping: ✅ response_chars=35

## Notes

- 이 리포트는 설정/모델 ID 확인용이며, API 호출 결과(성능/점수)를 포함하지 않습니다.
- 대회 규칙상 topk=[] 정답 케이스(21개)가 존재하므로, 게이팅 변경은 ‘추가로 비울 케이스’가 과학 질문을 침범하지 않게 보수적으로 진행해야 합니다.
