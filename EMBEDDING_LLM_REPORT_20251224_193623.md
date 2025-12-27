# Embedding/LLM Config Report (20251224_193623)

## Environment

- GEMINI_API_KEY: set
- UPSTAGE_API_KEY: set
- ES_PASSWORD: set

## Chat/Analysis LLM Models

- Gemini LLM effective: `gemini-3.0-flash-preview` (GEMINI_MODEL_ID env: <unset>)
- Solar chat effective: `solar-pro` (SOLAR_MODEL_ID env: <unset>)

## Embedding Models (Configured in Code)

- Gemini embedding model: `models/text-embedding-004`
- Upstage query embedding model: `solar-embedding-1-large-query`
- Upstage passage embedding model: `solar-embedding-1-large-passage`

## Health Checks

- SBERT (sentence_transformers) query: ✅ dim=768 l2_norm=18.6174 all_zero=False
- Gemini embedding retrieval_query: ✅ dim=768 l2_norm=1.0000 all_zero=False
- Gemini embedding retrieval_document: ✅ dim=768 l2_norm=1.0000 all_zero=False
- Upstage embedding query: ✅ dim=4096 l2_norm=1.0000 all_zero=False
- Upstage embedding passage: ✅ dim=4096 l2_norm=1.0002 all_zero=False

## Notes

- ‘최신 LLM’과 ‘임베딩 모델’은 별개입니다. 이 프로젝트에서 임베딩은 Gemini `text-embedding-004` / Upstage `solar-embedding-1-large-*`로 호출됩니다.
- Gemini/Upstage API 호출 실패 시 코드가 zero-vector로 fallback 하므로, 위 health check에서 all_zero=true가 나오면 ‘모델/키/호출 실패’를 의미할 가능성이 큽니다.
