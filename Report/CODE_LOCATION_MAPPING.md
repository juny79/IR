# IR 시스템 - 파일별 코드 위치 및 주요 함수 맵핑

## 📍 각 파일의 주요 함수 위치

### 1️⃣ main.py (~70줄)

**파일 위치**: `/root/IR/main.py`

```python
# 주요 로직
for i, line in enumerate(f, 1):
    data = json.loads(line)
    if data["eval_id"] in processed_ids:
        continue
    print(f"[{i}/220] ID {data['eval_id']} 처리 중...")
    result = answer_question_optimized(data["msg"])  # ← 핵심
    output = {"eval_id": data["eval_id"], "standalone_query": result["standalone_query"], ...}
    of.write(json.dumps(output, ensure_ascii=False) + "\n")
```

**함수**: `main()`  
**입출력**:
- 입력: `data/eval.jsonl` (220개 질문)
- 출력: `submission.csv` (실시간 저장)

**역할**: 평가 루프 전체 관리, 중단/재시작 안전

---

### 2️⃣ eval_rag.py (~70줄)

**파일 위치**: `/root/IR/eval_rag.py`

**🔴 설정값 (가장 자주 수정)**:
```python
# 라인 1-15
VOTING_WEIGHTS = [5, 4, 2]
USE_MULTI_EMBEDDING = True
USE_GEMINI_ONLY = False
TOP_K_RETRIEVE = 60
USE_RRF = False
RRF_K = 60
USE_GATING = True
```

**주요 함수** `answer_question_optimized(messages)`:
```python
# 라인 17-70
def answer_question_optimized(messages):
    res = {"standalone_query": "", "topk": [], "answer": ""}
    
    # Step 1: 쿼리 분석
    analysis = llm_client.analyze_query(messages)
    
    if analysis.tool_calls:
        # Step 2: 쿼리 정제
        query = json.loads(analysis.tool_calls[0].function.arguments)['standalone_query']
        res["standalone_query"] = query
        
        # Step 3: HyDE 확장
        hypothetical_answer = solar_client.generate_hypothetical_answer(query)
        hyde_query = f"{query}\n{hypothetical_answer}" if hypothetical_answer else query
        
        # Step 4: 하이브리드 검색
        final_ranked_results = run_hybrid_search(
            original_query=query,
            sparse_query=hyde_query,
            reranker_query=query,
            voting_weights=VOTING_WEIGHTS,
            use_multi_embedding=USE_MULTI_EMBEDDING,
            top_k_retrieve=TOP_K_RETRIEVE,
            use_gemini_only=USE_GEMINI_ONLY,
            use_rrf=USE_RRF,
            rrf_k=RRF_K
        )
        
        # Step 5: 문서 내용 조회
        res["topk"] = final_ranked_results[:5]
        context_docs = []
        for docid in final_ranked_results[:3]:
            search_result = es.search(index="test", query={"term": {"docid": docid}}, size=1)
            if search_result['hits']['hits']:
                context_docs.append(search_result['hits']['hits'][0]['_source']['content'])
        context = " ".join(context_docs)
        
        # Step 6: 답변 생성
        res["answer"] = solar_client.generate_answer(messages, context)
    else:
        # 비과학 질문 처리 (게이팅)
        res["standalone_query"] = ""
        res["topk"] = []
        res["answer"] = analysis.content
    
    return res
```

**핵심**: 이 파일의 설정값을 바꾸면 전체 동작이 변경됨

---

### 3️⃣ models/llm_client.py

**파일 위치**: `/root/IR/models/llm_client.py`

**클래스**: `LLMClient`

**주요 함수**:
```python
def analyze_query(messages):
    """
    Gemini API로 쿼리 분석
    
    [반환값]
    - tool_calls 있음: 과학 질문
      {
        "tool_calls": [{...}],
        "content": None
      }
    - tool_calls 없음: 비과학 질문
      {
        "tool_calls": None,
        "content": "안녕하세요!"
      }
    """
    # Gemini API 호출
    response = genai.GenerativeModel(...).generate_content(
        content=messages,
        tools=[...],
        tool_config=...
    )
    return response
```

**특징**: 캐싱 없음 (매번 새 API 호출)

---

### 4️⃣ models/solar_client.py

**파일 위치**: `/root/IR/models/solar_client.py`

**클래스**: `SolarClient`

**주요 함수 1**: `generate_hypothetical_answer(query)`
```python
def generate_hypothetical_answer(self, query):
    """
    Solar Pro 2 HyDE 쿼리 확장
    
    [캐싱]
    1. cache/hyde_cache.pkl 확인
    2. 캐시 미스 → Upstage API 호출
    3. 캐시 저장 (20개마다)
    
    [반환]
    가설적 답변 문자열
    """
    # MD5 캐시 키 생성
    cache_key = hashlib.md5(query.encode()).hexdigest()
    
    # 캐시 조회
    if cache_key in self.hyde_cache:
        return self.hyde_cache[cache_key]
    
    # API 호출
    response = self.client.messages.create(
        model="solar-pro",
        messages=[{"role": "user", "content": f"쿼리 확장: {query}"}]
    )
    
    # 캐시 저장
    self.hyde_cache[cache_key] = response.content
    return response.content
```

**주요 함수 2**: `generate_answer(messages, context)`
```python
def generate_answer(self, messages, context):
    """
    최종 답변 생성
    
    [입력]
    - messages: 사용자 메시지
    - context: 검색된 문서 내용
    
    [반환]
    생성된 답변 텍스트
    """
    # Solar Pro 2 API로 답변 생성
    system_prompt = f"다음 문서를 참고하여 질문에 답하세요:\n{context}"
    response = self.client.messages.create(...)
    return response.content
```

**특징**:
- ✅ HyDE 생성은 캐싱 (pickle)
- ❌ 답변 생성은 캐싱 없음

---

### 5️⃣ models/embedding_client.py

**파일 위치**: `/root/IR/models/embedding_client.py`

**클래스**: `EmbeddingClient`

**주요 함수**: `get_query_embedding(query, use_gemini_only=False)`
```python
def get_query_embedding(self, query, use_gemini_only=False):
    """
    쿼리 임베딩 생성
    
    [경우 1] use_gemini_only=False (기본)
    - SBERT로 로컬 임베딩
    - 빠름 (~100ms)
    - 캐싱 없음
    
    [경우 2] use_gemini_only=True
    - Gemini API 호출
    - 캐싱 적용 (34,893배 속도)
    - MD5 해싱으로 캐시 키 생성
    
    [반환]
    768차원 벡터
    """
    
    if use_gemini_only:
        # Gemini 캐싱 로직
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.query_embedding_cache:
            return self.query_embedding_cache[cache_key]
        
        # API 호출
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=query
        )
        embedding = response['embedding']
        
        # 캐시 저장
        self.query_embedding_cache[cache_key] = embedding
        return embedding
    else:
        # SBERT 로컬 임베딩
        return self.sbert_model.encode(query)
```

**두 임베딩 모델**:
1. SBERT: `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (로컬, 빠름)
2. Gemini: `text-embedding-004` (API, 캐싱)

---

### 6️⃣ retrieval/hybrid_search.py

**파일 위치**: `/root/IR/retrieval/hybrid_search.py`

**주요 함수**: `run_hybrid_search(...)`
```python
def run_hybrid_search(
    original_query,
    sparse_query,
    reranker_query,
    voting_weights=[5, 4, 2],
    use_multi_embedding=True,
    top_k_retrieve=50,
    use_gemini_only=False,
    use_rrf=False,
    rrf_k=60
):
    """
    하이브리드 검색 + 재순위화
    
    [처리 순서]
    1. Sparse 검색 (BM25)
    2. Dense 검색 (SBERT)
    3. Dense 검색 (Gemini)
    4. 점수 합산 (Hard Voting)
    5. Reranker 적용
    """
    
    # Step 1: Sparse 검색
    sparse_results = es.search_sparse(sparse_query, top_k=top_k_retrieve)
    
    # Step 2-3: Dense 검색
    query_emb_sbert = embedding_client.get_query_embedding(original_query, use_gemini_only=False)
    dense_results_sbert = es.search_dense(query_emb_sbert, top_k=top_k_retrieve)
    
    query_emb_gemini = embedding_client.get_query_embedding(original_query, use_gemini_only=True)
    dense_results_gemini = es.search_dense(query_emb_gemini, top_k=top_k_retrieve)
    
    # Step 4: Hard Voting 또는 RRF
    if use_rrf:
        final_results = rrf_fusion([sparse_results, dense_results_sbert, dense_results_gemini], k=rrf_k)
    else:
        # Hard Voting [5, 4, 2]
        final_results = hard_vote_results([
            sparse_results,
            dense_results_sbert,
            dense_results_gemini
        ], voting_weights=voting_weights)
    
    # Step 5: Reranker
    final_ranked = reranker.rerank_documents(reranker_query, final_results[:top_k_retrieve])
    
    return [doc_id for doc_id, _ in final_ranked]
```

**Hard Voting 함수** `hard_vote_results()`
```python
def hard_vote_results(search_results_list, voting_weights):
    """
    여러 검색 결과를 가중치로 투표
    
    [로직]
    각 문서별로:
    - Sparse에서 1위면: score += voting_weights[0] * (1 - rank/100)
    - SBERT에서 2위면: score += voting_weights[1] * (1 - rank/100)
    - Gemini에서 3위면: score += voting_weights[2] * (1 - rank/100)
    
    점수로 재정렬 → Top K 선정
    """
    vote_scores = defaultdict(float)
    
    for idx, results in enumerate(search_results_list):
        weight = voting_weights[idx]
        for rank, (doc_id, score) in enumerate(results, 1):
            vote_scores[doc_id] += weight * (1 - rank / 100)
    
    sorted_results = sorted(vote_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_results]
```

**핵심**: [5, 4, 2] 가중치로 Sparse, SBERT, Gemini 결과 융합

---

### 7️⃣ retrieval/es_connector.py

**파일 위치**: `/root/IR/retrieval/es_connector.py`

**클래스**: `ESConnector`

**주요 함수 1**: `search_sparse(query, top_k=50)`
```python
def search_sparse(self, query, top_k=50):
    """
    BM25 알고리즘으로 sparse 검색
    
    [쿼리]
    Solar HyDE 확장 쿼리
    
    [반환]
    [(doc_id, bm25_score), ...]
    """
    response = self.es.search(
        index="test",
        query={
            "match": {
                "content": {
                    "query": query,
                    "operator": "or"
                }
            }
        },
        size=top_k
    )
    
    results = []
    for hit in response['hits']['hits']:
        results.append((hit['_source']['docid'], hit['_score']))
    
    return results
```

**주요 함수 2**: `search_dense(embedding, top_k=50)`
```python
def search_dense(self, embedding, top_k=50):
    """
    Dense 검색 (임베딩 기반)
    
    [입력]
    embedding: 768차원 벡터 (SBERT 또는 Gemini)
    
    [반환]
    [(doc_id, similarity_score), ...]
    """
    response = self.es.search(
        index="test",
        query={
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embeddings_field') + 1.0",
                    "params": {"query_vector": embedding}
                }
            }
        },
        size=top_k
    )
    
    results = []
    for hit in response['hits']['hits']:
        results.append((hit['_source']['docid'], hit['_score']))
    
    return results
```

**주요 함수 3**: `get_document(doc_id)`
```python
def get_document(self, doc_id):
    """
    특정 문서의 내용 조회
    
    [반환]
    문서 객체 (docid, content, metadata 등)
    """
    response = self.es.search(
        index="test",
        query={"term": {"docid": doc_id}},
        size=1
    )
    
    if response['hits']['hits']:
        return response['hits']['hits'][0]['_source']
    return None
```

---

### 8️⃣ retrieval/reranker.py

**파일 위치**: `/root/IR/retrieval/reranker.py`

**클래스**: `Reranker`

**주요 함수**: `rerank_documents(query, documents)`
```python
def rerank_documents(self, query, document_ids, top_k=None):
    """
    BAAI Reranker로 최종 순위 조정
    
    [입력]
    - query: 원본 쿼리
    - document_ids: 재순위화할 문서 ID 리스트
    
    [처리]
    각 (query, document) 쌍을 Reranker에 입력
    관련성 점수 (0-1) 계산
    
    [반환]
    재정렬된 [(doc_id, score), ...] 리스트
    """
    
    # 문서 내용 조회
    documents = []
    for doc_id in document_ids:
        doc = es.get_document(doc_id)
        documents.append(doc['content'])
    
    # Reranker 입력: (query, document) 쌍 리스트
    pairs = [[query, doc] for doc in documents]
    
    # BAAI Reranker 실행
    scores = self.model.predict(pairs)
    
    # 점수로 정렬
    reranked = sorted(zip(document_ids, scores), key=lambda x: x[1], reverse=True)
    
    if top_k:
        return reranked[:top_k]
    return reranked
```

---

## 📊 함수 호출 체인

```
main()
  └─ answer_question_optimized()
      ├─ llm_client.analyze_query()
      ├─ solar_client.generate_hypothetical_answer()  ✅ 캐싱
      ├─ run_hybrid_search()
      │   ├─ es_connector.search_sparse()
      │   ├─ embedding_client.get_query_embedding()  (SBERT)
      │   ├─ embedding_client.get_query_embedding()  ✅ 캐싱 (Gemini)
      │   ├─ hard_vote_results()
      │   └─ reranker.rerank_documents()
      ├─ es_connector.get_document()  (3회)
      └─ solar_client.generate_answer()
```

---

## 🔧 설정 변경 가이드

| 변경 목표 | 수정 파일 | 설정값 |
|----------|----------|--------|
| 가중치 조정 | eval_rag.py | VOTING_WEIGHTS |
| TOP_K 변경 | eval_rag.py | TOP_K_RETRIEVE |
| SBERT만 사용 | eval_rag.py | USE_MULTI_EMBEDDING=False |
| RRF 사용 | eval_rag.py | USE_RRF=True |
| 게이팅 OFF | eval_rag.py | USE_GATING=False |
| Gemini만 사용 | eval_rag.py | USE_GEMINI_ONLY=True |

---

**가장 중요한 파일**: eval_rag.py (설정) + hybrid_search.py (검색 로직)  
**캐싱 적용 모듈**: solar_client (HyDE), embedding_client (Gemini)  
**성능 조정 포인트**: VOTING_WEIGHTS, TOP_K, 멀티 임베딩
