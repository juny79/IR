import json
from retrieval.es_connector import es
from models.embedding_client import embedding_client
from collections import defaultdict
import operator
import math

# (es_connector.py에 sparse_retrieve, dense_retrieve 함수가 정의되어 있다고 가정)

# Phase 5-2: RRF (Reciprocal Rank Fusion) 알고리즘
def rrf_fusion(sparse_res, dense_res_list, top_k=20, k=60):
    """
    RRF (Reciprocal Rank Fusion) 알고리즘
    
    score(doc) = Σ(1 / (k + rank_i))
    
    Args:
        sparse_res: BM25 검색 결과
        dense_res_list: Dense 검색 결과 리스트
        top_k: 반환할 문서 개수
        k: RRF 파라미터 (기본값 60, 실험 범위: 1-100)
    
    Returns:
        상위 top_k 문서 리스트
    """
    rrf_scores = defaultdict(float)
    
    # Sparse 결과 처리
    for rank, hit in enumerate(sparse_res['hits']['hits'], 1):
        docid = hit['_source']['docid']
        rrf_scores[docid] += 1.0 / (k + rank)
    
    # Dense 결과 처리
    for dense_res in dense_res_list:
        for rank, hit in enumerate(dense_res['hits']['hits'], 1):
            docid = hit['_source']['docid']
            rrf_scores[docid] += 1.0 / (k + rank)
    
    # 점수 기준 정렬
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Top-K 반환
    return [doc_id for doc_id, score in sorted_docs[:top_k]]

# MAP 최고점 전략의 핵심: Hard Voting 앙상블 함수
def hard_vote_results(sparse_res, dense_res_list, top_k=5, weights=[5, 3, 1]):
    """
    Sparse 및 Dense 검색 결과를 취합하여 Hard Voting (5:3:1 가중치)으로 최종 순위를 결정합니다.
    """
    all_hits = defaultdict(list)
    
    # 1. 모든 검색 결과의 docid와 순위(rank)를 추출
    
    # Sparse 결과 처리 (Rank 1 순위 기준)
    for i, hit in enumerate(sparse_res['hits']['hits']):
        docid = hit['_source']['docid']
        all_hits[docid].append({'source': 'sparse', 'rank': i + 1, 'score': hit['_score']})

    # Dense 결과 처리
    for dense_res in dense_res_list:
        for i, hit in enumerate(dense_res['hits']['hits']):
            docid = hit['_source']['docid']
            all_hits[docid].append({'source': 'dense', 'rank': i + 1, 'score': hit['_score']})
    
    # 2. Hard Voting 스코어 계산 (5:3:1 가중치 적용)
    final_scores = {}
    
    for docid, hits in all_hits.items():
        score = 0
        for hit in hits:
            # MAP 극대화를 위한 Hard Voting 가중치
            if hit['rank'] == 1:
                score += weights[0]
            elif hit['rank'] == 2 and len(weights) > 1:
                score += weights[1]
            elif hit['rank'] == 3 and len(weights) > 2:
                score += weights[2]
            else:
                # 4위 이하는 작은 가중치 (0.5)
                score += 0.5
            
        final_scores[docid] = score
        
    # 3. 최종 점수 기준으로 정렬 및 Top-K 선택
    sorted_docids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
    top_docids = sorted_docids[:top_k]
    
    # 4. 최종 Top-K 문서 정보 반환
    final_topk_results = []
    for docid in top_docids:
        final_topk_results.append({
            "docid": docid,
            "score": final_scores[docid]
        })
        
    return final_topk_results

def get_document_content(docid):
    """
    Elasticsearch에서 docid로 문서 내용 가져오기 (캐싱 가능하도록 최적화)
    """
    try:
        result = es.search(
            index="test",
            query={"term": {"docid": docid}},
            size=1,
            _source=["content"]  # content만 가져오기
        )
        if result['hits']['hits']:
            content = result['hits']['hits'][0]['_source'].get('content', '')
            return content[:1000]  # 첫 1000자만 사용 (속도 향상)
        return ""
    except:
        return ""

def get_documents_batch(docids):
    """
    여러 문서를 한번에 가져오기 (배치 처리)
    """
    try:
        result = es.search(
            index="test",
            query={"terms": {"docid": docids}},
            size=len(docids),
            _source=["docid", "content"]
        )
        
        # docid를 키로 하는 딕셔너리 생성
        docs_dict = {}
        for hit in result['hits']['hits']:
            doc_id = hit['_source']['docid']
            content = hit['_source'].get('content', '')[:1000]
            docs_dict[doc_id] = content
        
        return docs_dict
    except:
        return {}

def run_hybrid_search(original_query, sparse_query=None, reranker_query=None, top_k_retrieve=50, top_k_final=5, voting_weights=None, use_multi_embedding=False, use_gemini_only=False, use_rrf=False, rrf_k=60, multi_queries=None, candidate_pool_size=50):
    """
    Hybrid Search with Reranker (Phase 7: 멀티 쿼리 지원)
    
    Args:
        original_query: 원본 검색 질문 (Reranker에 사용)
        sparse_query: Sparse Search용 쿼리 (HyDE 확장, None이면 original_query 사용)
        reranker_query: Reranker용 쿼리 (None이면 original_query 사용)
        top_k_retrieve: 초기 검색에서 가져올 문서 개수 (넓게 검색)
        top_k_final: 최종 반환할 문서 개수
        voting_weights: Hard Voting 가중치 (기본: [6, 3, 1])
        use_multi_embedding: 다중 임베딩 사용 여부 (Phase 3-2, 3-3)
        use_gemini_only: Gemini embedding만 사용 (SBERT 비활성화, Phase 4A)
        use_rrf: RRF 알고리즘 사용 여부 (Phase 5-2)
        rrf_k: RRF 파라미터 (기본 60)
        multi_queries: ⭐ Phase 7D: 멀티 쿼리 리스트 (추가 Sparse 검색용)
    """
    from retrieval.es_connector import sparse_retrieve, dense_retrieve
    
    # 기본값 설정
    if voting_weights is None:
        voting_weights = [6, 3, 1]  # Phase 2 최적값
    if sparse_query is None:
        sparse_query = original_query
    if reranker_query is None:
        reranker_query = original_query
    
    # Step 1: Sparse + Dense 검색 (넓게 검색 - Top 50)
    # Sparse: HyDE 확장 쿼리 사용 (키워드 풍부화)
    sparse_res = sparse_retrieve(sparse_query, top_k_retrieve)
    
    # ⭐ Phase 7D: 멀티 쿼리로 추가 Sparse 검색 (재현율 향상)
    additional_sparse_results = []
    if multi_queries:
        for mq in multi_queries:
            try:
                additional_res = sparse_retrieve(mq, top_k_retrieve // 2)  # 각 쿼리당 절반
                additional_sparse_results.append(additional_res)
            except Exception as e:
                pass  # 실패 시 무시
    
    # Dense 검색 결과 리스트
    dense_results = []
    
    # Phase 4A: Gemini embedding만 사용 (SBERT 비활성화)
    if use_gemini_only:
        # Gemini embedding만 사용
        try:
            dense_res_gemini = dense_retrieve(sparse_query, top_k_retrieve, "embeddings_gemini")
            dense_results.append(dense_res_gemini)
        except Exception as e:
            print(f"⚠️ Gemini 임베딩 검색 실패: {e}")
            # fallback: SBERT 사용
            dense_res_sbert = dense_retrieve(sparse_query, top_k_retrieve, "embeddings_sbert")
            dense_results.append(dense_res_sbert)
    else:
        # Dense: SBERT (기본)
        dense_res_sbert = dense_retrieve(sparse_query, top_k_retrieve, "embeddings_sbert")
        dense_results.append(dense_res_sbert)
    
        # Phase 8: Gemini 임베딩 추가 (선택적, Upstage 제외)
        if use_multi_embedding:
            try:
                dense_res_gemini = dense_retrieve(sparse_query, top_k_retrieve, "embeddings_gemini")
                dense_results.append(dense_res_gemini)
            except Exception as e:
                print(f"⚠️ Gemini 임베딩 검색 실패: {e}")
    
    # Step 2: Fusion 알고리즘 선택 (Hard Voting 또는 RRF)
    # ⭐ Phase 7D: 멀티 쿼리 결과도 Sparse에 포함
    all_sparse_results = [sparse_res] + additional_sparse_results
    
    # 후보군 크기 정규화
    try:
        candidate_pool_size = int(candidate_pool_size)
    except Exception:
        candidate_pool_size = 50
    if candidate_pool_size <= 0:
        candidate_pool_size = 50

    if use_rrf:
        # Phase 5-2: RRF 알고리즘 사용
        # 멀티 쿼리의 Sparse 결과를 모두 합산
        combined_sparse = sparse_res
        for add_res in additional_sparse_results:
            # RRF에서는 모든 결과를 통합
            pass  # RRF 함수 내부에서 처리
        
        candidates = rrf_fusion(
            sparse_res,
            dense_results,
            top_k=candidate_pool_size,
            k=rrf_k
        )
    else:
        # 기존: Hard Voting 사용
        # ⭐ Phase 7D: 멀티 쿼리 Sparse 결과도 투표에 포함
        combined_dense = dense_results.copy()
        for add_res in additional_sparse_results:
            combined_dense.append(add_res)  # 추가 Sparse 결과를 Dense처럼 취급
        
        candidates_with_scores = hard_vote_results(
            sparse_res, 
            combined_dense,  # 멀티 쿼리 결과 포함
            top_k=candidate_pool_size,
            weights=voting_weights
        )
        # docid만 추출
        candidates = [item['docid'] for item in candidates_with_scores]
    
    # Step 3: 배치로 문서 내용 가져오기 (최적화) ⭐
    docs_dict = get_documents_batch(candidates)
    
    # (docid, content) 형태로 변환
    docs_with_content = [(doc_id, docs_dict.get(doc_id, '')) 
                         for doc_id in candidates if docs_dict.get(doc_id)]
    
    # Step 4: Reranker로 Top-K 최종 선정
    from retrieval.reranker import reranker
    
    # Reranker 쿼리 결정 (None이면 original_query 사용)
    if reranker_query is None:
        reranker_query = original_query
    
    if docs_with_content:
        final_ranked_results = reranker.rerank(
            reranker_query,  # 원본 쿼리 사용 (Phase 2 확정)
            docs_with_content, 
            top_k=top_k_final,
            batch_size=32  # 배치 크기 지정
        )
    else:
        final_ranked_results = candidates[:top_k_final]
    
    return final_ranked_results