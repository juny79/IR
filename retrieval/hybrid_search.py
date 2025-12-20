import json
from retrieval.es_connector import es
from models.embedding_client import embedding_client
from collections import defaultdict
import operator

# (es_connector.py에 sparse_retrieve, dense_retrieve 함수가 정의되어 있다고 가정)

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

def run_hybrid_search(original_query, sparse_query=None, reranker_query=None, top_k_retrieve=50, top_k_final=5, voting_weights=None):
    """
    Hybrid Search with Reranker (Phase 2: HyDE 전체 적용)
    
    Args:
        original_query: 원본 검색 질문 (Reranker에 사용)
        sparse_query: Sparse Search용 쿼리 (HyDE 확장, None이면 original_query 사용)
        reranker_query: Reranker용 쿼리 (None이면 original_query 사용)
        top_k_retrieve: 초기 검색에서 가져올 문서 개수 (넓게 검색)
        top_k_final: 최종 반환할 문서 개수
        voting_weights: Hard Voting 가중치 (기본: [5, 3, 1])
    """
    from retrieval.es_connector import sparse_retrieve, dense_retrieve
    
    # 기본값 설정
    if voting_weights is None:
        voting_weights = [5, 3, 1]
    if sparse_query is None:
        sparse_query = original_query
    if reranker_query is None:
        reranker_query = original_query
    
    # Step 1: Sparse + Dense 검색 (넓게 검색 - Top 50)
    # Sparse: HyDE 확장 쿼리 사용 (키워드 풍부화)
    sparse_res = sparse_retrieve(sparse_query, top_k_retrieve)
    
    # Dense: HyDE 확장 쿼리 사용 (Phase 2)
    dense_res = dense_retrieve(sparse_query, top_k_retrieve, "embeddings_sbert")
    
    # Step 2: Hard Voting으로 Top 20 후보 추출
    candidates_with_scores = hard_vote_results(
        sparse_res, 
        [dense_res], 
        top_k=20,  # Reranker에 넘길 후보 개수
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
            reranker_query,  # ⭐ HyDE 쿼리로 Sparse와 일관성 확보
            docs_with_content, 
            top_k=top_k_final,
            batch_size=32  # 배치 크기 지정
        )
    else:
        final_ranked_results = candidates[:top_k_final]
    
    return final_ranked_results