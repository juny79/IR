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
            # Hard Voting을 위해 중복된 docid는 점수가 합산될 수 있도록 list에 추가
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
            
        final_scores[docid] = score
        
    # 3. 최종 점수 기준으로 정렬 및 Top-K 선택
    # docid를 기준으로 정렬하고, 최종 Top-K를 선정합니다.
    sorted_docids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
    top_docids = sorted_docids[:top_k]
    
    # 4. 최종 Top-K 문서 정보를 Elasticsearch에서 조회
    # (실제 구현 시, ES의 mget 또는 filter query를 사용하여 문서 내용(content)을 가져와야 함)
    # 여기서는 docid와 final_score만 반환 (실제 content 조회 로직 필요)
    
    final_topk_results = []
    # (가져온 content와 docid를 매핑하여 final_topk_results 리스트 구성)
    # 임시 코드:
    for docid in top_docids:
        # ES에서 docid로 문서 content를 가져오는 함수 호출 가정
        # doc_content = get_content_by_docid(docid) 
        doc_content = f"문서 내용: {docid}"
        final_topk_results.append({
            "docid": docid,
            "score": final_scores[docid],
            "content": doc_content
        })
        
    return final_topk_results

def run_hybrid_search(query):
    # 2-1. 하이브리드 검색 실행 (넉넉한 Top-K, 예: 10개)
    top_k_retrieve = 10
    
    # Sparse Retrieval (BM25)
    from retrieval.es_connector import sparse_retrieve
    sparse_res = sparse_retrieve(query, top_k_retrieve)
    
    # Dense Retrieval (다중 백엔드)
    from retrieval.es_connector import dense_retrieve
    dense_res_list = [
        dense_retrieve(query, top_k_retrieve, "embeddings_sbert"),
        # dense_retrieve(query, top_k_retrieve, "embeddings_solar"), # 주석 해제 후 사용
        # dense_retrieve(query, top_k_retrieve, "embeddings_gemini")  # 주석 해제 후 사용
    ]
    
    # 2-2. Hard Voting 앙상블 (MAP 최고점 전략 - Top 5 선정)
    topk_final = 5 
    final_ranked_results = hard_vote_results(
        sparse_res, 
        dense_res_list, 
        top_k=topk_final, 
        weights=[5, 3, 1] # 최고 MAP를 달성한 가중치
    )
    
    return final_ranked_results