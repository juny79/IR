from elasticsearch import Elasticsearch
from models.embedding_client import embedding_client

# 설정 값 (환경에 맞게 수정)
es_username = 'elastic'
es_password = 'YOUR_PASSWORD' # 베이스라인에서 생성된 비밀번호 입력
es_ca_cert = '/content/elasticsearch-8.8.0/config/certs/http_ca.crt'

es = Elasticsearch(
    ['https://localhost:9200'], 
    basic_auth=(es_username, es_password), 
    ca_certs=es_ca_cert
)

def sparse_retrieve(query_str, size=10):
    """BM25 기반 역색인 검색"""
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size)

def dense_retrieve(query_str, size=10, embedding_field="embeddings_sbert"):
    """Vector 유사도 기반 KNN 검색 (MAP 극대화 전략)"""
    # 필드명에 맞는 임베딩 모델 선택
    model_name = "sbert"
    if "solar" in embedding_field: model_name = "solar"
    elif "gemini" in embedding_field: model_name = "gemini"
    
    query_embedding = embedding_client.get_query_embedding(query_str, model_name=model_name)

    knn = {
        "field": embedding_field,
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    return es.search(index="test", knn=knn)

def get_content_by_docid(docid):
    """Hard Voting 후 최종 문서 내용을 가져오기 위한 유틸리티"""
    query = {"term": {"docid": docid}}
    res = es.search(index="test", query=query, size=1)
    if res['hits']['total']['value'] > 0:
        return res['hits']['hits'][0]['_source']['content']
    return ""