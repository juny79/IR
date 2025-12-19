import os
import urllib3
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from models.embedding_client import embedding_client

# 1. 환경변수 로드
load_dotenv()

# 2. 보안 경고 끄기 (SSL 검증 무시)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 3. Elasticsearch 클라이언트 설정 (HTTP로 접속)
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", os.getenv("ES_PASSWORD"))
)

def sparse_retrieve(query_str, size=10):
    """BM25 기반 키워드 검색"""
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    # 검색 실행
    return es.search(index="test", query=query, size=size)

def dense_retrieve(query_str, size=10, embedding_field="embeddings_sbert"):
    """벡터 기반 KNN 검색"""
    # 쿼리를 벡터로 변환 (embedding_client 활용)
    query_embedding = embedding_client.get_query_embedding(query_str, model_name="sbert").tolist()

    # KNN 검색 쿼리 구성
    knn = {
        "field": embedding_field,
        "query_vector": query_embedding,
        "k": size,
        "num_candidates": 100
    }

    # 검색 실행
    return es.search(index="test", knn=knn)