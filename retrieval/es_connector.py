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
    """
    벡터 기반 KNN 검색 (Phase 3: 다중 임베딩 지원)
    
    Args:
        query_str: 검색 쿼리
        size: 반환할 문서 개수
        embedding_field: 검색할 임베딩 필드
            - "embeddings_sbert": SBERT (기본)
            - "embeddings_upstage_2048": Upstage Solar (4096 -> 2048 trunc + L2 norm)
            - "embeddings_gemini": Gemini
        - "embeddings_bge_m3_v3": BGE-M3 V3 (Fine-tuned)
    """
    # 임베딩 필드에 맞는 모델 선택
    if embedding_field == "embeddings_upstage_2048":
        model_name = "upstage"
    elif embedding_field == "embeddings_gemini":
        model_name = "gemini"
    elif embedding_field == "embeddings_bge_m3_v3":
        model_name = "bge_m3_v3"
    else:
        model_name = "sbert"
    
    # 쿼리를 벡터로 변환 (embedding_client 활용)
    query_embedding = embedding_client.get_query_embedding(query_str, model_name=model_name).tolist()

    # ES dense_vector dims 제한(<=2048) 대응: Upstage(4096) 벡터는 앞 2048 dims를 사용하고 L2 normalize
    if embedding_field == "embeddings_upstage_2048":
        if len(query_embedding) > 2048:
            query_embedding = query_embedding[:2048]
        # L2 normalize (cosine similarity에 유리)
        norm = sum(x * x for x in query_embedding) ** 0.5
        if norm > 0:
            query_embedding = [x / norm for x in query_embedding]

    # KNN 검색 쿼리 구성
    # Elasticsearch KNN은 일반적으로 num_candidates >= k 를 요구합니다.
    # TOP_K_RETRIEVE 등을 키웠을 때 쿼리가 깨지지 않도록 자동 보정합니다.
    default_num_candidates = int(os.getenv("ES_NUM_CANDIDATES", "100"))
    num_candidates = max(int(size), default_num_candidates)
    knn = {
        "field": embedding_field,
        "query_vector": query_embedding,
        "k": int(size),
        "num_candidates": num_candidates,
    }

    # 검색 실행
    return es.search(index="test", knn=knn)