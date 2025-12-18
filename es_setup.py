import json
from elasticsearch import Elasticsearch, helpers
from models.embedding_client import embedding_client
from retrieval.es_connector import es_username, es_password, es_ca_cert

# Elasticsearch 클라이언트 설정
es = Elasticsearch(
    ['https://localhost:9200'], 
    basic_auth=(es_username, es_password), 
    ca_certs=es_ca_cert
)

def create_index():
    index_name = "test"
    # Nori 형태소 분석기 및 다중 임베딩 필드 설정
    settings = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["nori_posfilter"]
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                }
            }
        }
    }
    
    # 최고점 전략: 다양한 차원의 벡터DB 필드 구성
    mappings = {
        "properties": {
            "docid": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "nori"},
            "embeddings_sbert": {
                "type": "dense_vector", "dims": 768, "index": True, "similarity": "l2_norm"
            },
            "embeddings_solar": {
                "type": "dense_vector", "dims": 4096, "index": True, "similarity": "l2_norm"
            },
            "embeddings_gemini": {
                "type": "dense_vector", "dims": 3072, "index": True, "similarity": "l2_norm"
            }
        }
    }

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, settings=settings, mappings=mappings)
    print(f"Index '{index_name}' created.")

def index_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    actions = []
    print("Starting embedding and indexing...")
    for i, doc in enumerate(docs):
        # 텍스트 추출 및 다중 임베딩 생성 (전략에 따라 선택)
        content = doc["content"]
        
        # SBERT 임베딩 (필수)
        doc["embeddings_sbert"] = embedding_client.get_embedding(content, model_name="sbert").tolist()
        
        # (옵션) Solar/Gemini 임베딩 추가 시 주석 해제
        # doc["embeddings_solar"] = embedding_client.get_embedding(content, model_name="solar").tolist()
        # doc["embeddings_gemini"] = embedding_client.get_embedding(content, model_name="gemini").tolist()
        
        actions.append({
            "_index": "test",
            "_source": doc
        })
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} documents...")

    helpers.bulk(es, actions)
    print("Bulk indexing completed.")

if __name__ == "__main__":
    create_index()
    index_documents("./data/documents.jsonl")