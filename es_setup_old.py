import os
import json
import urllib3
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
from models.embedding_client import embedding_client

# 1. 환경변수 로드
load_dotenv()

cd /root/IR && git remote -v && echo "---" && git branch -a && echo "---" && git status 위해 필수)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 3. Elasticsearch 클        직접 생성 (HTTP로 접속)
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", os.getenv("ES_PASSWORD"))
)

def create_index():
    index_name = "test"
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
    
    mappings = {
        "properties": {
            "docid": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "nori"},
            "embeddings_sbert": {
                "type": "dense_vector", "dims": 768, "index": True, "similarity": "l2_norm"
            }
        }
    }

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, settings=settings, mappings=mappings)
    print(f"Index '{index_name}' created successfully.")

def index_documents(file_path):
    print(f"Reading documents from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    actions = []
    print(f"Start embedding and indexing {len(docs)} documents...")
    
    for i, doc in enumerate(docs):
        # 문서 내용 추출 및 임베딩 생성
        content = doc["content"]
        doc_id = doc.get("documentID") or doc.get("docid")
        
        # SBERT 임베딩 생성
        embedding = embedding_client.get_embedding(content, model_name="sbert").tolist()
        
        doc["embeddings_sbert"] = embedding
        doc["docid"] = doc_id
        
        actions.append({
            "_index": "test",
            "_source": doc
        })
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} documents...")

    # Bulk Indexing 수
    helpers.bulk(es, actions)
    print("Bulk indexing completed.")

if __name__ == "__main__":
    # 실행 순서: 인덱스 생성 -> 데이터 삽입
    create_index()
    index_documents("./data/documents.jsonl")
