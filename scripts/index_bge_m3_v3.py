import os
import json
import urllib3
import numpy as np
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv

# 1. 환경변수 로드
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 2. 설정
BGE_M3_MODEL_PATH = "/root/IR/finetuned_bge_m3_v3"
DOC_PATH = "/root/IR/data/documents.jsonl"
INDEX_NAME = "test"

# 3. Elasticsearch 클라이언트
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", os.getenv("ES_PASSWORD"))
)

def update_mapping():
    print("Updating mapping for BGE-M3 V3...")
    mapping = {
        "properties": {
            "embeddings_bge_m3_v3": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
    es.indices.put_mapping(index=INDEX_NAME, body=mapping)
    print("Mapping updated.")

def get_docid_to_esid():
    print("Fetching docid to ES _id mapping...")
    mapping = {}
    query = {
        "query": {"match_all": {}},
        "_source": ["docid"],
        "size": 10000
    }
    res = es.search(index=INDEX_NAME, body=query)
    for hit in res['hits']['hits']:
        docid = hit['_source'].get('docid')
        if docid:
            mapping[docid] = hit['_id']
    print(f"Found {len(mapping)} mappings.")
    return mapping

def index_bge_m3_v3():
    docid_to_esid = get_docid_to_esid()
    
    print(f"Loading BGE-M3 V3 model from {BGE_M3_MODEL_PATH}...")
    model = BGEM3FlagModel(BGE_M3_MODEL_PATH, use_fp16=True)
    
    print(f"Reading documents from {DOC_PATH}...")
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    
    doc_contents = [d["content"] for d in docs]
    doc_ids = [d.get("documentID") or d.get("docid") for d in docs]
    
    print(f"Generating embeddings for {len(docs)} documents...")
    batch_size = 32
    all_embeddings = []
    for i in tqdm(range(0, len(doc_contents), batch_size)):
        batch_texts = doc_contents[i:i+batch_size]
        output = model.encode(batch_texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        all_embeddings.append(output['dense_vecs'])
    
    embeddings = np.vstack(all_embeddings)
    
    print("Updating documents in Elasticsearch...")
    actions = []
    for i, doc_id in enumerate(doc_ids):
        es_id = docid_to_esid.get(doc_id)
        if not es_id:
            print(f"Warning: docid {doc_id} not found in ES.")
            continue
            
        actions.append({
            "_op_type": "update",
            "_index": INDEX_NAME,
            "_id": es_id,
            "doc": {
                "embeddings_bge_m3_v3": embeddings[i].tolist()
            }
        })
        
        if len(actions) >= 500:
            helpers.bulk(es, actions)
            actions = []
            
    if actions:
        helpers.bulk(es, actions)
    
    print("BGE-M3 V3 indexing completed.")

if __name__ == "__main__":
    update_mapping()
    index_bge_m3_v3()
