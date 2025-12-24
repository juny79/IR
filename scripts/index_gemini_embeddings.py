import os
import sys
import json
import urllib3
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
from tqdm import tqdm
import time

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.embedding_client import embedding_client

# 환경변수 로드
load_dotenv()

# 보안 경고 끄기
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Elasticsearch 클라이언트 생성
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", os.getenv("ES_PASSWORD"))
)

def add_gemini_embedding_field():
    """Elasticsearch 인덱스에 gemini_embedding 필드 추가"""
    index_name = "test"
    
    print(f"\n=== Adding gemini_embedding field to index '{index_name}' ===\n")
    
    # 새로운 필드 매핑 추가
    mapping_update = {
        "properties": {
            "embeddings_gemini": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"  # Gemini는 cosine similarity 사용
            }
        }
    }
    
    try:
        es.indices.put_mapping(index=index_name, body=mapping_update)
        print("✅ gemini_embedding 필드 추가 완료\n")
    except Exception as e:
        print(f"⚠️  필드 추가 실패 또는 이미 존재: {e}\n")

def update_documents_with_gemini_embeddings(batch_size=10):
    """모든 문서에 Gemini embedding 추가"""
    index_name = "test"
    
    print(f"=== Updating {index_name} with Gemini embeddings ===\n")
    
    # 전체 문서 수 확인
    total_docs = es.count(index=index_name)['count']
    print(f"Total documents: {total_docs}\n")
    
    # 스크롤 검색으로 모든 문서 가져오기
    query = {"query": {"match_all": {}}}
    
    # 초기 검색 (스크롤 시작)
    response = es.search(
        index=index_name,
        body=query,
        scroll='5m',
        size=batch_size
    )
    
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']
    
    processed = 0
    failed = 0
    
    # Progress bar
    pbar = tqdm(total=total_docs, desc="Indexing Gemini embeddings")
    
    while hits:
        actions = []
        
        for hit in hits:
            doc_id = hit['_id']
            content = hit['_source'].get('content', '')
            
            try:
                # Gemini embedding 생성 (retrieval_document task)
                gemini_embedding = embedding_client.get_embedding(
                    [content], 
                    model_name="gemini"
                )[0].tolist()
                
                # 업데이트 액션 준비
                actions.append({
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": doc_id,
                    "doc": {
                        "embeddings_gemini": gemini_embedding
                    }
                })
                
            except Exception as e:
                print(f"\n⚠️  Document {doc_id} failed: {e}")
                failed += 1
        
        # Bulk update 실행
        if actions:
            try:
                helpers.bulk(es, actions, raise_on_error=False)
                processed += len(actions)
                pbar.update(len(actions))
            except Exception as e:
                print(f"\n❌ Bulk update error: {e}")
                failed += len(actions)
        
        # 다음 배치 가져오기
        response = es.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        
        # API 레이트 리밋 방지
        time.sleep(0.5)
    
    # 스크롤 컨텍스트 삭제
    es.clear_scroll(scroll_id=scroll_id)
    pbar.close()
    
    print(f"\n✅ Indexing completed:")
    print(f"   - Processed: {processed}")
    print(f"   - Failed: {failed}")
    print(f"   - Success rate: {(processed/(processed+failed)*100):.2f}%")

def verify_gemini_embeddings():
    """Gemini embedding이 제대로 추가되었는지 확인"""
    index_name = "test"
    
    print("\n=== Verifying Gemini embeddings ===\n")
    
    # 샘플 문서 3개 확인
    response = es.search(
        index=index_name,
        body={"query": {"match_all": {}}, "size": 3}
    )
    
    for i, hit in enumerate(response['hits']['hits'], 1):
        doc_id = hit['_source'].get('docid', 'N/A')
        has_sbert = 'embeddings_sbert' in hit['_source']
        has_gemini = 'embeddings_gemini' in hit['_source']
        
        print(f"Document {i} (ID: {doc_id}):")
        print(f"  - SBERT embedding: {'✅' if has_sbert else '❌'}")
        print(f"  - Gemini embedding: {'✅' if has_gemini else '❌'}")
        
        if has_gemini:
            gemini_vec = hit['_source']['embeddings_gemini']
            print(f"  - Gemini dim: {len(gemini_vec)}")
            print(f"  - Gemini sample: {gemini_vec[:3]}")
        print()
    
    # Gemini embedding이 있는 문서 수 확인
    gemini_count = es.count(
        index=index_name,
        body={"query": {"exists": {"field": "embeddings_gemini"}}}
    )['count']
    
    total_count = es.count(index=index_name)['count']
    
    print(f"Documents with Gemini embedding: {gemini_count} / {total_count}")
    print(f"Coverage: {(gemini_count/total_count*100):.2f}%")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Gemini Embedding Indexing Script")
    print("  Estimated time: 2-3 hours for 4,272 documents")
    print("="*60 + "\n")
    
    # 단계 1: 필드 추가
    add_gemini_embedding_field()
    
    # 단계 2: Gemini embedding 생성 및 업데이트
    # batch_size=10: 한 번에 10개씩 처리 (API 레이트 리밋 고려)
    update_documents_with_gemini_embeddings(batch_size=10)
    
    # 단계 3: 검증
    verify_gemini_embeddings()
    
    print("\n✅ All steps completed!")
