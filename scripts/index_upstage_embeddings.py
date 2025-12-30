import os
import sys
import urllib3
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
from tqdm import tqdm
import time
import argparse

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.embedding_client import embedding_client

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", os.getenv("ES_PASSWORD"))
)

INDEX_NAME = "test"
# NOTE: ES dense_vector dims 제한(<=2048) 때문에 Upstage(4096)는 2048로 truncation 해서 별도 필드로 저장
FIELD_NAME = "embeddings_upstage_2048"
DIMS = 2048


def add_upstage_embedding_field():
    """Elasticsearch 인덱스에 Upstage embedding 필드 추가"""
    print(f"\n=== Adding {FIELD_NAME} field to index '{INDEX_NAME}' ===\n")

    mapping_update = {
        "properties": {
            FIELD_NAME: {
                "type": "dense_vector",
                "dims": DIMS,
                "index": True,
                "similarity": "cosine",
            }
        }
    }

    try:
        es.indices.put_mapping(index=INDEX_NAME, body=mapping_update)
        print(f"✅ {FIELD_NAME} 필드 추가 완료\n")
    except Exception as e:
        print(f"⚠️  필드 추가 실패 또는 이미 존재: {e}\n")


def update_documents_with_upstage_embeddings(batch_size=5, sleep_s=0.7, limit=0, only_missing=True):
    """문서에 Upstage embedding 추가 (기본: 미존재 문서만)"""
    if not os.getenv("UPSTAGE_API_KEY"):
        raise RuntimeError("UPSTAGE_API_KEY is not set. Aborting to avoid silent failures.")

    print(f"=== Updating {INDEX_NAME} with Upstage embeddings ===\n")

    total_docs = es.count(index=INDEX_NAME)["count"]
    print(f"Total documents in index: {total_docs}")

    if only_missing:
        base_query = {"query": {"bool": {"must_not": [{"exists": {"field": FIELD_NAME}}]}}}
        missing = es.count(index=INDEX_NAME, body=base_query)["count"]
        print(f"Documents missing {FIELD_NAME}: {missing}\n")
        target_total = missing
    else:
        base_query = {"query": {"match_all": {}}}
        target_total = total_docs

    response = es.search(index=INDEX_NAME, body=base_query, scroll="5m", size=batch_size)
    scroll_id = response.get("_scroll_id")
    hits = response["hits"]["hits"]

    processed = 0
    failed = 0

    pbar = tqdm(total=(limit if limit and limit < target_total else target_total), desc="Indexing Upstage embeddings")

    try:
        while hits:
            actions = []

            for hit in hits:
                if limit and processed >= limit:
                    hits = []
                    break

                doc_es_id = hit["_id"]
                content = hit["_source"].get("content", "")

                try:
                    last_err = None
                    vec = None
                    for attempt in range(3):
                        try:
                            vec = embedding_client.get_embedding([content], model_name="upstage")[0]
                            vec = vec.tolist() if hasattr(vec, "tolist") else list(vec)

                            # Upstage는 4096 dims. ES 제한으로 앞 2048 dims 사용 + L2 normalize
                            if len(vec) > DIMS:
                                vec = vec[:DIMS]
                            if len(vec) != DIMS:
                                raise ValueError(f"Unexpected embedding dim after trunc: {len(vec)} (expected {DIMS})")

                            norm = sum(x * x for x in vec) ** 0.5
                            if norm <= 0:
                                raise ValueError("Zero-norm embedding (likely API failure)")
                            vec = [x / norm for x in vec]
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            time.sleep(1.5 * (attempt + 1))
                    if last_err is not None:
                        raise last_err

                    actions.append({
                        "_op_type": "update",
                        "_index": INDEX_NAME,
                        "_id": doc_es_id,
                        "doc": {FIELD_NAME: vec},
                    })
                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        print(f"⚠️  Upstage embedding failed (doc_es_id={doc_es_id}): {e}")
                    continue

            if actions:
                helpers.bulk(es, actions, raise_on_error=False)
                processed += len(actions)
                pbar.update(len(actions))

            if not hits:
                break

            response = es.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = response.get("_scroll_id")
            hits = response["hits"]["hits"]

            time.sleep(sleep_s)

    finally:
        if scroll_id:
            try:
                es.clear_scroll(scroll_id=scroll_id)
            except Exception:
                pass
        pbar.close()

    print("\n✅ Indexing completed:")
    print(f"   - Processed: {processed}")
    print(f"   - Failed: {failed}")


def verify_upstage_embeddings(sample_size=3):
    print("\n=== Verifying Upstage embeddings ===\n")

    resp = es.search(index=INDEX_NAME, body={"query": {"match_all": {}}, "size": sample_size})
    for i, hit in enumerate(resp["hits"]["hits"], 1):
        doc_id = hit["_source"].get("docid", "N/A")
        has_field = FIELD_NAME in hit["_source"]
        print(f"Document {i} (ID: {doc_id}): {FIELD_NAME}={'✅' if has_field else '❌'}")
        if has_field:
            vec = hit["_source"][FIELD_NAME]
            print(f"  - dim: {len(vec)}")
            print(f"  - sample: {vec[:3]}")

    with_field = es.count(index=INDEX_NAME, body={"query": {"exists": {"field": FIELD_NAME}}})["count"]
    total = es.count(index=INDEX_NAME)["count"]
    print(f"\nDocuments with {FIELD_NAME}: {with_field} / {total}  (coverage={(with_field/total*100):.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Index Upstage embeddings into Elasticsearch")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0, help="0=all")
    parser.add_argument("--all", action="store_true", help="Process all docs (not only missing)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Upstage Embedding Indexing Script")
    print("  WARNING: This makes MANY Upstage API calls (cost/time).")
    print("=" * 60 + "\n")

    add_upstage_embedding_field()
    update_documents_with_upstage_embeddings(
        batch_size=max(1, args.batch_size),
        sleep_s=max(0.0, args.sleep),
        limit=max(0, args.limit),
        only_missing=(not args.all),
    )
    verify_upstage_embeddings()


if __name__ == "__main__":
    main()
