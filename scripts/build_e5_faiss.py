"""
E5-large 임베딩 생성 및 FAISS 인덱싱 스크립트
모델: intfloat/multilingual-e5-large
"""
import os
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 설정
MODEL_NAME = "intfloat/multilingual-e5-large"
DATA_PATH = "/root/IR/data/documents.jsonl"
CACHE_DIR = Path("/root/IR/cache")
CACHE_DIR.mkdir(exist_ok=True)

EMB_PATH = CACHE_DIR / "doc_embeddings_e5.npy"
INDEX_PATH = CACHE_DIR / "faiss_e5.index"
DOC_IDS_PATH = CACHE_DIR / "doc_ids_e5.json"

def main():
    print(f"🚀 E5 인덱싱 시작: {MODEL_NAME}")
    
    # 1. 문서 로드
    documents = []
    doc_ids = []
    print("📂 문서 로딩 중...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                documents.append(doc['content'])
                doc_ids.append(doc['docid'])
    
    print(f"   - 총 문서 수: {len(documents)}개")
    
    # 2. 임베딩 생성
    if EMB_PATH.exists():
        print("✅ 기존 임베딩 로드 중...")
        embeddings = np.load(EMB_PATH)
    else:
        print("⚡ 임베딩 생성 중 (GPU)...")
        model = SentenceTransformer(MODEL_NAME)
        
        # E5는 문서에 'passage: ' 접두사 필요
        passage_texts = ["passage: " + doc for doc in documents]
        
        embeddings = model.encode(
            passage_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        ).astype("float32")
        
        np.save(EMB_PATH, embeddings)
        print("✅ 임베딩 저장 완료")

    # 3. FAISS 인덱스 생성
    print("🔍 FAISS 인덱스 생성 중...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, str(INDEX_PATH))
    
    # Doc ID 매핑 저장
    with open(DOC_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
        
    print(f"🎉 작업 완료!")
    print(f"   - Index: {INDEX_PATH}")
    print(f"   - Embeddings: {EMB_PATH}")
    print(f"   - Doc IDs: {DOC_IDS_PATH}")

if __name__ == "__main__":
    main()
