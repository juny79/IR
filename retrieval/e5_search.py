import faiss
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# 전역 변수로 리소스 관리
_model = None
_index = None
_doc_ids = None

CACHE_DIR = Path("/root/IR/cache")
MODEL_NAME = "intfloat/multilingual-e5-large"
INDEX_PATH = CACHE_DIR / "faiss_e5.index"
DOC_IDS_PATH = CACHE_DIR / "doc_ids_e5.json"

def _load_resources():
    global _model, _index, _doc_ids
    
    if _model is None:
        print(f"⚡ Loading E5 Model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    
    if _index is None:
        print(f"📂 Loading FAISS Index: {INDEX_PATH}")
        _index = faiss.read_index(str(INDEX_PATH))
        
    if _doc_ids is None:
        print(f"📄 Loading Doc IDs: {DOC_IDS_PATH}")
        with open(DOC_IDS_PATH, "r", encoding="utf-8") as f:
            _doc_ids = json.load(f)

def search_e5(query: str, top_k: int = 5):
    """
    E5 모델 + FAISS 인덱스를 사용한 검색
    """
    _load_resources()
    
    # E5 쿼리 접두사 추가
    query_text = f"query: {query}"
    
    # 임베딩 생성
    query_embedding = _model.encode([query_text])
    
    # FAISS 검색
    distances, indices = _index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        score = float(distances[0][i])
        doc_id = _doc_ids[idx]
        results.append({"docid": doc_id, "score": score})
        
    return results

if __name__ == "__main__":
    # 테스트
    res = search_e5("과학 기술의 발전", top_k=3)
    print(json.dumps(res, indent=2, ensure_ascii=False))
