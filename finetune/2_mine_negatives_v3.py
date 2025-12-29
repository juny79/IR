import sys
import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.es_connector import es

# 설정
INPUT_QA_FILE = "./data/synthetic_qa_solar.jsonl"
OUTPUT_TRAIN_FILE = "./data/train_data_v3.jsonl"
DOC_PATH = "./data/documents.jsonl"
NEG_COUNT = 7
POOL_SIZE = 50

def mine_hard_negatives_v3():
    if not os.path.exists(INPUT_QA_FILE):
        print(f"오류: {INPUT_QA_FILE} 파일이 없습니다.")
        return

    print(">>> 2단계 V3: 고도화된 Hard Negative Mining 시작 (Hybrid + Reranker)...")
    
    # 1. 문서 데이터 로드
    print("⏳ 문서 로딩 중...")
    documents = {}
    with open(DOC_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            documents[obj['docid']] = obj['content']
    doc_ids = list(documents.keys())
    doc_contents = list(documents.values())

    # 2. 모델 및 인덱스 로드
    print("⏳ BGE-M3 모델 및 FAISS 인덱스 로드 중...")
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    CACHE_DIR = "./cache/bge_m3"
    FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "bge_m3_dense.index")
    if not os.path.exists(FAISS_INDEX_PATH):
        print("오류: FAISS 인덱스가 없습니다. 먼저 인덱싱을 완료하세요.")
        return
    index = faiss.read_index(FAISS_INDEX_PATH)

    print("⏳ Reranker 로드 중...")
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda')

    # 3. QA 데이터 로드 및 Flatten
    qa_pairs = []
    with open(INPUT_QA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            docid = item['docid']
            content = item['content']
            for q in item['questions']:
                qa_pairs.append({
                    "query": q,
                    "docid": docid,
                    "content": content
                })
    
    training_data = []
    
    print(f"🚀 Mining 시작 (총 {len(qa_pairs)}개 질문)...")
    for item in tqdm(qa_pairs):
        query = item['query']
        positive_id = item['docid']
        positive_content = item['content']
        
        candidate_contents = set()
        
        try:
            # A. BM25 Retrieval
            res_bm25 = es.search(
                index="test", 
                query={"match": {"content": query}}, 
                size=POOL_SIZE 
            )
            for hit in res_bm25['hits']['hits']:
                if hit['_source']['docid'] != positive_id:
                    candidate_contents.add(hit['_source']['content'])
            
            # B. Dense Retrieval
            q_emb = model.encode([query], return_dense=True)['dense_vecs']
            _, indices = index.search(q_emb.astype('float32'), POOL_SIZE)
            for idx in indices[0]:
                if doc_ids[idx] != positive_id:
                    candidate_contents.add(doc_contents[idx])
            
            # C. Reranking to find the hardest negatives
            candidates = list(candidate_contents)
            if not candidates:
                continue
                
            pairs = [[query, c] for c in candidates]
            # Reranker로 점수 계산
            scores = reranker.predict(pairs, batch_size=128, show_progress_bar=False)
            
            # 점수가 높은 순(Hardest)으로 정렬
            scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            
            negatives = [c for c, s in scored_candidates[:NEG_COUNT]]
            
            if len(negatives) >= 1: # 최소 1개라도 있으면 추가
                training_data.append({
                    "query": query,
                    "pos": [positive_content],
                    "neg": negatives
                })
        except Exception as e:
            print(f"Error mining for query '{query}': {e}")
            continue
            
        # 중간 저장 (1000개마다)
        if len(training_data) % 1000 == 0:
            with open(OUTPUT_TRAIN_FILE, 'w', encoding='utf-8') as f:
                for d in training_data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # 최종 저장
    with open(OUTPUT_TRAIN_FILE, 'w', encoding='utf-8') as f:
        for d in training_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    
    print(f"✅ Mining 완료! {len(training_data)}개의 학습 데이터가 {OUTPUT_TRAIN_FILE}에 저장되었습니다.")

if __name__ == "__main__":
    mine_hard_negatives_v3()
