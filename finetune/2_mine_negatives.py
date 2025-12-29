import sys
import os
import json
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.es_connector import es

# 설정
INPUT_QA_FILE = "./data/synthetic_qa.jsonl"
OUTPUT_TRAIN_FILE = "./data/train_data.jsonl"
NEG_COUNT = 7  # Positive 1개 + Negative 7개 = 총 8개 (Batch Size)

def mine_hard_negatives():
    if not os.path.exists(INPUT_QA_FILE):
        print(f"오류: {INPUT_QA_FILE} 파일이 없습니다. 1단계 먼저 실행하세요.")
        return

    print(">>> 2단계: Hard Negative Mining 시작...")
    
    with open(INPUT_QA_FILE, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    training_data = []
    
    for item in tqdm(qa_pairs):
        query = item['query']
        positive_id = item['docid']
        positive_content = item['content']
        
        try:
            # BM25로 유사 문서 검색 (Negative 후보군)
            # 2배수 정도 넉넉하게 가져와서 필터링
            res = es.search(
                index="test", 
                query={"match": {"content": query}}, 
                size=15 
            )
            
            negatives = []
            for hit in res['hits']['hits']:
                found_id = hit['_source']['docid']
                found_content = hit['_source']['content']
                
                # 정답 문서가 아닌 것만 Negative로 추가
                if found_id != positive_id:
                    negatives.append(found_content)
                
                if len(negatives) >= NEG_COUNT:
                    break
            
            # Negative가 충분히 모인 경우에만 학습 데이터로 사용
            if len(negatives) >= NEG_COUNT:
                training_data.append({
                    "query": query,
                    "pos": [positive_content],
                    "neg": negatives
                })
                
        except Exception as e:
            print(f"ES Error: {e}")
            continue
            
    # 결과 저장
    print(f">>> 총 {len(training_data)}개의 학습 데이터셋이 구축되었습니다.")
    with open(OUTPUT_TRAIN_FILE, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    print("DEBUG: Mining started")
    mine_hard_negatives()
    print("DEBUG: Mining finished")