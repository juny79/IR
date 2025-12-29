import sys
import os
import json
import time
from tqdm import tqdm

# 프로젝트 루트 경로를 path에 추가 (모듈 import용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solar_client import solar_client

# 설정
INPUT_FILE = "./data/documents.jsonl"
OUTPUT_FILE = "./data/synthetic_qa.jsonl"

def generate_synthetic_qa():
    # 파일이 없으면 에러 처리
    if not os.path.exists(INPUT_FILE):
        print(f"오류: {INPUT_FILE} 파일을 찾을 수 없습니다.")
        return

    print(">>> 1단계: 문서 로딩 중...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f]

    print(f">>> 총 {len(docs)}개 문서에 대해 질문 생성을 시작합니다.")
    
    # 이어쓰기를 위해 기존 데이터 확인 (중단 후 재시작 대비)
    processed_doc_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_doc_ids.add(data['docid'])
                except: pass
    
    print(f">>> 이미 처리된 문서: {len(processed_doc_ids)}개")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for doc in tqdm(docs):
            if doc['docid'] in processed_doc_ids:
                continue

            context = doc['content']
            
            prompt = f"""
            다음은 과학 상식 문서의 내용입니다.
            이 문서를 검색 시스템에서 찾기 위해 사용자가 입력할 법한 '구체적인 질문' 1개를 생성하세요.
            
            [문서 내용]
            {context[:1000]}
            
            조건:
            1. 질문은 반드시 한글로 작성할 것.
            2. 문서의 핵심 키워드가 포함되도록 할 것.
            3. "이 문서는 무엇인가요?" 같은 모호한 질문 금지.
            4. 오직 질문 텍스트만 출력할 것.
            
            질문:"""
            
            try:
                # Solar API 호출
                query = solar_client._call_with_retry(prompt)
                
                if query:
                    # 불필요한 따옴표나 접두어 제거
                    query = query.replace("질문:", "").replace('"', '').strip()
                    
                    item = {
                        "docid": doc['docid'],
                        "query": query,
                        "content": context
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    f.flush() # 실시간 저장
                    
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1) # 에러 발생 시 잠시 대기

if __name__ == "__main__":
    print("DEBUG: Script started")
    generate_synthetic_qa()
    print("DEBUG: Script finished")