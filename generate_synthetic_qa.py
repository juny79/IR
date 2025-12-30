import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from models.solar_client import SolarClient

def generate_questions_for_doc(solar, doc_content, num_questions=3):
    prompt = f"""당신은 과학 교육 전문가입니다. 아래 제공된 [문서]의 내용을 바탕으로, 이 문서를 '정답'으로 채택할 수밖에 없는 구체적이고 명확한 과학 질문을 {num_questions}개 생성하세요.

[지침]
1. 질문은 한국어로 작성하세요.
2. 질문은 문서의 핵심 정보를 포함해야 하며, 너무 일반적이지 않아야 합니다.
3. 각 질문은 한 줄씩 작성하고, 질문 번호나 기호 없이 질문 내용만 출력하세요.
4. 문서와 관련 없는 질문은 만들지 마세요.

[문서]
{doc_content}

[질문 목록]"""
    
    try:
        response = solar._call_with_retry(prompt)
        questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return questions[:num_questions]
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def main():
    solar = SolarClient()
    
    input_file = "data/documents.jsonl"
    output_file = "data/synthetic_qa_solar.jsonl"
    cache_file = "data/synthetic_qa_cache.json"
    
    # Load cache
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    
    documents = []
    with open(input_file, 'r') as f:
        for line in f:
            documents.append(json.loads(line))
    
    print(f"Total documents: {len(documents)}")
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for doc in tqdm(documents):
            docid = doc['docid']
            if docid in cache:
                continue
            
            content = doc['content']
            questions = generate_questions_for_doc(solar, content, num_questions=3)
            
            if questions:
                result = {
                    "docid": docid,
                    "questions": questions,
                    "content": content
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                
                cache[docid] = True
                # Save cache every 10 docs
                if len(cache) % 10 == 0:
                    with open(cache_file, 'w') as f_cache:
                        json.dump(cache, f_cache)
            
            # Rate limiting safety
            time.sleep(0.1)

if __name__ == "__main__":
    main()
