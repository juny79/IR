"""
Step 1: 검색 결과 캐싱 (Heavy Task)
- 각 질문에 대해 TOP_K=100 기준으로 모든 검색 결과 저장
- Sparse(BM25), Dense(SBERT, Gemini) 결과를 DocID + 순위로 캐싱
- 이후 파라미터만 바꿔가며 시뮬레이션 가능
"""

import json
import time
from retrieval.es_connector import sparse_retrieve, dense_retrieve
from models.llm_client import llm_client
from models.solar_client import solar_client

# 설정
TOP_K_CACHE = 100  # 캐싱할 최대 문서 개수
CACHE_FILE = 'search_results_cache.json'

def cache_all_search_results():
    """
    220개 질문에 대한 모든 검색 결과를 캐싱
    """
    print("=" * 80)
    print("Step 1: 검색 결과 캐싱 시작")
    print("=" * 80)
    print(f"캐싱 대상: 220개 질문")
    print(f"TOP_K: {TOP_K_CACHE}")
    print(f"검색 엔진: Sparse(BM25) + Dense(SBERT, Gemini)")
    print()
    
    # 평가 데이터 로드
    questions = []
    with open('./data/eval.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append({
                'eval_id': data['eval_id'],
                'msg': data['msg']
            })
    
    print(f"✅ {len(questions)}개 질문 로드 완료")
    print()
    
    # 캐시 데이터 구조
    cache = {}
    
    # 각 질문 처리
    for i, q in enumerate(questions, 1):
        eval_id = q['eval_id']
        msg = q['msg']
        
        print(f"[{i}/{len(questions)}] {eval_id}: {msg[:50]}...")
        
        try:
            # 1. LLM 쿼리 분석
            analysis = llm_client.analyze_query([{'role': 'user', 'content': msg}])
            
            if not analysis.tool_calls:
                print("  ⚠️ 검색 불필요 (tool_calls 없음)")
                cache[eval_id] = {
                    'original_query': msg,
                    'standalone_query': None,
                    'confidence': 0.0,
                    'should_search': False,
                    'sparse_results': [],
                    'dense_sbert_results': [],
                    'dense_gemini_results': []
                }
                continue
            
            # 2. 쿼리 추출
            args = json.loads(analysis.tool_calls[0].function.arguments)
            query = args.get('standalone_query', '')
            confidence = args.get('confidence', 0.0)
            
            # 3. HyDE 생성
            hypothetical_answer = solar_client.generate_hypothetical_answer(query)
            if hypothetical_answer:
                hyde_query = f"{query}\n{hypothetical_answer}"
            else:
                hyde_query = query
            
            # 4. Sparse 검색 (BM25)
            sparse_res = sparse_retrieve(hyde_query, TOP_K_CACHE)
            sparse_docids = [hit['_source']['docid'] for hit in sparse_res['hits']['hits']]
            
            # 5. Dense 검색 (SBERT)
            dense_sbert_res = dense_retrieve(hyde_query, TOP_K_CACHE, "embeddings_sbert")
            sbert_docids = [hit['_source']['docid'] for hit in dense_sbert_res['hits']['hits']]
            
            # 6. Dense 검색 (Gemini)
            dense_gemini_res = dense_retrieve(hyde_query, TOP_K_CACHE, "embeddings_gemini")
            gemini_docids = [hit['_source']['docid'] for hit in dense_gemini_res['hits']['hits']]
            
            # 7. 캐시 저장
            cache[eval_id] = {
                'original_query': msg,
                'standalone_query': query,
                'hyde_query': hyde_query,
                'confidence': confidence,
                'should_search': True,
                'sparse_results': sparse_docids,
                'dense_sbert_results': sbert_docids,
                'dense_gemini_results': gemini_docids
            }
            
            print(f"  ✅ Sparse: {len(sparse_docids)}, SBERT: {len(sbert_docids)}, Gemini: {len(gemini_docids)}")
            
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            cache[eval_id] = {
                'original_query': msg,
                'error': str(e)
            }
        
        # 진행률 표시
        if i % 10 == 0:
            print()
    
    # 캐시 저장
    print()
    print("=" * 80)
    print("캐시 저장 중...")
    print("=" * 80)
    
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    
    print(f"✅ {CACHE_FILE}에 저장 완료")
    print()
    
    # 통계
    total = len(cache)
    searchable = sum(1 for v in cache.values() if v.get('should_search', False))
    filtered = total - searchable
    
    print("=" * 80)
    print("캐싱 완료!")
    print("=" * 80)
    print(f"전체: {total}개")
    print(f"검색 수행: {searchable}개 ({searchable/total:.1%})")
    print(f"필터링: {filtered}개 ({filtered/total:.1%})")
    print()
    print("다음 단계: python grid_search_cached.py")
    print()


if __name__ == "__main__":
    start = time.time()
    cache_all_search_results()
    duration = time.time() - start
    print(f"총 소요 시간: {duration/60:.1f}분")
