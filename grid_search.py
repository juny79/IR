"""
Grid Search for Optimal Hyperparameters
빠른 평가를 위해 eval.jsonl의 일부만 사용하여 최적 파라미터 조합 탐색
"""

import json
import time
from itertools import product
from eval_rag import answer_question_optimized
from retrieval.es_connector import es

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 그리드 서치 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 샘플 크기 (빠른 테스트용 - 전체 220개 중 일부)
SAMPLE_SIZE = 50  # 50개로 빠른 평가 (약 5-10분/조합)

# 테스트할 파라미터 조합
PARAM_GRID = {
    'voting_weights': [
        [5, 4, 2],  # Phase 4D 기본
        [6, 4, 2],  # Sparse 강조
        [4, 4, 2],  # Sparse/SBERT 균형
        [5, 5, 2],  # SBERT 강조
        [5, 4, 3],  # Gemini 강조
    ],
    'top_k_retrieve': [50, 60, 70],  # Reranker 후보군
    'confidence_threshold': [0.65, 0.7, 0.75],  # 게이팅 임계값
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_map(results, ground_truth):
    """
    MAP (Mean Average Precision) 계산
    
    Args:
        results: 예측 결과 리스트 [{"eval_id": ..., "topk": [...]}, ...]
        ground_truth: 정답 데이터 {eval_id: [correct_doc_ids], ...}
    
    Returns:
        MAP 점수 (0.0 ~ 1.0)
    """
    total_ap = 0.0
    count = 0
    
    for result in results:
        eval_id = result['eval_id']
        predicted = result['topk']
        
        if eval_id not in ground_truth:
            continue
        
        correct = set(ground_truth[eval_id])
        
        if not correct:
            continue
        
        # Average Precision 계산
        hits = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(predicted, 1):
            if doc_id in correct:
                hits += 1
                precision_sum += hits / i
        
        if hits > 0:
            ap = precision_sum / len(correct)
        else:
            ap = 0.0
        
        total_ap += ap
        count += 1
    
    return total_ap / count if count > 0 else 0.0


def load_sample_data(sample_size=50):
    """
    eval.jsonl에서 샘플 데이터 로드
    
    Args:
        sample_size: 샘플 크기
    
    Returns:
        questions: 질문 리스트
        ground_truth: 정답 데이터
    """
    questions = []
    ground_truth = {}
    
    with open('./data/eval.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            
            data = json.loads(line)
            questions.append({
                'eval_id': data['eval_id'],
                'msg': data['msg']
            })
            
            # 정답 데이터 로드 (있다면)
            # 실제 정답 파일이 있다면 여기서 로드
            # 없으면 리더보드 제출로만 확인 가능
            ground_truth[data['eval_id']] = []  # 임시
    
    return questions, ground_truth


def evaluate_params(voting_weights, top_k_retrieve, confidence_threshold, questions):
    """
    특정 파라미터 조합으로 평가 수행
    
    Args:
        voting_weights: Hard Voting 가중치
        top_k_retrieve: Reranker 후보군 크기
        confidence_threshold: 게이팅 신뢰도 임계값
        questions: 평가할 질문 리스트
    
    Returns:
        results: 예측 결과 리스트
        stats: 통계 정보
    """
    import eval_rag
    
    # 파라미터 동적 변경
    eval_rag.VOTING_WEIGHTS = voting_weights
    eval_rag.TOP_K_RETRIEVE = top_k_retrieve
    eval_rag.CONFIDENCE_THRESHOLD = confidence_threshold
    
    results = []
    stats = {
        'total': len(questions),
        'with_topk': 0,
        'without_topk': 0,
        'avg_time': 0.0
    }
    
    total_time = 0.0
    
    for q in questions:
        start_time = time.time()
        
        try:
            result = answer_question_optimized(q['msg'])
            
            output = {
                'eval_id': q['eval_id'],
                'standalone_query': result['standalone_query'],
                'topk': result['topk'],
                'answer': result['answer']
            }
            
            results.append(output)
            
            if result['topk']:
                stats['with_topk'] += 1
            else:
                stats['without_topk'] += 1
            
        except Exception as e:
            print(f"  오류 [ID {q['eval_id']}]: {str(e)[:50]}")
            results.append({
                'eval_id': q['eval_id'],
                'standalone_query': '',
                'topk': [],
                'answer': ''
            })
            stats['without_topk'] += 1
        
        elapsed = time.time() - start_time
        total_time += elapsed
    
    stats['avg_time'] = total_time / len(questions)
    
    return results, stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 그리드 서치
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_grid_search():
    """
    그리드 서치 실행
    """
    print("=" * 80)
    print("Grid Search for Optimal Hyperparameters")
    print("=" * 80)
    print(f"샘플 크기: {SAMPLE_SIZE}개")
    print(f"파라미터 조합 수: {len(PARAM_GRID['voting_weights']) * len(PARAM_GRID['top_k_retrieve']) * len(PARAM_GRID['confidence_threshold'])}개")
    print()
    
    # 샘플 데이터 로드
    print("샘플 데이터 로드 중...")
    questions, ground_truth = load_sample_data(SAMPLE_SIZE)
    print(f"✅ {len(questions)}개 질문 로드 완료")
    print()
    
    # 모든 파라미터 조합 생성
    param_combinations = list(product(
        PARAM_GRID['voting_weights'],
        PARAM_GRID['top_k_retrieve'],
        PARAM_GRID['confidence_threshold']
    ))
    
    print(f"총 {len(param_combinations)}개 조합 테스트 시작...")
    print()
    
    # 결과 저장
    all_results = []
    
    for idx, (weights, top_k, conf_thresh) in enumerate(param_combinations, 1):
        print("-" * 80)
        print(f"[{idx}/{len(param_combinations)}] 테스트 중...")
        print(f"  VOTING_WEIGHTS: {weights}")
        print(f"  TOP_K_RETRIEVE: {top_k}")
        print(f"  CONFIDENCE_THRESHOLD: {conf_thresh}")
        
        start_time = time.time()
        
        # 평가 실행
        results, stats = evaluate_params(weights, top_k, conf_thresh, questions)
        
        elapsed = time.time() - start_time
        
        # 통계 출력
        print(f"  완료: {elapsed:.1f}초 소요")
        print(f"  topk 반환: {stats['with_topk']}/{stats['total']} ({stats['with_topk']/stats['total']*100:.1f}%)")
        print(f"  평균 처리 시간: {stats['avg_time']:.2f}초/문항")
        
        # MAP 계산 (정답 데이터가 있을 경우)
        # map_score = calculate_map(results, ground_truth)
        # print(f"  추정 MAP: {map_score:.4f}")
        
        # 결과 저장
        all_results.append({
            'params': {
                'voting_weights': weights,
                'top_k_retrieve': top_k,
                'confidence_threshold': conf_thresh
            },
            'stats': stats,
            'elapsed': elapsed,
            # 'map_score': map_score
        })
        
        print()
    
    # 결과 정렬 및 출력
    print("=" * 80)
    print("그리드 서치 완료!")
    print("=" * 80)
    print()
    
    # topk 반환 비율로 정렬 (MAP 0점 방지 관점)
    # 실제로는 MAP 점수가 있다면 그것으로 정렬
    all_results.sort(key=lambda x: x['stats']['with_topk'], reverse=True)
    
    print("상위 5개 조합:")
    print()
    
    for i, result in enumerate(all_results[:5], 1):
        params = result['params']
        stats = result['stats']
        
        print(f"{i}. VOTING_WEIGHTS={params['voting_weights']}, "
              f"TOP_K={params['top_k_retrieve']}, "
              f"CONF={params['confidence_threshold']}")
        print(f"   topk 반환: {stats['with_topk']}/{stats['total']} ({stats['with_topk']/stats['total']*100:.1f}%)")
        print(f"   처리 시간: {result['elapsed']:.1f}초")
        # print(f"   추정 MAP: {result['map_score']:.4f}")
        print()
    
    # 결과 저장
    with open('grid_search_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 전체 결과가 grid_search_results.json에 저장되었습니다.")
    print()
    
    # 최적 파라미터 추천
    best = all_results[0]
    print("=" * 80)
    print("추천 파라미터 (전체 평가 권장):")
    print("=" * 80)
    print(f"VOTING_WEIGHTS = {best['params']['voting_weights']}")
    print(f"TOP_K_RETRIEVE = {best['params']['top_k_retrieve']}")
    print(f"CONFIDENCE_THRESHOLD = {best['params']['confidence_threshold']}")
    print()
    print("위 설정을 eval_rag.py에 적용 후 전체 220개로 평가하세요!")


if __name__ == "__main__":
    run_grid_search()
