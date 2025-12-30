"""
Step 2: 캐싱 기반 그리드 서치 (Light Task)
- 캐싱된 검색 결과로 파라미터 시뮬레이션
- VOTING_WEIGHTS, RRF vs Hard Voting, confidence threshold 최적화
- Ground Truth가 없으므로 Top-1 일치율, 검색 수행률로 평가
"""

import json
from collections import defaultdict

# 캐시 로드
CACHE_FILE = 'search_results_cache.json'

def load_cache():
    """캐시 로드"""
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def hard_vote_simulation(sparse_docids, dense_docids_list, top_k=5, weights=[5, 4, 2]):
    """
    Hard Voting 시뮬레이션 (캐시 기반)
    
    Args:
        sparse_docids: Sparse 검색 결과 DocID 리스트
        dense_docids_list: Dense 검색 결과 리스트 (SBERT, Gemini)
        top_k: 반환할 문서 개수
        weights: [1위, 2위, 3위] 가중치
    
    Returns:
        Top-K DocID 리스트
    """
    scores = defaultdict(float)
    
    # Sparse 결과 처리
    for rank, docid in enumerate(sparse_docids, 1):
        if rank == 1:
            scores[docid] += weights[0]
        elif rank == 2 and len(weights) > 1:
            scores[docid] += weights[1]
        elif rank == 3 and len(weights) > 2:
            scores[docid] += weights[2]
        else:
            scores[docid] += 0.5  # 4위 이하
    
    # Dense 결과 처리
    for dense_docids in dense_docids_list:
        for rank, docid in enumerate(dense_docids, 1):
            if rank == 1:
                scores[docid] += weights[0]
            elif rank == 2 and len(weights) > 1:
                scores[docid] += weights[1]
            elif rank == 3 and len(weights) > 2:
                scores[docid] += weights[2]
            else:
                scores[docid] += 0.5
    
    # 점수 기준 정렬
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docid for docid, _ in sorted_docs[:top_k]]


def rrf_simulation(sparse_docids, dense_docids_list, top_k=5, k=60):
    """
    RRF (Reciprocal Rank Fusion) 시뮬레이션
    
    score(doc) = Σ(1 / (k + rank))
    """
    scores = defaultdict(float)
    
    # Sparse 결과
    for rank, docid in enumerate(sparse_docids, 1):
        scores[docid] += 1.0 / (k + rank)
    
    # Dense 결과
    for dense_docids in dense_docids_list:
        for rank, docid in enumerate(dense_docids, 1):
            scores[docid] += 1.0 / (k + rank)
    
    # 점수 기준 정렬
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docid for docid, _ in sorted_docs[:top_k]]


def evaluate_params(cache, confidence_threshold, voting_weights, use_rrf=False, rrf_k=60):
    """
    파라미터 조합 평가
    
    Returns:
        (with_topk_count, top1_consistency, avg_topk_size)
    """
    with_topk = 0
    top1_docs = []
    topk_sizes = []
    
    for eval_id, data in cache.items():
        # 에러 케이스 스킵
        if 'error' in data:
            continue
        
        # 게이팅 적용
        if not data.get('should_search', False):
            continue
        
        conf = data.get('confidence', 0.0)
        if conf < confidence_threshold:
            continue  # 필터링
        
        # 검색 결과 추출
        sparse = data.get('sparse_results', [])
        sbert = data.get('dense_sbert_results', [])
        gemini = data.get('dense_gemini_results', [])
        
        if not sparse and not sbert and not gemini:
            continue
        
        # 투표 알고리즘 적용
        if use_rrf:
            topk = rrf_simulation(sparse, [sbert, gemini], top_k=5, k=rrf_k)
        else:
            topk = hard_vote_simulation(sparse, [sbert, gemini], top_k=5, weights=voting_weights)
        
        if topk:
            with_topk += 1
            top1_docs.append(topk[0])
            topk_sizes.append(len(topk))
    
    # 통계
    total = len([v for v in cache.values() if 'error' not in v])
    topk_ratio = with_topk / total if total > 0 else 0
    avg_size = sum(topk_sizes) / len(topk_sizes) if topk_sizes else 0
    
    # Top-1 일치율 (Phase 4D 기준과 비교)
    # Ground Truth가 없으므로 "일관성" 지표로 사용
    top1_consistency = len(set(top1_docs)) / len(top1_docs) if top1_docs else 0
    
    return {
        'with_topk': with_topk,
        'total': total,
        'topk_ratio': topk_ratio,
        'avg_topk_size': avg_size,
        'top1_consistency': top1_consistency
    }


def grid_search():
    """
    그리드 서치 실행
    """
    print("=" * 80)
    print("Step 2: 캐싱 기반 그리드 서치")
    print("=" * 80)
    print()
    
    # 캐시 로드
    print("캐시 로드 중...")
    cache = load_cache()
    print(f"✅ {len(cache)}개 질문 로드 완료")
    print()
    
    # 파라미터 그리드
    # Gemini 3 제안: Top-3 집중 전략
    voting_weights_grid = [
        [5, 4, 2],   # Phase 4D 기준
        [6, 4, 2],   # 1위 강화
        [7, 4, 2],   # 1위 더 강화
        [5, 5, 2],   # 2위 강화
        [5, 4, 3],   # 3위 강화
        [6, 5, 3],   # 전체 강화
    ]
    
    confidence_thresholds = [0.55, 0.60, 0.65, 0.70]  # 0.55가 최적이지만 주변 탐색
    
    # RRF vs Hard Voting
    algorithms = [
        {'name': 'Hard Voting', 'use_rrf': False},
        {'name': 'RRF (k=60)', 'use_rrf': True, 'rrf_k': 60},
        {'name': 'RRF (k=80)', 'use_rrf': True, 'rrf_k': 80},
    ]
    
    print("그리드 설정:")
    print(f"  - Voting Weights: {len(voting_weights_grid)}개")
    print(f"  - Confidence Threshold: {len(confidence_thresholds)}개")
    print(f"  - Algorithms: {len(algorithms)}개")
    print(f"  - 총 조합: {len(voting_weights_grid) * len(confidence_thresholds) * len(algorithms)}개")
    print()
    
    # 그리드 서치
    results = []
    
    print("=" * 80)
    print("그리드 서치 시작")
    print("=" * 80)
    print()
    
    for algo in algorithms:
        for conf_th in confidence_thresholds:
            for weights in voting_weights_grid:
                # 평가
                metrics = evaluate_params(
                    cache, 
                    conf_th, 
                    weights, 
                    use_rrf=algo['use_rrf'],
                    rrf_k=algo.get('rrf_k', 60)
                )
                
                result = {
                    'algorithm': algo['name'],
                    'confidence_threshold': conf_th,
                    'voting_weights': weights,
                    'rrf_k': algo.get('rrf_k', None),
                    **metrics
                }
                
                results.append(result)
                
                print(f"{algo['name']:20s} | conf={conf_th:.2f} | w={weights} | "
                      f"topk={metrics['topk_ratio']:.1%} ({metrics['with_topk']}/{metrics['total']})")
    
    print()
    print("=" * 80)
    print("그리드 서치 완료!")
    print("=" * 80)
    print()
    
    # 결과 정렬 (Phase 4D 기준 83% 근처 + Top-1 일관성 높은 순)
    target_ratio = 0.83
    
    # 목표: topk_ratio가 83%에 가까우면서, Top-1 일관성 높은 것
    results_sorted = sorted(
        results, 
        key=lambda x: (
            abs(x['topk_ratio'] - target_ratio),  # Phase 4D와 유사도 (작을수록 좋음)
            -x['topk_ratio']  # 검색 수행률 (높을수록 좋음)
        )
    )
    
    # 상위 10개 출력
    print("상위 10개 파라미터 조합:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Algorithm':<20} {'Conf':<6} {'Weights':<15} {'검색률':<10} {'차이':<10}")
    print("-" * 80)
    
    for i, r in enumerate(results_sorted[:10], 1):
        diff = abs(r['topk_ratio'] - target_ratio)
        print(f"{i:<5} {r['algorithm']:<20} {r['confidence_threshold']:<6.2f} "
              f"{str(r['voting_weights']):<15} {r['topk_ratio']:<10.1%} {diff:<10.1%}")
    
    print()
    
    # 최적 파라미터
    best = results_sorted[0]
    
    print("=" * 80)
    print("✅ 최적 파라미터 (Phase 4D 기준 83%와 가장 유사)")
    print("=" * 80)
    print()
    print(f"Algorithm: {best['algorithm']}")
    print(f"CONFIDENCE_THRESHOLD = {best['confidence_threshold']}")
    print(f"VOTING_WEIGHTS = {best['voting_weights']}")
    if best['rrf_k']:
        print(f"RRF_K = {best['rrf_k']}")
        print(f"USE_RRF = True")
    else:
        print(f"USE_RRF = False")
    print()
    print(f"예상 검색 수행률: {best['topk_ratio']:.1%} ({best['with_topk']}/{best['total']})")
    print(f"Phase 4D와 차이: {abs(best['topk_ratio'] - target_ratio):.1%}p")
    print()
    
    # 결과 저장
    with open('grid_search_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': best,
            'top_10': results_sorted[:10],
            'all_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print("✅ 결과가 grid_search_results.json에 저장되었습니다.")
    print()
    
    # 다음 단계
    print("=" * 80)
    print("다음 단계")
    print("=" * 80)
    print()
    print("1. eval_rag.py에 최적 파라미터 적용:")
    print(f"   CONFIDENCE_THRESHOLD = {best['confidence_threshold']}")
    print(f"   VOTING_WEIGHTS = {best['voting_weights']}")
    if best['rrf_k']:
        print(f"   USE_RRF = True")
        print(f"   RRF_K = {best['rrf_k']}")
    else:
        print(f"   USE_RRF = False")
    print()
    print("2. 전체 평가 실행:")
    print("   python main.py")
    print()
    print("3. 리더보드 제출 및 MAP 점수 확인")
    print()


if __name__ == "__main__":
    grid_search()
