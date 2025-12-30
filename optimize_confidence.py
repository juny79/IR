"""
Confidence Threshold 최적화
과학지식 vs 일상대화 구분의 핵심 임계치 탐색
"""

import json
import time
from eval_rag import answer_question_optimized
import eval_rag

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_SIZE = 50  # 빠른 평가용
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]  # 7개 후보

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_sample_questions(sample_size=50):
    """샘플 질문 로드"""
    questions = []
    with open('./data/eval.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            data = json.loads(line)
            questions.append({
                'eval_id': data['eval_id'],
                'msg': data['msg']
            })
    return questions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_threshold(threshold, questions):
    """
    특정 confidence threshold 평가
    
    Returns:
        (with_topk, filtered, avg_time, error_count)
    """
    # eval_rag.py의 CONFIDENCE_THRESHOLD 동적 변경
    eval_rag.CONFIDENCE_THRESHOLD = threshold
    
    with_topk = 0
    filtered = 0
    total_time = 0.0
    error_count = 0
    
    for q in questions:
        start_time = time.time()
        try:
            result = answer_question_optimized([{'role': 'user', 'content': q['msg']}])
            if result['topk']:
                with_topk += 1
            else:
                filtered += 1
        except Exception as e:
            error_count += 1
            filtered += 1
        
        total_time += time.time() - start_time
    
    avg_time = total_time / len(questions)
    
    return with_topk, filtered, avg_time, error_count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 최적화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_optimization():
    """Confidence threshold 최적화"""
    print("=" * 80)
    print("🎯 Confidence Threshold 최적화")
    print("=" * 80)
    print(f"샘플 크기: {SAMPLE_SIZE}개")
    print(f"테스트 임계치: {THRESHOLDS}")
    print()
    print("목표:")
    print("  - 과학지식은 검색 수행 (topk 반환)")
    print("  - 일상대화는 필터링 (topk 빈 리스트)")
    print("  - Phase 4D: 83% topk 반환 (183/220) ← 이 정도 유지가 목표")
    print()
    
    # 샘플 데이터 로드
    print("샘플 데이터 로드 중...")
    questions = load_sample_questions(SAMPLE_SIZE)
    print(f"✅ {len(questions)}개 질문 로드 완료")
    print()
    
    # 각 임계치 평가
    results = []
    
    print("=" * 80)
    print("임계치 평가 시작")
    print("=" * 80)
    print()
    
    for i, threshold in enumerate(THRESHOLDS, 1):
        print(f"[{i}/{len(THRESHOLDS)}] Threshold: {threshold}")
        print("-" * 80)
        
        start = time.time()
        with_topk, filtered, avg_time, errors = evaluate_threshold(threshold, questions)
        duration = time.time() - start
        
        topk_ratio = with_topk / len(questions)
        filter_ratio = filtered / len(questions)
        
        results.append({
            'threshold': threshold,
            'with_topk': with_topk,
            'filtered': filtered,
            'topk_ratio': topk_ratio,
            'filter_ratio': filter_ratio,
            'avg_time': avg_time,
            'errors': errors,
            'total_time': duration
        })
        
        print(f"  검색 수행: {with_topk}/{len(questions)} ({topk_ratio:.1%})")
        print(f"  필터링됨: {filtered}/{len(questions)} ({filter_ratio:.1%})")
        print(f"  평균 시간: {avg_time:.2f}초")
        print(f"  에러 수: {errors}")
        print(f"  소요 시간: {duration:.1f}초")
        print()
    
    # 결과 분석
    print()
    print("=" * 80)
    print("결과 요약")
    print("=" * 80)
    print()
    
    # Phase 4D 기준: 83% (183/220) topk 반환
    target_ratio = 0.83
    
    print(f"{'Threshold':<12} {'검색 수행':<12} {'필터링':<12} {'비율':<10} {'평균시간':<10}")
    print("-" * 80)
    
    for r in results:
        marker = ""
        # Phase 4D 기준 (83%) 근처면 ⭐ 표시
        if abs(r['topk_ratio'] - target_ratio) < 0.03:  # ±3% 범위
            marker = "⭐"
        elif r['topk_ratio'] > target_ratio:
            marker = "↑"  # 더 많이 검색
        else:
            marker = "↓"  # 더 적게 검색
        
        print(f"{r['threshold']:<12.2f} {r['with_topk']:<12} {r['filtered']:<12} "
              f"{r['topk_ratio']:<10.1%} {r['avg_time']:<10.2f} {marker}")
    
    print()
    print("-" * 80)
    print(f"⭐ Phase 4D 기준: 83% (183/220) 검색 수행")
    print(f"↑ 더 많이 검색 (over-searching)")
    print(f"↓ 더 적게 검색 (over-filtering)")
    print()
    
    # 최적 임계치 추천
    # 목표: Phase 4D와 비슷한 topk 비율 (83%)
    best = min(results, key=lambda x: abs(x['topk_ratio'] - target_ratio))
    
    print()
    print("=" * 80)
    print("✅ 추천 임계치")
    print("=" * 80)
    print()
    print(f"CONFIDENCE_THRESHOLD = {best['threshold']}")
    print()
    print(f"예상 결과:")
    print(f"  - 검색 수행: {best['topk_ratio']:.1%} ({best['with_topk']}/{len(questions)})")
    print(f"  - 필터링: {best['filter_ratio']:.1%} ({best['filtered']}/{len(questions)})")
    print(f"  - 평균 처리 시간: {best['avg_time']:.2f}초")
    print()
    print(f"Phase 4D와 차이: {abs(best['topk_ratio'] - target_ratio):.1%}p")
    print()
    
    # 민감도 분석
    print("=" * 80)
    print("📊 민감도 분석")
    print("=" * 80)
    print()
    print("임계치 변화에 따른 검색 비율:")
    for i in range(len(results) - 1):
        curr = results[i]
        next_ = results[i + 1]
        delta_threshold = next_['threshold'] - curr['threshold']
        delta_ratio = next_['topk_ratio'] - curr['topk_ratio']
        
        print(f"  {curr['threshold']:.2f} → {next_['threshold']:.2f}: "
              f"{delta_ratio:+.1%}p (민감도: {delta_ratio/delta_threshold:+.1f}%/0.01)")
    print()
    
    # 결과 저장
    with open('confidence_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'sample_size': SAMPLE_SIZE,
            'target_ratio': target_ratio,
            'best_threshold': best['threshold'],
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print("✅ 결과가 confidence_optimization_results.json에 저장되었습니다.")
    print()
    
    # 다음 단계
    print("=" * 80)
    print("다음 단계")
    print("=" * 80)
    print()
    print("1. eval_rag.py 수정:")
    print(f"   CONFIDENCE_THRESHOLD = {best['threshold']}")
    print()
    print("2. 전체 평가 실행:")
    print("   python main.py")
    print()
    print("3. 리더보드 제출 및 MAP 점수 확인")
    print()


if __name__ == "__main__":
    try:
        run_optimization()
    except KeyboardInterrupt:
        print("\n\n중단되었습니다.")
    except Exception as e:
        print(f"\n에러 발생: {e}")
        import traceback
        traceback.print_exc()
