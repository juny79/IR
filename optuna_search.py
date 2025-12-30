"""
Smart Parameter Optimization using Optuna
베이지안 최적화로 효율적인 하이퍼파라미터 탐색
"""

import json
import time
import optuna
from eval_rag import answer_question_optimized

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_SIZE = 50  # 빠른 평가용 샘플
N_TRIALS = 20  # Optuna 시도 횟수

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_sample_data(sample_size=50):
    """샘플 데이터 로드"""
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


def evaluate_params(voting_weights, top_k_retrieve, confidence_threshold, questions):
    """파라미터 조합 평가"""
    import eval_rag
    
    # 파라미터 동적 변경
    eval_rag.VOTING_WEIGHTS = voting_weights
    eval_rag.TOP_K_RETRIEVE = top_k_retrieve
    eval_rag.CONFIDENCE_THRESHOLD = confidence_threshold
    
    with_topk = 0
    total_time = 0.0
    
    for q in questions:
        start_time = time.time()
        try:
            result = answer_question_optimized(q['msg'])
            if result['topk']:
                with_topk += 1
        except Exception as e:
            pass
        total_time += time.time() - start_time
    
    # 목표: topk 반환 비율 최대화 (83% 목표 - Phase 4D 수준)
    # 실제 MAP 점수가 없으므로 대리 지표 사용
    topk_ratio = with_topk / len(questions)
    avg_time = total_time / len(questions)
    
    # 스코어: topk 비율 중시 (처리 시간 패널티)
    score = topk_ratio - (avg_time / 100.0)  # 시간 패널티 추가
    
    return score, topk_ratio, avg_time


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optuna Objective 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def objective(trial, questions):
    """
    Optuna가 최적화할 목적 함수
    """
    # 파라미터 제안
    weight_sparse = trial.suggest_int('weight_sparse', 4, 6)
    weight_sbert = trial.suggest_int('weight_sbert', 3, 5)
    weight_gemini = trial.suggest_int('weight_gemini', 1, 3)
    
    top_k_retrieve = trial.suggest_int('top_k_retrieve', 40, 80, step=10)
    confidence_threshold = trial.suggest_float('confidence_threshold', 0.6, 0.8, step=0.05)
    
    voting_weights = [weight_sparse, weight_sbert, weight_gemini]
    
    # 평가
    score, topk_ratio, avg_time = evaluate_params(
        voting_weights, 
        top_k_retrieve, 
        confidence_threshold, 
        questions
    )
    
    # 추가 정보 로깅
    trial.set_user_attr('topk_ratio', topk_ratio)
    trial.set_user_attr('avg_time', avg_time)
    trial.set_user_attr('voting_weights', voting_weights)
    
    print(f"  Params: weights={voting_weights}, top_k={top_k_retrieve}, conf={confidence_threshold}")
    print(f"  Result: topk_ratio={topk_ratio:.2%}, time={avg_time:.2f}s, score={score:.4f}")
    
    return score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 최적화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_optimization():
    """
    Optuna 베이지안 최적화 실행
    """
    print("=" * 80)
    print("Smart Parameter Optimization (Optuna)")
    print("=" * 80)
    print(f"샘플 크기: {SAMPLE_SIZE}개")
    print(f"시도 횟수: {N_TRIALS}회")
    print()
    
    # 샘플 데이터 로드
    print("샘플 데이터 로드 중...")
    questions = load_sample_data(SAMPLE_SIZE)
    print(f"✅ {len(questions)}개 질문 로드 완료")
    print()
    
    # Optuna Study 생성
    study = optuna.create_study(
        direction='maximize',
        study_name='ir_hyperparameter_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 최적화 실행
    print(f"베이지안 최적화 시작 ({N_TRIALS}회 시도)...")
    print()
    
    study.optimize(
        lambda trial: objective(trial, questions), 
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    # 결과 출력
    print()
    print("=" * 80)
    print("최적화 완료!")
    print("=" * 80)
    print()
    
    best_trial = study.best_trial
    
    print(f"최고 점수: {best_trial.value:.4f}")
    print(f"topk 반환 비율: {best_trial.user_attrs['topk_ratio']:.2%}")
    print(f"평균 처리 시간: {best_trial.user_attrs['avg_time']:.2f}초")
    print()
    
    print("최적 파라미터:")
    print("-" * 80)
    weights = best_trial.user_attrs['voting_weights']
    print(f"VOTING_WEIGHTS = {weights}")
    print(f"TOP_K_RETRIEVE = {best_trial.params['top_k_retrieve']}")
    print(f"CONFIDENCE_THRESHOLD = {best_trial.params['confidence_threshold']}")
    print()
    
    # 상위 5개 조합 출력
    print("상위 5개 조합:")
    print("-" * 80)
    
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False)
    
    for i, (idx, row) in enumerate(trials_df.head(5).iterrows(), 1):
        print(f"{i}. Score: {row['value']:.4f}")
        print(f"   weights=[{int(row['params_weight_sparse'])}, {int(row['params_weight_sbert'])}, {int(row['params_weight_gemini'])}], "
              f"top_k={int(row['params_top_k_retrieve'])}, "
              f"conf={row['params_confidence_threshold']:.2f}")
        print(f"   topk_ratio: {row['user_attrs_topk_ratio']:.2%}, "
              f"time: {row['user_attrs_avg_time']:.2f}s")
        print()
    
    # 결과 저장
    trials_df.to_csv('optuna_results.csv', index=False)
    print(f"✅ 전체 결과가 optuna_results.csv에 저장되었습니다.")
    print()
    
    # eval_rag.py 업데이트 가이드
    print("=" * 80)
    print("다음 단계:")
    print("=" * 80)
    print("1. eval_rag.py에 위 최적 파라미터 적용")
    print("2. 전체 220개 문항으로 평가 실행")
    print("3. 리더보드 제출 및 MAP 점수 확인")
    print()
    print("복사할 코드:")
    print("-" * 80)
    print(f"VOTING_WEIGHTS = {weights}")
    print(f"TOP_K_RETRIEVE = {best_trial.params['top_k_retrieve']}")
    print(f"CONFIDENCE_THRESHOLD = {best_trial.params['confidence_threshold']}")


if __name__ == "__main__":
    try:
        import optuna
        run_optimization()
    except ImportError:
        print("❌ Optuna가 설치되지 않았습니다.")
        print("설치: pip install optuna")
        print()
        print("또는 grid_search.py를 사용하세요 (그리드 서치 방식)")
