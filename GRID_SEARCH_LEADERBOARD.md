# Leaderboard Grid Search Plan (MAP/MRR)

리더보드 점수가 유의미하게 변했던 경험(TopK/후보군 50→80) 기준으로, **비용을 제어하면서** 그리드서치 형태로 후보 파라미터를 찾는 방법입니다.

## 전제
- 로컬에는 정답(ground truth)이 없어 MAP/MRR를 정확히 계산할 수 없습니다.
- 따라서 로컬은 **프리필터(프록시 지표)**만 하고, 최종 평가는 리더보드로 합니다.

## 추천 변수(유의미한 변화 가능성 높은 순)
1) `TOP_K_RETRIEVE` (초기 후보 풀 크기)
2) `CANDIDATE_POOL_SIZE` (리랭커에 들어가는 후보 수)
3) `VOTING_WEIGHTS_JSON` (Hard Voting의 rank1/2/3 가중치)
4) `HYDE_MAX_LENGTH` (Sparse HyDE 길이)

## 운영 방식(2단계)
### 1) Dry run(프리필터)
- 목표: 비용을 줄이고 “나쁜 조합”을 빠르게 제거
- 방법: `EVAL_LIMIT=50` 같은 제한으로 부분 생성
- 산출: 실행시간/empty_topk_count/베이스라인 대비 overlap(프록시)

### 2) Full run(리더보드 제출용)
- Dry run에서 상위 2~4개만 뽑아 **220개 전체 생성 후 제출**

## 자동화 스크립트
- 파일: `experiments/leaderboard_grid_search.py`

### 실행
- Dry:
  - `python3 experiments/leaderboard_grid_search.py --mode dry`
- Full:
  - `python3 experiments/leaderboard_grid_search.py --mode full --timeout 1800`

### TOP3만 선택해서 Full로 돌리기(프록시 기반)
- 먼저 Dry를 돌려 `experiments/results.jsonl`을 만든 뒤,
- 아래처럼 결과 파일에서 `proxy_overlap_vs_baseline.avg_top5_overlap` 기준 TOP3만 골라 Full을 생성:
  - `python3 experiments/leaderboard_grid_search.py --mode full --from-results experiments/results.jsonl --top-n 3 --metric proxy_overlap_vs_baseline.avg_top5_overlap --prefer-mode dry`

### 결과
- 결과 요약: `experiments/results.jsonl`
- 각 실험 산출물:
  - `experiments/out/<exp_id>/submission.jsonl`
  - `experiments/out/<exp_id>/run.log`

## Baseline 비교(프록시)
- baseline 제출물(예: `submission_best_map08765.csv`)과의
  - `top1_same_ratio`
  - `avg_top5_overlap`

주의: overlap이 높다고 MAP가 반드시 오르는 건 아니지만, **대규모 성능 붕괴 조합을 제거**하는 데는 유용합니다.

## 지금 바로 추천하는 다음 3개 Full 실험
- A: `TOP_K_RETRIEVE=100`, `CANDIDATE_POOL_SIZE=100` (현재 상승 트렌드 연장)
- B: `TOP_K_RETRIEVE=120`, `CANDIDATE_POOL_SIZE=120` (더 넓은 후보)
- C: `TOP_K_RETRIEVE=80`, `CANDIDATE_POOL_SIZE=80`, `VOTING_WEIGHTS_JSON=[6,4,2]` (rank1 bias 강화)
