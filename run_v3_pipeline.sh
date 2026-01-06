#!/bin/bash
set -e

echo ">>> [Phase 1] Synthetic QA Generation (Solar Pro 2)..."
# 이미 백그라운드에서 실행 중이라면 기다리거나, 새로 실행
# 여기서는 백그라운드 실행 중인 프로세스가 끝날 때까지 기다리는 로직을 넣거나
# 단순히 생성이 완료되었는지 체크하는 루프를 돌립니다.

while true; do
    TOTAL_DOCS=4272
    CURRENT_DOCS=$(wc -l < data/synthetic_qa_solar.jsonl 2>/dev/null || echo 0)
    echo "Progress: $CURRENT_DOCS / $TOTAL_DOCS documents processed..."
    
    if [ "$CURRENT_DOCS" -ge "$TOTAL_DOCS" ]; then
        echo "QA Generation Complete!"
        break
    fi
    sleep 300 # 5분마다 체크
done

echo ">>> [Phase 2] Hard Negative Mining (V3)..."
python3 finetune/2_mine_negatives_v3.py

echo ">>> [Phase 3] BGE-M3 V3 Fine-tuning..."
bash finetune/3_run_train_v3.sh

echo ">>> [Phase 4] Evaluation & Submission Generation..."
# V3 모델을 사용하는 평가 스크립트 실행 (기존 eval_rag_bge_m3_v2.py를 복사하여 수정 필요)
# 이 부분은 학습 완료 후 진행
