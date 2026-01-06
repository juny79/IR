#!/bin/bash

# Phase 4D 최적화 시리즈 자동 실행 스크립트

echo "========================================"
echo "Phase 4D 최적화 테스트 시작"
echo "========================================"

phases=(
    "Phase 4D-NoGating"
    "Phase 4D-TopK60"
    "Phase 4D-Weight[5,5,2]"
)

log_files=(
    "phase_4d_nogating_evaluation.log"
    "phase_4d_topk60_evaluation.log"
    "phase_4d_weight552_evaluation.log"
)

configs=(
    "eval_rag.py"
    "eval_rag_topk60.py"
    "eval_rag_weight552.py"
)

# 각 평가 실행
for i in {0..0}; do
    echo ""
    echo "=========================================="
    echo "${phases[$i]} 실행"
    echo "=========================================="
    
    # 현재 평가가 진행 중인지 확인
    if pgrep -f "python3 main.py" > /dev/null; then
        echo "아직 이전 평가가 진행 중입니다. 대기..."
        while pgrep -f "python3 main.py" > /dev/null; do
            sleep 30
        done
    fi
    
    # 설정 파일 복사
    #cp ${configs[$i]} eval_rag.py
    
    # 평가 실행
    rm -f submission.csv
    timeout 2000 python3 main.py > ${log_files[$i]} 2>&1 &
    
    # 프로세스 ID 저장
    pid=$!
    echo "프로세스 ID: $pid"
    
    # 완료 대기
    wait $pid
    
    echo "${phases[$i]} 완료"
    
    # 결과 저장
    cp submission.csv submission_${i}.csv
done

echo ""
echo "=========================================="
echo "모든 테스트 완료"
echo "=========================================="
