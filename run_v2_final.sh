#!/bin/bash
set -e

# SOTA(v9) 기반 설정 + V2 Reranker
export USE_RRF=true
export RRF_K=20
export USE_MULTI_QUERY=true
export TOP_K_RETRIEVE=100
export CANDIDATE_POOL_SIZE=100
export USE_V2_RERANKER=true
export SUBMISSION_FILE="submission_v2_final_rerank.csv"

echo "Starting V2 Final Evaluation..."
python3 main.py

echo "Done! Result saved to $SUBMISSION_FILE"
