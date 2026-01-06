#!/bin/bash

# Base settings
export ALPHA=0.5
export SKIP_ANSWER=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Variant 1: LLM Rerank Top 3 ON
echo "Starting Variant 1: LLM Rerank Top 3 ON"
export TOP_CANDIDATES=100
export USE_MULTI_QUERY=true
export USE_LLM_RERANK_TOP3=true
export SUBMISSION_FILE="submission_grid_v1_llm_on_${TIMESTAMP}.csv"
python3 /root/IR/eval_rag_bge_m3_v6.py

# Variant 2: Multi-Query OFF (Single Query)
echo "Starting Variant 2: Multi-Query OFF"
export TOP_CANDIDATES=100
export USE_MULTI_QUERY=false
export USE_LLM_RERANK_TOP3=false
export SUBMISSION_FILE="submission_grid_v2_mq_off_${TIMESTAMP}.csv"
python3 /root/IR/eval_rag_bge_m3_v6.py

# Variant 3: Top Candidates 200 (Higher Recall for Reranker)
echo "Starting Variant 3: Top Candidates 200"
export TOP_CANDIDATES=200
export USE_MULTI_QUERY=true
export USE_LLM_RERANK_TOP3=false
export SUBMISSION_FILE="submission_grid_v3_tk200_${TIMESTAMP}.csv"
python3 /root/IR/eval_rag_bge_m3_v6.py

echo "All variants completed."
