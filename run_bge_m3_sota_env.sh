#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ts=$(date +%Y%m%d_%H%M%S)
out="submission_ready_bge_m3_sota_${ts}.csv"
log="run_bge_m3_sota_${ts}.log"

echo "generating $out"
rm -f "$out"

# Reproduce the original BGE-M3 SOTA style settings:
# - Unified retrieval: BGE-M3 dense + lexical sparse
# - Hybrid fusion: alpha=0.5
# - Rerank: top100 -> top10
# - Gating: EMPTY_IDS hardcoded inside eval_rag_bge_m3_v6.py
# Notes:
# - USE_LLM_RERANK_TOP3=false to match 'reranker만' 중심 세팅
# - SKIP_ANSWER=true because leaderboard seems retrieval-only (MAP=MRR) and it speeds up.
SUBMISSION_FILE="$out" \
TOP_CANDIDATES=100 \
FINAL_TOPK=10 \
ALPHA=0.5 \
RRF_K=60 \
USE_MULTI_QUERY=true \
USE_LLM_RERANK_TOP3=false \
SKIP_ANSWER=true \
python3 eval_rag_bge_m3_v6.py > "$log" 2>&1

echo "done"
wc -l "$out"
sha256sum "$out" | awk '{print $1}'

python3 - << PY
import json
from pathlib import Path
p=Path("$out")
empty=0
parse_errors=0
rows=0
for ln in p.read_text(encoding='utf-8',errors='ignore').splitlines():
    ln=ln.strip()
    if not ln:
        continue
    rows += 1
    try:
        o=json.loads(ln)
        if isinstance(o.get('topk'), list) and len(o['topk'])==0:
            empty += 1
    except Exception:
        parse_errors += 1
print('rows:', rows)
print('empty_topk_count:', empty)
print('parse_errors:', parse_errors)
PY
