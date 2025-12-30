#!/usr/bin/env python3
"""MVP auditing framework for the IR pipeline.

Goal
- No local ground-truth labels available.
- For a small eval slice (default 50), decompose the pipeline into modules and
  write per-eval_id diagnostic records as JSONL.

Input
- eval.jsonl (same schema as ./data/eval.jsonl)

Output (JSONL)
- One line per eval_id with:
  - A/B/C module scores (0~1)
  - Evidence (gating decision, retrieval signals, agreement)
  - Failure-type tags
  - Next-experiment actions with runnable commands

This script is intentionally heuristic/proxy-based.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a standalone script: `python3 auditing/audit_mvp.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.solar_client import solar_client
from retrieval.es_connector import sparse_retrieve, dense_retrieve
from retrieval.hybrid_search import hard_vote_results, rrf_fusion, get_documents_batch


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _last_user_text(messages: Any) -> str:
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
    return str(messages)


def _extract_docids(res: Dict[str, Any]) -> List[str]:
    hits = (((res or {}).get("hits") or {}).get("hits") or [])
    out: List[str] = []
    for h in hits:
        try:
            out.append(h["_source"]["docid"])
        except Exception:
            continue
    return out


def _top_score(res: Dict[str, Any]) -> float:
    hits = (((res or {}).get("hits") or {}).get("hits") or [])
    if not hits:
        return 0.0
    try:
        return float(hits[0].get("_score") or 0.0)
    except Exception:
        return 0.0


def _jaccard_topk(a: List[str], b: List[str], k: int = 5) -> float:
    sa = set(a[:k])
    sb = set(b[:k])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _contains_any(text: str, needles: List[str]) -> bool:
    t = (text or "").lower()
    return any(n.lower() in t for n in needles)


def _is_korean_chatty(messages: Any) -> bool:
    # strict chat-only markers to avoid false positives on knowledge queries
    q = _last_user_text(messages)
    patterns = [
        r"^(안녕|안녕하세요)\b",
        r"\b반가(워|워요)?\b",
        r"\b고마(워|워요|워!)\b",
        r"\b감사(해|합니다|해요|!)\b",
        r"\b우울\b",
        r"\b힘들(어|다|어요)?\b",
        r"\b기분(이|은)?\b",
        r"\b즐거웠\b",
        r"\b신나\b",
        r"\bㅋㅋ+\b|\bㅎㅎ+\b",
        r"너는 누구(야|니)?",
        r"너 누구(야|니)?",
        r"너 정말 똑똑",
        r"너 모르는 것도",
        r"너 (뭘|뭐) 잘해|너 잘하는(게|거)",
        r"이제 그만",
        r"그만 얘기",
    ]
    rx = re.compile("|".join(f"(?:{p})" for p in patterns))
    return bool(rx.search(q))


def _no_search_reason(messages: Any) -> Tuple[str, str]:
    """Heuristic categorization for 'no-search ground truth' style cases."""
    q = _last_user_text(messages)
    q_norm = (q or "").strip()

    if _is_korean_chatty(messages):
        # further sub-classify
        if _contains_any(q_norm, ["우울", "힘들", "기분", "즐거", "신나"]):
            return "CHAT_EMOTION", "emotional / small-talk"
        if _contains_any(q_norm, ["너는 누구", "너 누구", "뭘 잘해", "뭐 잘해", "잘하는게"]):
            return "CHAT_SELF", "assistant self-intro / capability chat"
        if _contains_any(q_norm, ["안녕", "반가", "고마", "감사", "ㅋㅋ", "ㅎㅎ", "그만"]):
            return "CHAT_MISC", "greeting / casual chat / stop-talk"
        return "CHAT", "chatty marker matched"

    # Dynamic / statistics / up-to-date status: often OOD vs static corpus
    if _contains_any(
        q_norm,
        [
            "현황",
            "통계",
            "비율",
            "순위",
            "최근",
            "요즘",
            "올해",
            "작년",
            "내년",
            "현재",
            "나라별",
            "각 나라",
            "지역별",
            "가격",
            "주가",
            "환율",
            "날씨",
        ],
    ):
        return "OOD_DYNAMIC_STATS", "likely requires up-to-date or per-country stats"

    # Opinion / preference / recommendation: may be no-search in this benchmark
    if _contains_any(q_norm, ["추천", "취향", "생각", "의견", "어때", "좋을까", "골라", "비교해줘"]):
        return "NOSEARCH_OPINION", "opinion/recommendation style"

    # Creative requests
    if _contains_any(q_norm, ["시 써", "소설", "이야기", "농담", "드립", "대사", "각본", "역할극"]):
        return "NOSEARCH_CREATIVE", "creative generation request"

    return "UNKNOWN", "no strong no-search markers"


@dataclass
class AuditConfig:
    eval_path: Path
    out_path: Path
    limit: int
    top_k_retrieve: int
    candidate_pool_size: int
    hyde_max_length: int
    use_multi_embedding: bool
    use_gemini_only: bool
    use_rrf: bool
    rrf_k: int
    voting_weights: List[int]
    no_search_conf_threshold: float
    use_reranker: bool


def _parse_args() -> AuditConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", dest="eval_path", default="./data/eval.jsonl")
    ap.add_argument("--out", dest="out_path", default="")
    ap.add_argument("--limit", type=int, default=50)

    ap.add_argument("--top-k-retrieve", type=int, default=int(os.getenv("TOP_K_RETRIEVE", "80")))
    ap.add_argument("--candidate-pool-size", type=int, default=int(os.getenv("CANDIDATE_POOL_SIZE", "80")))
    ap.add_argument("--hyde-max-length", type=int, default=int(os.getenv("HYDE_MAX_LENGTH", "200")))

    ap.add_argument(
        "--use-multi-embedding",
        action="store_true",
        default=(os.getenv("USE_MULTI_EMBEDDING", "true").lower() in ("1", "true", "yes", "y", "on")),
    )
    ap.add_argument(
        "--use-gemini-only",
        action="store_true",
        default=(os.getenv("USE_GEMINI_ONLY", "false").lower() in ("1", "true", "yes", "y", "on")),
    )

    ap.add_argument("--use-rrf", action="store_true", default=(os.getenv("USE_RRF", "false").lower() in ("1", "true", "yes", "y", "on")))
    ap.add_argument("--rrf-k", type=int, default=int(os.getenv("RRF_K", "60")))

    ap.add_argument(
        "--voting-weights-json",
        default=os.getenv("VOTING_WEIGHTS_JSON", "[5,4,2]"),
        help="JSON list, e.g. [5,4,2]",
    )

    ap.add_argument(
        "--no-search-conf-threshold",
        type=float,
        default=float(os.getenv("NO_SEARCH_CONFIDENCE_THRESHOLD", "0.0")),
    )
    ap.add_argument("--no-reranker", dest="use_reranker", action="store_false", default=True)

    args = ap.parse_args()

    try:
        voting_weights = json.loads(args.voting_weights_json)
        if not (isinstance(voting_weights, list) and all(isinstance(x, int) for x in voting_weights)):
            voting_weights = [5, 4, 2]
    except Exception:
        voting_weights = [5, 4, 2]

    out_path = Path(args.out_path) if args.out_path else Path(f"./auditing/audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    return AuditConfig(
        eval_path=Path(args.eval_path),
        out_path=out_path,
        limit=max(1, int(args.limit)),
        top_k_retrieve=max(5, int(args.top_k_retrieve)),
        candidate_pool_size=max(5, int(args.candidate_pool_size)),
        hyde_max_length=max(0, int(args.hyde_max_length)),
        use_multi_embedding=bool(args.use_multi_embedding),
        use_gemini_only=bool(args.use_gemini_only),
        use_rrf=bool(args.use_rrf),
        rrf_k=max(1, int(args.rrf_k)),
        voting_weights=voting_weights,
        no_search_conf_threshold=float(args.no_search_conf_threshold),
        use_reranker=bool(args.use_reranker),
    )


def _make_actions(base_env: Dict[str, str]) -> List[Dict[str, str]]:
    # “버튼”은 UI 대신 실행 가능한 커맨드로 제공
    # (리더보드 제출용 full run은 시간이 오래 걸리니, 기본은 EVAL_LIMIT=50으로 빠른 검증)
    def cmd(extra_env: Dict[str, str]) -> str:
        env = {**base_env, **extra_env}
        parts: List[str] = []
        for k, v in env.items():
            sv = str(v)
            if " " in sv or "\t" in sv:
                parts.append(f"{k}={json.dumps(sv)}")
            else:
                parts.append(f"{k}={sv}")
        return " ".join(parts) + " python3 main.py"

    return [
        {"label": "TopK↑", "command": cmd({"TOP_K_RETRIEVE": "120"})},
        {"label": "HyDE↓", "command": cmd({"HYDE_MAX_LENGTH": "120"})},
        {"label": "weights 변경", "command": cmd({"VOTING_WEIGHTS_JSON": "[6,4,2]"})},
        {"label": "gating 수정", "command": cmd({"NO_SEARCH_CONFIDENCE_THRESHOLD": "0.9"})},
        {"label": "RRF 켜기", "command": cmd({"USE_RRF": "true", "RRF_K": "20"})},
        {"label": "MultiQuery 켜기", "command": cmd({"USE_MULTI_QUERY": "true"})},
    ]


def _recommended_action_labels(tags: List[str]) -> List[str]:
    """Map tags -> recommended experiment knobs (labels must exist in actions)."""
    t = set(tags or [])

    if any(x.startswith("NOSEARCH_") or x.startswith("CHAT_") for x in t):
        # no-search cases: tune gating threshold or OOD rules
        return ["gating 수정"]

    if "SEARCH_BUT_WEAK_SUPPORT" in t or "SEARCH_LIKELY_OOD" in t:
        # may be OOD, or needs better recall
        return ["gating 수정", "TopK↑"]

    if "SPARSE_DENSE_DIVERGENCE" in t:
        return ["RRF 켜기", "TopK↑"]

    if "RERANKER_FLIP_HEAVY" in t:
        # When reranker strongly disagrees, candidate diversity/recall knobs are usually higher leverage
        return ["TopK↑", "RRF 켜기"]

    return []


def audit_one(messages: Any, cfg: AuditConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (audit_record, debug)."""
    original_query = _last_user_text(messages)

    solar = solar_client.analyze_query_and_hyde(messages, hyde_max_chars=cfg.hyde_max_length)
    is_science = bool(solar.get("is_science", False))
    solar_conf = _safe_float(solar.get("confidence", 0.0), 0.0)

    standalone_query = (solar.get("standalone_query") or original_query).strip()
    hyde = (solar.get("hyde") or "").strip()

    sparse_query = standalone_query
    if hyde:
        sparse_query = f"{standalone_query}\n{hyde}"

    # Shadow retrieval signals (always compute for auditing)
    sparse_res = sparse_retrieve(sparse_query, cfg.top_k_retrieve)

    dense_results: List[Dict[str, Any]] = []
    if cfg.use_gemini_only:
        try:
            dense_results.append(dense_retrieve(sparse_query, cfg.top_k_retrieve, "embeddings_gemini"))
        except Exception:
            dense_results.append(dense_retrieve(sparse_query, cfg.top_k_retrieve, "embeddings_sbert"))
    else:
        dense_results.append(dense_retrieve(sparse_query, cfg.top_k_retrieve, "embeddings_sbert"))
        if cfg.use_multi_embedding:
            try:
                dense_results.append(dense_retrieve(sparse_query, cfg.top_k_retrieve, "embeddings_gemini"))
            except Exception:
                pass

    sparse_top = _extract_docids(sparse_res)
    dense_tops = [_extract_docids(dr) for dr in dense_results]

    dense0_score = _top_score(dense_results[0]) if dense_results else 0.0

    # Fusion candidates
    if cfg.use_rrf:
        fusion_top = rrf_fusion(sparse_res, dense_results, top_k=cfg.candidate_pool_size, k=cfg.rrf_k)
    else:
        candidates_with_scores = hard_vote_results(
            sparse_res,
            dense_results,
            top_k=cfg.candidate_pool_size,
            weights=cfg.voting_weights,
        )
        fusion_top = [x["docid"] for x in candidates_with_scores]

    # Reranker (optional)
    reranked: List[str] = []
    reranker_max_score: float = 0.0
    if cfg.use_reranker and fusion_top:
        docs_dict = get_documents_batch(fusion_top)
        docs_with_content = [(docid, docs_dict.get(docid, "")) for docid in fusion_top if docs_dict.get(docid)]
        if docs_with_content:
            from retrieval.reranker import reranker

            # Score a small candidate slice for auditing (faster than full rerank over huge pools)
            slice_docs = docs_with_content[: min(32, len(docs_with_content))]
            pairs = [[original_query, d[1][:512]] for d in slice_docs]
            try:
                scores = reranker.model.predict(
                    pairs,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                # reranker scores are unbounded; keep max as support signal
                reranker_max_score = float(max(scores)) if len(scores) else 0.0
                # derive reranked docids by score
                order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:5]
                reranked = [slice_docs[i][0] for i in order]
            except Exception:
                reranked = reranker.rerank(original_query, slice_docs, top_k=5, batch_size=32)
        else:
            reranked = fusion_top[:5]

    predicted_no_search = (not is_science) and (solar_conf >= cfg.no_search_conf_threshold)

    # Proxy scores
    # A: analyzer confidence
    A = max(0.0, min(1.0, solar_conf))

    # B: supportability (combines retrieval + (optional) reranker max score)
    bm25_score = _top_score(sparse_res)
    bm25_sig = max(0.0, min(1.0, _sigmoid(math.log1p(max(0.0, bm25_score)))))
    dense_sig = max(0.0, min(1.0, _sigmoid(math.log1p(max(0.0, dense0_score)))))
    rerank_sig = max(0.0, min(1.0, _sigmoid(reranker_max_score))) if cfg.use_reranker else 0.0
    if cfg.use_reranker:
        B = 0.35 * bm25_sig + 0.25 * dense_sig + 0.40 * rerank_sig
    else:
        B = 0.55 * bm25_sig + 0.45 * dense_sig
    B = max(0.0, min(1.0, B))

    overlaps: List[float] = []
    if dense_tops:
        overlaps.append(_jaccard_topk(sparse_top, dense_tops[0], k=5))
    overlaps.append(_jaccard_topk(sparse_top, fusion_top, k=5))
    if reranked:
        overlaps.append(_jaccard_topk(fusion_top, reranked, k=5))
    C = sum(overlaps) / len(overlaps) if overlaps else 0.0

    tags: List[str] = []
    reasons: List[str] = []

    if predicted_no_search:
        reason_tag, reason_desc = _no_search_reason(messages)
        tags.append(f"NOSEARCH_{reason_tag}")
        reasons.append(reason_desc)
        if solar_conf < 0.85:
            tags.append("NOSEARCH_LOW_CONF")
            reasons.append("no-search predicted but confidence is not high")
        if B > 0.85:
            tags.append("NOSEARCH_BUT_STRONG_SUPPORT")
            reasons.append("no-search predicted but supportability signal is strong")
    else:
        if solar_conf < 0.4:
            tags.append("SEARCH_LOW_CONF")
            reasons.append("search predicted but analyzer confidence is low")

        if B < 0.55:
            # could be OOD (should be no-search), or needs better query expansion
            reason_tag, reason_desc = _no_search_reason(messages)
            if reason_tag != "UNKNOWN":
                tags.append("SEARCH_LIKELY_OOD")
                reasons.append(f"weak support + no-search-like pattern: {reason_desc}")
            else:
                tags.append("SEARCH_BUT_WEAK_SUPPORT")
                reasons.append("search predicted but supportability is weak")

    if reranked and fusion_top:
        if (reranked[:1] != fusion_top[:1]) and (_jaccard_topk(reranked, fusion_top, k=5) < 0.4):
            tags.append("RERANKER_FLIP_HEAVY")
            reasons.append("reranker heavily disagrees with fusion ordering")

    if dense_tops:
        if _jaccard_topk(sparse_top, dense_tops[0], k=5) < 0.2:
            tags.append("SPARSE_DENSE_DIVERGENCE")
            reasons.append("sparse and dense retrieve very different sets")

    if not tags:
        tags.append("OK")

    debug = {
        "solar": {
            "is_science": is_science,
            "confidence": solar_conf,
            "standalone_query": standalone_query,
            "hyde": hyde[:200],
        },
        "decision": {
            "predicted_no_search": predicted_no_search,
            "no_search_conf_threshold": cfg.no_search_conf_threshold,
        },
        "retrieval": {
            "bm25_top_score": bm25_score,
            "dense0_top_score": dense0_score,
            "reranker_max_score": reranker_max_score if cfg.use_reranker else None,
            "sparse_top5": sparse_top[:5],
            "dense0_top5": dense_tops[0][:5] if dense_tops else [],
            "fusion_top5": fusion_top[:5],
            "rerank_top5": reranked[:5] if reranked else [],
        },
        "overlaps": {
            "sparse_vs_dense0": overlaps[0] if dense_tops and overlaps else None,
            "sparse_vs_fusion": overlaps[1] if len(overlaps) > 1 else None,
            "fusion_vs_rerank": overlaps[-1] if reranked else None,
        },
    }

    record = {
        "eval_id": None,
        "query": original_query,
        "scores": {"A": round(A, 4), "B": round(B, 4), "C": round(C, 4)},
        "tags": tags,
        "reasons": reasons,
        "evidence": debug,
        "actions": [],
        "next_actions": [],
    }

    return record, debug


def main() -> int:
    cfg = _parse_args()
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    base_env = {
        "EVAL_FILE": str(cfg.eval_path),
        "EVAL_LIMIT": str(cfg.limit),
        "USE_SOLAR_ANALYZER": "true",
        "TOP_K_RETRIEVE": str(cfg.top_k_retrieve),
        "CANDIDATE_POOL_SIZE": str(cfg.candidate_pool_size),
        "HYDE_MAX_LENGTH": str(cfg.hyde_max_length),
        "VOTING_WEIGHTS_JSON": json.dumps(cfg.voting_weights, separators=(",", ":")),
        "NO_SEARCH_CONFIDENCE_THRESHOLD": str(cfg.no_search_conf_threshold),
        "SUBMISSION_FILE": f"submission_audit_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    }

    actions = _make_actions(base_env)
    action_map = {a["label"]: a for a in actions}

    counts: Dict[str, int] = {}

    with cfg.eval_path.open("r", encoding="utf-8") as f, cfg.out_path.open("w", encoding="utf-8") as out:
        for i, line in enumerate(f, 1):
            if i > cfg.limit:
                break
            o = json.loads(line)
            eval_id = o.get("eval_id")
            msgs = o.get("msg")

            rec, _ = audit_one(msgs, cfg)
            rec["eval_id"] = eval_id
            rec["actions"] = actions

            # Attach per-record recommended actions ("buttons")
            rec_labels = _recommended_action_labels(rec.get("tags") or [])
            rec["next_actions"] = [action_map[lbl] for lbl in rec_labels if lbl in action_map]

            for t in rec["tags"]:
                counts[t] = counts.get(t, 0) + 1

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_path": str(cfg.eval_path),
        "limit": cfg.limit,
        "tag_counts": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
        "out_path": str(cfg.out_path),
        "notes": [
            "Scores are proxy/heuristic (no ground-truth labels).",
            "Use tags to choose next experiment knobs.",
        ],
    }

    Path(str(cfg.out_path) + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
