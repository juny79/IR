import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SwapDecision:
    eval_id: int
    from_docid: str
    to_docid: str
    margin: float


REQUIRED_KEYS = ("eval_id", "topk", "answer")


def _load_submission(path: str) -> List[dict]:
    rows = []
    for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def _index_by_eval_id(rows: List[dict]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for o in rows:
        out[int(o["eval_id"])] = o
    return out


def _validate_row(o: dict) -> None:
    for k in REQUIRED_KEYS:
        if k not in o:
            raise ValueError(f"missing key {k} in row eval_id={o.get('eval_id')}")
    if not isinstance(o["topk"], list):
        raise ValueError(f"topk must be list eval_id={o.get('eval_id')}")


def _move_to_front(topk: List[str], docid: str) -> List[str]:
    if not topk:
        return topk
    if docid not in topk:
        return topk
    new = [docid] + [d for d in topk if d != docid]
    return new


def build_v17(
    base_path: str,
    margins_path: str,
    output_path: str,
    min_margin: float = 0.01,
    limit_swaps: int = 3,
) -> Tuple[List[SwapDecision], int]:
    base_rows = _load_submission(base_path)
    for o in base_rows:
        _validate_row(o)

    base_by_id = _index_by_eval_id(base_rows)

    scored = json.loads(Path(margins_path).read_text(encoding="utf-8"))
    # scored is sorted by margin desc in artifact generator
    decisions: List[SwapDecision] = []
    for r in scored:
        eid = int(r["eval_id"])
        margin = float(r["margin"])
        if margin < min_margin:
            continue
        base_topk = base_by_id[eid].get("topk") or []
        v16_top1 = r["v16_top1"]
        if not base_topk or v16_top1 not in base_topk:
            continue
        decisions.append(
            SwapDecision(
                eval_id=eid,
                from_docid=base_topk[0],
                to_docid=v16_top1,
                margin=margin,
            )
        )
        if len(decisions) >= limit_swaps:
            break

    # Apply decisions
    changed = 0
    for d in decisions:
        row = base_by_id[d.eval_id]
        topk = row.get("topk") or []
        new_topk = _move_to_front(topk, d.to_docid)
        if new_topk != topk:
            row["topk"] = new_topk
            changed += 1

    # Write JSONL
    out = Path(output_path)
    with out.open("w", encoding="utf-8") as f:
        for o in base_rows:
            # keep formatting stable
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    return decisions, changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a conservative v17 submission by starting from v9 and applying a few safe top1 swaps.")
    parser.add_argument("--base", default="submission_v9_sota.csv")
    parser.add_argument("--margins", default="artifacts/v16_vs_v9_reranker_margins.json")
    parser.add_argument("--min-margin", type=float, default=0.01)
    parser.add_argument("--limit-swaps", type=int, default=3)
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = args.output.strip() or f"submission_v17_conservative_from_v9_{ts}.csv"
    report_path = args.report.strip() or f"artifacts/v17_report_{Path(out).stem}.json"

    decisions, changed = build_v17(
        base_path=args.base,
        margins_path=args.margins,
        output_path=out,
        min_margin=args.min_margin,
        limit_swaps=args.limit_swaps,
    )

    report = {
        "base": args.base,
        "margins": args.margins,
        "output": out,
        "changed_rows": changed,
        "min_margin": args.min_margin,
        "limit_swaps": args.limit_swaps,
        "decisions": [d.__dict__ for d in decisions],
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
