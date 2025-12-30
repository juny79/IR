"""Leaderboard-oriented grid search runner.

Why this exists:
- The leaderboard is the only authoritative MAP/MRR scorer.
- Full 220-question runs are expensive.
- This runner helps you:
  1) generate submission files for selected parameter grids
  2) snapshot hashes + sanity stats
  3) compute *proxy* similarity metrics vs a baseline submission

It does NOT compute real MAP/MRR locally.

Usage examples:
- Dry run (first 50 evals) to filter configs quickly:
    python3 experiments/leaderboard_grid_search.py --mode dry

- Full run for a small set of configs:
    python3 experiments/leaderboard_grid_search.py --mode full

Outputs:
- experiments/out/<exp_id>/submission.jsonl
- experiments/out/<exp_id>/run.log
- experiments/results.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "experiments" / "out"
RESULTS_PATH = REPO_ROOT / "experiments" / "results.jsonl"


@dataclass(frozen=True)
class Config:
    top_k_retrieve: int
    candidate_pool_size: int
    hyde_max_length: int
    voting_weights: list[int]

    def to_env(self) -> dict[str, str]:
        return {
            "TOP_K_RETRIEVE": str(self.top_k_retrieve),
            "CANDIDATE_POOL_SIZE": str(self.candidate_pool_size),
            "HYDE_MAX_LENGTH": str(self.hyde_max_length),
            "VOTING_WEIGHTS_JSON": json.dumps(self.voting_weights),
        }

    def slug(self) -> str:
        w = "".join(str(x) for x in self.voting_weights)
        return f"tk{self.top_k_retrieve}_cp{self.candidate_pool_size}_h{self.hyde_max_length}_w{w}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_submission_map(path: Path) -> dict[int, list[Any]]:
    out: dict[int, list[Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        eid = obj.get("eval_id")
        if eid is None:
            continue
        out[int(eid)] = obj.get("topk", [])
    return out


def overlap_metrics(a: dict[int, list[Any]], b: dict[int, list[Any]]) -> dict[str, float]:
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0, "top1_same_ratio": 0.0, "avg_top5_overlap": 0.0}

    top1_same = 0
    top5_overlap = 0.0

    for eid in ids:
        ta = a.get(eid) or []
        tb = b.get(eid) or []
        if ta and tb and ta[0] == tb[0]:
            top1_same += 1
        top5_overlap += len(set(ta[:5]) & set(tb[:5])) / 5.0

    n = len(ids)
    return {
        "n": float(n),
        "top1_same_ratio": top1_same / n,
        "avg_top5_overlap": top5_overlap / n,
    }


def run_one(config: Config, mode: str, baseline_path: Path, timeout_s: int) -> dict[str, Any]:
    exp_id = f"{int(time.time())}_{config.slug()}_{mode}"
    exp_dir = OUT_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    submission_path = exp_dir / "submission.jsonl"
    log_path = exp_dir / "run.log"

    env = os.environ.copy()
    env.update(config.to_env())
    env["SUBMISSION_FILE"] = str(submission_path)

    # Dry run: evaluate only first N to reduce cost.
    if mode == "dry":
        env["EVAL_LIMIT"] = env.get("EVAL_LIMIT", "50")
    else:
        env["EVAL_LIMIT"] = "0"

    # Ensure clean output
    if submission_path.exists():
        submission_path.unlink()

    started_at = time.time()
    ok = True
    err: str | None = None

    cmd = ["python3", "main.py"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_path.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            ok = False
            err = f"returncode={proc.returncode}"
    except subprocess.TimeoutExpired:
        ok = False
        err = f"timeout>{timeout_s}s"

    elapsed = time.time() - started_at

    result: dict[str, Any] = {
        "exp_id": exp_id,
        "mode": mode,
        "config": {
            "top_k_retrieve": config.top_k_retrieve,
            "candidate_pool_size": config.candidate_pool_size,
            "hyde_max_length": config.hyde_max_length,
            "voting_weights": config.voting_weights,
        },
        "ok": ok,
        "error": err,
        "elapsed_s": elapsed,
        "paths": {
            "submission": str(submission_path),
            "log": str(log_path),
        },
    }

    if submission_path.exists():
        # Snapshot stats
        raw = submission_path.read_bytes()
        rows = len([ln for ln in raw.splitlines() if ln.strip()])
        empty = 0
        parse_errors = 0
        for ln in raw.splitlines():
            s = ln.strip()
            if not s:
                continue
            try:
                o = json.loads(s)
            except Exception:
                parse_errors += 1
                continue
            topk = o.get("topk")
            if isinstance(topk, list) and len(topk) == 0:
                empty += 1

        result["submission"] = {
            "sha256": sha256_file(submission_path),
            "rows": rows,
            "parse_errors": parse_errors,
            "empty_topk_count": empty,
        }

        # Proxy overlap vs baseline
        if baseline_path.exists():
            try:
                base_map = load_submission_map(baseline_path)
                cur_map = load_submission_map(submission_path)
                result["proxy_overlap_vs_baseline"] = overlap_metrics(cur_map, base_map)
            except Exception as e:
                result["proxy_overlap_vs_baseline"] = {"error": str(e)}

    return result


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _get_nested(d: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def select_top_configs_from_results(
    results_path: Path,
    top_n: int,
    metric_key: str,
    prefer_mode: str | None = "dry",
    min_rows: int = 0,
) -> list[Config]:
    """Select top-N configs from experiments/results.jsonl.

    Since local MAP/MRR is not available, this is a *proxy* selector.
    Typical metrics:
      - proxy_overlap_vs_baseline.avg_top5_overlap
      - proxy_overlap_vs_baseline.top1_same_ratio
    """
    if not results_path.exists():
        raise FileNotFoundError(f"results not found: {results_path}")

    # Keep best score per config slug
    best_by_slug: dict[str, tuple[float, dict[str, Any]]] = {}

    for line in results_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        if not obj.get("ok"):
            continue

        if min_rows > 0:
            sub = obj.get("submission") or {}
            try:
                rows = int(sub.get("rows") or 0)
            except Exception:
                rows = 0
            if rows < min_rows:
                continue

        if prefer_mode is not None and obj.get("mode") != prefer_mode:
            continue

        metric_val = _get_nested(obj, metric_key)
        try:
            score = float(metric_val)
        except Exception:
            continue

        cfg = obj.get("config") or {}
        try:
            config = Config(
                top_k_retrieve=int(cfg["top_k_retrieve"]),
                candidate_pool_size=int(cfg["candidate_pool_size"]),
                hyde_max_length=int(cfg["hyde_max_length"]),
                voting_weights=list(cfg["voting_weights"]),
            )
        except Exception:
            continue

        slug = config.slug()
        prev = best_by_slug.get(slug)
        if prev is None or score > prev[0]:
            best_by_slug[slug] = (score, obj)

    ranked = sorted(best_by_slug.items(), key=lambda kv: kv[1][0], reverse=True)
    picked = []
    for slug, (score, obj) in ranked[:top_n]:
        cfg = obj["config"]
        picked.append(
            Config(
                top_k_retrieve=int(cfg["top_k_retrieve"]),
                candidate_pool_size=int(cfg["candidate_pool_size"]),
                hyde_max_length=int(cfg["hyde_max_length"]),
                voting_weights=list(cfg["voting_weights"]),
            )
        )
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry", "full"], default="dry")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--baseline", type=str, default="submission_best_map08765.csv")
    parser.add_argument(
        "--from-results",
        type=str,
        default="",
        help="Select configs from a previous run results.jsonl (proxy ranking).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="When used with --from-results, run only the top-N configs.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="proxy_overlap_vs_baseline.avg_top5_overlap",
        help="Dotted key used to rank configs from results.jsonl.",
    )
    parser.add_argument(
        "--prefer-mode",
        type=str,
        default="dry",
        help="Only consider entries of this mode when selecting from results.jsonl. Set to '' to consider all.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=0,
        help="When selecting from results.jsonl, ignore entries whose submission rows are below this value (e.g., 50 for dry runs).",
    )
    args = parser.parse_args()

    baseline_path = REPO_ROOT / args.baseline

    grid: list[Config]

    if args.from_results and args.top_n > 0:
        results_path = REPO_ROOT / args.from_results
        prefer = args.prefer_mode.strip() if args.prefer_mode is not None else "dry"
        if prefer == "":
            prefer = None
        grid = select_top_configs_from_results(
            results_path=results_path,
            top_n=args.top_n,
            metric_key=args.metric,
            prefer_mode=prefer,
            min_rows=args.min_rows,
        )
        if not grid:
            raise SystemExit(
                "No configs selected from results.jsonl (check --metric / --prefer-mode / results file)."
            )
    else:
        # Budgeted grid (start small). You can extend this grid gradually.
        # Based on your latest results, candidate_pool_size/top_k_retrieve are high-impact.
        grid = [
            # keep weights fixed first, sweep top-k + candidate pool
            Config(top_k_retrieve=80, candidate_pool_size=80, hyde_max_length=200, voting_weights=[5, 4, 2]),
            Config(top_k_retrieve=100, candidate_pool_size=100, hyde_max_length=200, voting_weights=[5, 4, 2]),
            Config(top_k_retrieve=120, candidate_pool_size=120, hyde_max_length=200, voting_weights=[5, 4, 2]),
            # hyde length sweeps (keep top-k=80)
            Config(top_k_retrieve=80, candidate_pool_size=80, hyde_max_length=150, voting_weights=[5, 4, 2]),
            Config(top_k_retrieve=80, candidate_pool_size=80, hyde_max_length=250, voting_weights=[5, 4, 2]),
            # weight sweeps (keep top-k=80)
            Config(top_k_retrieve=80, candidate_pool_size=80, hyde_max_length=200, voting_weights=[6, 4, 2]),
            Config(top_k_retrieve=80, candidate_pool_size=80, hyde_max_length=200, voting_weights=[5, 3, 1]),
            Config(top_k_retrieve=80, candidate_pool_size=80, hyde_max_length=200, voting_weights=[4, 4, 2]),
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in grid:
        res = run_one(cfg, args.mode, baseline_path=baseline_path, timeout_s=args.timeout)
        append_jsonl(RESULTS_PATH, res)
        status = "OK" if res.get("ok") else "FAIL"
        sub = res.get("submission") or {}
        print(
            f"[{status}] {cfg.slug()} mode={args.mode} rows={sub.get('rows')} empty={sub.get('empty_topk_count')} sha={sub.get('sha256')}"
        )


if __name__ == "__main__":
    main()
