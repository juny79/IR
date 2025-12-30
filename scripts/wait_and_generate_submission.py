import os
import re
import time
import subprocess
from pathlib import Path


def _read_env_from_dotenv(repo_root: Path, key: str) -> str | None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def _latest_log(repo_root: Path) -> Path | None:
    logs = sorted(repo_root.glob("upstage_index_full_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _extract_progress_line(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines[-200:]):
        if "Indexing Upstage embeddings" in line:
            return line
    return None


def _is_indexer_running() -> bool:
    # True if any index_upstage_embeddings.py process exists
    r = subprocess.run(["pgrep", "-af", "index_upstage_embeddings.py"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def main():
    repo_root = Path(__file__).resolve().parents[1]

    # Ensure ES_PASSWORD is available for any downstream ES calls (main.py uses its own dotenv load)
    if not os.getenv("ES_PASSWORD"):
        pw = _read_env_from_dotenv(repo_root, "ES_PASSWORD")
        if pw:
            os.environ["ES_PASSWORD"] = pw

    # Wait for indexing to finish
    while _is_indexer_running():
        log_path = _latest_log(repo_root)
        if log_path:
            progress = _extract_progress_line(log_path)
            if progress:
                print(progress, flush=True)
        time.sleep(60)

    # Generate submission using multi-dense fields including Upstage 2048
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = repo_root / f"submission_ready_rrf_k20_mq_tk80_cp80_dense3_upstage2048_{ts}.csv"
    run_log = repo_root / f"run_rrf_k20_mq_tk80_cp80_dense3_upstage2048_{ts}.log"

    env = os.environ.copy()
    env.update(
        {
            "USE_RRF": "true",
            "RRF_K": "20",
            "USE_MULTI_QUERY": "true",
            "TOP_K_RETRIEVE": "80",
            "CANDIDATE_POOL_SIZE": "80",
            "DENSE_EMBEDDING_FIELDS": "embeddings_sbert,embeddings_gemini,embeddings_upstage_2048",
            "DENSE_K_PER_FIELD": "80",
            "SUBMISSION_FILE": str(out),
        }
    )

    print(f"Starting submission generation -> {out.name}")
    with open(run_log, "w", encoding="utf-8") as lf:
        p = subprocess.run(["python3", "main.py"], cwd=str(repo_root), env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"Done. exit_code={p.returncode} log={run_log.name}")


if __name__ == "__main__":
    main()
