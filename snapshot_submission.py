import json
import hashlib
from pathlib import Path

def submission_stats(path: Path):
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()

    rows = 0
    parse_errors = 0
    empty_topk_ids = []
    eval_ids = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rows += 1
        try:
            obj = json.loads(line)
        except Exception:
            parse_errors += 1
            continue

        eid = obj.get("eval_id")
        if eid is not None:
            eval_ids.append(eid)

        topk = obj.get("topk")
        if isinstance(topk, list) and len(topk) == 0:
            if eid is not None:
                empty_topk_ids.append(eid)

    unique_eval_ids = sorted(set(eval_ids))
    duplicates = sorted({x for x in eval_ids if eval_ids.count(x) > 1}) if eval_ids else []

    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": sha256,
        "rows": rows,
        "parse_errors": parse_errors,
        "unique_eval_ids": len(unique_eval_ids),
        "min_eval_id": unique_eval_ids[0] if unique_eval_ids else None,
        "max_eval_id": unique_eval_ids[-1] if unique_eval_ids else None,
        "duplicate_eval_ids": duplicates,
        "empty_topk_count": len(empty_topk_ids),
        "empty_topk_ids": sorted(empty_topk_ids),
    }


def main():
    base = Path('.')
    out = {
        "generated_from": "snapshot_submission.py",
        "files": {},
    }

    for name in ["submission.csv", "submission_27.csv"]:
        p = base / name
        if p.exists():
            out["files"][name] = submission_stats(p)
        else:
            out["files"][name] = {"missing": True}

    # quick equality hint
    if (base / "submission.csv").exists() and (base / "submission_27.csv").exists():
        out["submission_equals_backup"] = (
            out["files"]["submission.csv"]["sha256"] == out["files"]["submission_27.csv"]["sha256"]
        )

    Path('submission_snapshot.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


if __name__ == "__main__":
    main()
