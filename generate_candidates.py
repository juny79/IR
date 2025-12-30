import json
from pathlib import Path

BASE = Path('submission_82_surgical_v1.csv')

OUT_A = Path('submission_candidate_A_surgical.csv')
OUT_B = Path('submission_candidate_B_id271.csv')
OUT_C = Path('submission_candidate_C_id303.csv')
OUT_D = Path('submission_candidate_D_id271_id303.csv')

# Candidate swaps (Top-1 only)
SWAP_271 = '0598d1c1-f304-47c2-927c-6076838a69e8'
SWAP_303 = '6b6971ff-885f-48cf-adca-46bbd01041e6'


def load_jsonl(path: Path):
    rows = []
    for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def write_jsonl(path: Path, rows):
    with path.open('w', encoding='utf-8') as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def promote_top1(obj, new_docid: str):
    topk = obj.get('topk')
    if not isinstance(topk, list) or len(topk) == 0:
        # If it was intentionally empty, keep it empty (do not force-fill)
        return
    new_topk = [new_docid] + [d for d in topk if d != new_docid]
    obj['topk'] = new_topk[:5]


def make_variant(rows, change_271: bool, change_303: bool):
    out = []
    for obj in rows:
        eid = str(obj.get('eval_id'))
        if change_271 and eid == '271':
            promote_top1(obj, SWAP_271)
        if change_303 and eid == '303':
            promote_top1(obj, SWAP_303)
        out.append(obj)
    return out


def summarize(path: Path):
    rows = 0
    empty = 0
    parse_errors = 0
    top1_271 = None
    top1_303 = None
    for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows += 1
        try:
            obj = json.loads(ln)
            topk = obj.get('topk')
            if isinstance(topk, list) and len(topk) == 0:
                empty += 1
            eid = str(obj.get('eval_id'))
            if eid == '271':
                top1_271 = topk[0] if isinstance(topk, list) and len(topk) else None
            if eid == '303':
                top1_303 = topk[0] if isinstance(topk, list) and len(topk) else None
        except Exception:
            parse_errors += 1

    print(f'File: {path}')
    print(f'  rows: {rows}')
    print(f'  empty_topk_count: {empty}')
    print(f'  parse_errors: {parse_errors}')
    print(f'  ID 271 top1: {top1_271}')
    print(f'  ID 303 top1: {top1_303}')


if __name__ == '__main__':
    if not BASE.exists():
        raise SystemExit(f'Missing base file: {BASE}')

    base_rows = load_jsonl(BASE)

    # A: exact base
    write_jsonl(OUT_A, base_rows)

    # B: ID 271 swap
    write_jsonl(OUT_B, make_variant(load_jsonl(BASE), change_271=True, change_303=False))

    # C: ID 303 swap
    write_jsonl(OUT_C, make_variant(load_jsonl(BASE), change_271=False, change_303=True))

    # D: both swaps
    write_jsonl(OUT_D, make_variant(load_jsonl(BASE), change_271=True, change_303=True))

    print('Generated:')
    for p in [OUT_A, OUT_B, OUT_C, OUT_D]:
        print(f'- {p}')

    print('\nSanity checks:')
    for p in [OUT_A, OUT_B, OUT_C, OUT_D]:
        summarize(p)
