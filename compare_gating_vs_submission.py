#!/usr/bin/env python3
import json
from pathlib import Path


def load_audit(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        audit = json.load(f)
    non_science_ids = set(x['eval_id'] for x in audit.get('non_science', []))
    return audit.get('total'), non_science_ids


def load_submission(path: str):
    sub_path = Path(path)
    if not sub_path.exists():
        return set(), set(), 0

    sub_ids = set()
    empty_topk_ids = set()
    parse_errors = 0

    for line in sub_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            parse_errors += 1
            continue

        eid = obj.get('eval_id')
        if eid is None:
            continue
        sub_ids.add(eid)

        topk = obj.get('topk')
        if isinstance(topk, list) and len(topk) == 0:
            empty_topk_ids.add(eid)

    return sub_ids, empty_topk_ids, parse_errors


def main():
    audit_total, non_science_ids = load_audit('solar_gating_audit.json')
    sub_ids, empty_topk_ids, parse_errors = load_submission('submission.csv')

    # Only evaluate mismatches among IDs present in submission
    should_be_empty_but_not = sorted((non_science_ids & sub_ids) - empty_topk_ids)
    should_be_nonempty_but_empty = sorted((sub_ids - non_science_ids) & empty_topk_ids)

    print('=' * 80)
    print('Compare gating audit vs submission.csv')
    print('=' * 80)
    print(f'audit_total: {audit_total}')
    print(f'audit_non_science: {len(non_science_ids)}')
    print(f'submission_rows: {len(sub_ids)} (parse_errors={parse_errors})')
    print(f'submission_topk_empty: {len(empty_topk_ids)}')
    print('-' * 80)
    if len(sub_ids) != audit_total:
        print('WARNING: submission.csv is not complete (expected 220 rows).')
    print(f'non_science but topk NOT empty: {len(should_be_empty_but_not)}')
    print(f'science but topk empty: {len(should_be_nonempty_but_empty)}')

    if should_be_empty_but_not:
        print('\n[non_science but topk NOT empty]')
        print(should_be_empty_but_not[:50])
    if should_be_nonempty_but_empty:
        print('\n[science but topk empty]')
        print(should_be_nonempty_but_empty[:50])


if __name__ == '__main__':
    main()
