import json
from pathlib import Path

def load_top1(p):
    d={}
    if not Path(p).exists(): return d
    for ln in Path(p).read_text(encoding='utf-8', errors='ignore').splitlines():
        if not ln.strip(): continue
        try:
            o=json.loads(ln)
            topk=o.get('topk')
            d[str(o['eval_id'])]=topk[0] if isinstance(topk,list) and topk else None
        except: continue
    return d

surg = load_top1('submission_surgical_v1.csv')
base = load_top1('submission_88_ready_bge_m3_sota_20251229_023154.csv')
v2 = load_top1('submission_grid_v2_mq_off_20251229_025014.csv')
recovery = load_top1('submission_v8_recovery_recovery.csv')

all_ids = sorted(surg.keys(), key=int)

print(f"{'ID':<5} | {'Surg':<10} | {'Base':<10} | {'V2':<10} | {'Recov':<10}")
print("-" * 55)

missed_by_all = []
for eid in all_ids:
    s = surg.get(eid)
    b = base.get(eid)
    v = v2.get(eid)
    r = recovery.get(eid)
    
    if s != b and s != v and s != r:
        missed_by_all.append(eid)
        print(f"{eid:<5} | {s[:8] if s else 'None':<10} | {b[:8] if b else 'None':<10} | {v[:8] if v else 'None':<10} | {r[:8] if r else 'None':<10}")

print(f"\nTotal IDs missed by all 3 recent runs: {len(missed_by_all)}")
