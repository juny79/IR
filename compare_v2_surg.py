import json
from pathlib import Path

def load_full(p):
    d={}
    if not Path(p).exists(): return d
    for ln in Path(p).read_text(encoding='utf-8', errors='ignore').splitlines():
        if not ln.strip(): continue
        try:
            o=json.loads(ln)
            d[str(o['eval_id'])]=o
        except: continue
    return d

surg = load_full('submission_surgical_v1.csv')
v2 = load_full('submission_grid_v2_mq_off_20251229_025014.csv')

surg_empty = [k for k,v in surg.items() if not v['topk']]
v2_empty = [k for k,v in v2.items() if not v['topk']]

print(f"Surgical Empty: {len(surg_empty)}")
print(f"V2 Empty: {len(v2_empty)}")
print(f"Surg - V2: {set(surg_empty) - set(v2_empty)}")
print(f"V2 - Surg: {set(v2_empty) - set(surg_empty)}")

# Top-5 Overlap
overlap = 0
total = 0
for eid in surg:
    if eid in v2:
        s_top5 = set(surg[eid]['topk'][:5])
        v_top5 = set(v2[eid]['topk'][:5])
        if s_top5 and v_top5:
            overlap += len(s_top5.intersection(v_top5))
            total += 5

print(f"Top-5 Overlap: {overlap/total:.4%}")
