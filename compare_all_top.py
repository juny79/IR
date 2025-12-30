import json
from pathlib import Path

def load_top1(path):
    out = {}
    if not Path(path).exists(): return {}
    with open(path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                out[str(obj['eval_id'])] = obj['topk'][0] if obj.get('topk') else None
            except:
                continue
    return out

files = {
    'v9': 'submission_v9_sota.csv',
    'best_9394': 'submission_best_9394.csv',
    'surgical_v1': 'submission_surgical_v1.csv',
    'break_v2': 'submission_final_0.95_break_v2.csv',
    'attack5': 'submission_v17_attack5_from_v9_20251227_150049.csv',
    'ensemble_0.7': 'submission_ensemble_base0.7_ft0.3.csv'
}

data = {name: load_top1(path) for name, path in files.items()}
all_ids = set()
for d in data.values():
    all_ids.update(d.keys())

disagreements = {}
for eid in sorted(list(all_ids), key=lambda x: int(x)):
    vals = {name: data[name].get(eid) for name in data if eid in data[name]}
    unique_vals = set(vals.values())
    if len(unique_vals) > 1:
        disagreements[eid] = vals

print(f"Total disagreement IDs: {len(disagreements)}")
for eid, vals in disagreements.items():
    print(f"\nID {eid}:")
    for name, val in vals.items():
        print(f"  {name:12}: {val}")
