import json
from tqdm import tqdm

V1_PATH = "submission_surgical_v1.csv"
MASTER_PATH = "submission_final_0.95_master.csv"
CAND_D_PATH = "submission_candidate_D_id271_id303.csv"
OUTPUT_PATH = "submission_final_surgical_hybrid_0.95.csv"

def load_full(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                d[obj['eval_id']] = obj
    return d

print("Loading files...")
v1 = load_full(V1_PATH)
master = load_full(MASTER_PATH)
cand_d = load_full(CAND_D_PATH)

final_results = []
all_ids = sorted(v1.keys())

for eid in all_ids:
    obj = v1[eid]
    
    # Surgical changes
    if eid == 270:
        print(f"Changing ID 270 to master version...")
        obj = master[eid]
    elif eid == 271:
        print(f"Changing ID 271 to cand_d version...")
        obj = cand_d[eid]
        
    final_results.append(obj)

print(f"Saving to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for res in final_results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print("Done!")
