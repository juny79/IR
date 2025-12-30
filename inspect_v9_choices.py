import pandas as pd
import json

v5 = pd.read_csv('submission_bge_m3_sota_v5.csv')
best = pd.read_csv('submission_best_9394.csv')
v9 = pd.read_csv('submission_v9_sota.csv')

with open('solar_diff_analysis.json', 'r') as f:
    solar_analysis = json.load(f)

diff_ids = [int(k) for k in solar_analysis.keys()]

print(f"{'ID':<5} | {'v5':<10} | {'best':<10} | {'v9':<10} | {'Solar Pref'}")
print("-" * 60)

for qid in diff_ids:
    v5_top = v5[v5['id'] == qid]['top_k'].iloc[0].split()[0]
    best_top = best[best['id'] == qid]['top_k'].iloc[0].split()[0]
    v9_top = v9[v9['id'] == qid]['top_k'].iloc[0].split()[0]
    
    pref = solar_analysis[str(qid)]['preferred']
    
    print(f"{qid:<5} | {v5_top:<10} | {best_top:<10} | {v9_top:<10} | {pref}")
