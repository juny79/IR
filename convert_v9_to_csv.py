import json
import csv

input_file = 'submission_v9_sota.csv'
output_file = 'submission_v9_sota_fixed.csv'

with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['id', 'top_k'])
    
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        eval_id = data['eval_id']
        top_k = " ".join(data['topk'])
        writer.writerow([eval_id, top_k])

print(f"Converted {input_file} to {output_file}")
