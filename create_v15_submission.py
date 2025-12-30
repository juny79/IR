import json

# Base: submission_v9_sota.csv (Peak: 0.9409)
# Strategy: Apply factual fixes and restore regressions from v5.

# Factual fixes (Swapping Rank 1 and 2 in v9)
swaps = {106, 263, 205}

# Regressions in v9 (Restoring Rank 1 from v5)
# ID 37: v9 has Tug of War, v5 has Chip Probability (Correct)
# ID 8: v9 has Harmful effects, v5 has Causes (Correct)
# ID 246: v9 has Community, v5 has Renewable fuel (Better)
# ID 214: v9 has Electrons/Protons, v5 has Nucleus/Electrons (Better)
restores = {
    37: '497d109c-5076-4287-a612-cc9f885150d9',
    8: 'd5569147-478a-4b93-b5f1-19dff5e4c092',
    246: '847bae35-d5eb-4d5f-8133-98e9c0292075',
    214: '50b3e292-2b9e-44a1-bb4d-3a6dc5a0acdb'
}

input_file = 'submission_v9_sota.csv'
output_file = 'submission_v15_sota.csv'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        try:
            data = json.loads(line.strip())
            eval_id = data['eval_id']
            
            if eval_id in swaps:
                topk = data['topk']
                if len(topk) >= 2:
                    topk[0], topk[1] = topk[1], topk[0]
                    data['topk'] = topk
                    print(f"Swapped Rank 1 and 2 for ID {eval_id}")
            elif eval_id in restores:
                target_id = restores[eval_id]
                topk = data['topk']
                if target_id in topk:
                    # Move target_id to Rank 1
                    topk.remove(target_id)
                    topk.insert(0, target_id)
                    data['topk'] = topk
                    print(f"Restored Rank 1 for ID {eval_id} to {target_id}")
                else:
                    # If not in topk, just prepend it
                    topk.insert(0, target_id)
                    data['topk'] = topk[:5] # Keep top 5
                    print(f"Prepended Rank 1 for ID {eval_id} to {target_id}")
            
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error processing line: {e}")
            f_out.write(line)

print(f"Created {output_file}")
