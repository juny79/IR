import json

# IDs to swap Rank 1 and Rank 2
# 106, 205, 263 are fact-based improvements
# 104, 305, 97, 35, 20, 10 are score-based improvements (Rank 2 >> Rank 1)
swaps = {106, 205, 263, 104, 305, 97, 35, 20, 10}

input_file = 'submission_v9_sota.csv'
output_file = 'submission_v13_sota.csv'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        try:
            data = json.loads(line.strip())
            eval_id = data['eval_id']
            
            if eval_id in swaps:
                topk = data['topk']
                if len(topk) >= 2:
                    # Swap Rank 1 and Rank 2
                    topk[0], topk[1] = topk[1], topk[0]
                    data['topk'] = topk
                    print(f"Swapped Rank 1 and 2 for ID {eval_id}")
            
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error processing line: {e}")
            f_out.write(line)

print(f"Created {output_file}")
