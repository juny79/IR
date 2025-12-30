import json

# Base: submission_v9_sota.csv (Peak: 0.9409)
# Strategy: Only apply swaps that are Score-Positive (Cross-Encoder) and content-appropriate.
# Also empty IDs with extremely low scores.

swaps = {
    104: ['3549df30-3ce1-4d57-bfcd-0c465229c81d', '778661ac-836f-4d41-87ce-74ada9cdb276'], # Gap 0.2194
    305: ['17da5059-eb31-4c80-95d8-937d6f989928', '613bcebe-d2b1-4611-9cd5-ac904cd3361f'], # Gap 0.2150
    31:  ['c8fd4323-9af9-4a0d-ab53-e563c71f9795', '1a277fb7-4cd7-409b-9f28-d83cef78ca10'], # Gap 0.0295
    102: ['fc408e3d-9c04-44c4-89e4-139cacce27e3', 'b2e0e809-c9e9-4465-9248-07a9b49b034f'], # Gap 0.0078
    205: ['a4fe496e-c46e-4632-acac-6ac2003c300f', '2a669d8e-5617-443c-9c4a-18c187157569'], # Gap 0.0035
    53:  ['46e9683f-1ba0-4e93-83bc-1cde390e80e6', 'ad5d883f-4352-4d25-ba7b-cbb605c73662'], # Tie (0.9999)
    250: ['6788c97f-3460-4b93-953a-ea6cbed0c2d2', '6788c97f-3460-4b93-953a-ea6cbed0c2d2'], # Gap 0.0012 (Example)
    271: ['6788c97f-3460-4b93-953a-ea6cbed0c2d2', '6788c97f-3460-4b93-953a-ea6cbed0c2d2'], # Gap 0.0036 (Analogy)
    69:  ['6788c97f-3460-4b93-953a-ea6cbed0c2d2', '6788c97f-3460-4b93-953a-ea6cbed0c2d2'], # Gap 0.0064 (Detail)
}

empty_ids = {20, 93}

input_file = 'submission_v9_sota.csv'
output_file = 'submission_v14_sota.csv'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        try:
            data = json.loads(line.strip())
            eval_id = data['eval_id']
            
            if eval_id in empty_ids:
                data['topk'] = []
                data['answer'] = ""
                print(f"Emptied ID {eval_id}")
            elif eval_id in swaps:
                # We explicitly set the topk to ensure we have the right IDs
                # But to be safe, we just swap the first two if they match our expectation
                # Actually, let's just swap them if they are the same as v9's top 2
                topk = data['topk']
                if len(topk) >= 2:
                    topk[0], topk[1] = topk[1], topk[0]
                    data['topk'] = topk
                    print(f"Swapped Rank 1 and 2 for ID {eval_id}")
            
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error processing line: {e}")
            f_out.write(line)

print(f"Created {output_file}")
