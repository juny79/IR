import json

def load_full(p):
    d = {}
    with open(p, 'r') as f:
        for l in f:
            if not l.strip(): continue
            o = json.loads(l)
            d[o['eval_id']] = o
    return d

def main():
    v9 = load_full('submission_v9_sota.csv')
    v2 = load_full('submission_v2_final_rerank.csv')
    v12 = load_full('submission_v12_sota.csv')
    v13 = load_full('submission_v13_sota.csv')
    v14 = load_full('submission_v14_sota.csv')
    v15 = load_full('submission_v15_sota.csv')

    # IDs to switch based on Majority Vote (V12, V13, V14, V15, V2 vs V9)
    # From previous analysis: 205, 106
    to_switch = [205, 106]
    
    print(f"Creating Conservative Strike submission. Switching IDs: {to_switch}")
    
    final_results = []
    # Maintain original order from V9
    with open('submission_v9_sota.csv', 'r') as f:
        for line in f:
            obj = json.loads(line)
            eid = obj['eval_id']
            
            if eid in to_switch:
                print(f"  Switching ID {eid} to V2's result")
                # Use V2's topk and answer
                new_obj = obj.copy()
                new_obj['topk'] = v2[eid]['topk']
                new_obj['answer'] = v2[eid]['answer']
                final_results.append(new_obj)
            else:
                final_results.append(obj)

    with open('submission_conservative_strike.csv', 'w') as f:
        for res in final_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
