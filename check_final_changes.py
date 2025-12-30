import json

def check_final_changes(p1, p2):
    def load(p):
        return {json.loads(l)['eval_id']: json.loads(l) for l in open(p) if json.loads(l).get('topk')}
    
    d1 = load(p1)
    d2 = load(p2)
    
    for eid in d1:
        if eid in d2:
            t1_1 = d1[eid]['topk'][0]
            t1_2 = d2[eid]['topk'][0]
            if t1_1 != t1_2:
                print(f"ID {eid}")
                print(f"  Query: {d1[eid]['standalone_query']}")
                print(f"  Surgical_v1 Top-1: {t1_1}")
                print(f"  0.95_Break Top-1: {t1_2}")

if __name__ == "__main__":
    check_final_changes('/root/IR/submission_surgical_v1.csv', '/root/IR/submission_final_0.95_break.csv')
