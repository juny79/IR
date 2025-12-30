import json

def compare_submissions(path1, path2, k=5):
    def load(p):
        data = {}
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                data[obj['eval_id']] = obj
        return data

    d1 = load(path1)
    d2 = load(path2)

    common_ids = set(d1.keys()) & set(d2.keys())
    if not common_ids:
        print("No common IDs found.")
        return

    top1_match = 0
    top5_overlap = 0
    total = len(common_ids)

    for eid in common_ids:
        tk1 = d1[eid].get('topk', [])
        tk2 = d2[eid].get('topk', [])
        
        # Top-1 match
        if tk1 and tk2 and tk1[0] == tk2[0]:
            top1_match += 1
        
        # Top-5 overlap
        s1 = set(tk1[:k])
        s2 = set(tk2[:k])
        if s1 and s2:
            top5_overlap += len(s1.intersection(s2))

    print(f"Comparison between {path1} and {path2}")
    print(f"Total IDs: {total}")
    print(f"Top-1 Agreement: {top1_match / total:.4f} ({top1_match}/{total})")
    print(f"Top-5 Overlap: {top5_overlap / (total * k):.4f}")

if __name__ == "__main__":
    import sys
    p1 = '/root/IR/submission_v3_final.csv'
    p2 = '/root/IR/submission_v9_sota.csv'
    compare_submissions(p1, p2)
