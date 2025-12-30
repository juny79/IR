import json
from collections import defaultdict
import sys

def ensemble_rrf(file1, file2, output_file, w1=0.7, w2=0.3, k=60):
    def load_results_full(path):
        res = {}
        with open(path, 'r') as f:
            for l in f:
                if not l.strip(): continue
                obj = json.loads(l)
                res[obj['eval_id']] = obj
        return res

    res1 = load_results_full(file1)
    res2 = load_results_full(file2)
    
    # file1 (SOTA)의 순서를 그대로 따름
    with open(file1, 'r') as f:
        order = [json.loads(l)['eval_id'] for l in f if l.strip()]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for eid in order:
            scores = defaultdict(float)
            
            # File 1 (SOTA)
            topk1 = res1.get(eid, {}).get('topk', [])
            for rank, docid in enumerate(topk1, 1):
                scores[docid] += w1 * (1.0 / (k + rank))
            
            # File 2 (V2 Final)
            topk2 = res2.get(eid, {}).get('topk', [])
            for rank, docid in enumerate(topk2, 1):
                scores[docid] += w2 * (1.0 / (k + rank))
            
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            new_topk = [docid for docid, score in sorted_docs[:5]]
            
            # 기존 SOTA(res1)의 데이터를 복사하고 topk만 업데이트
            final_obj = res1.get(eid, res2.get(eid)).copy()
            final_obj['topk'] = new_topk
            
            f.write(json.dumps(final_obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # 기본 가중치 0.7 : 0.3
    ensemble_rrf('submission_v9_sota.csv', 'submission_v2_final_rerank.csv', 'submission_final_ensemble_v9_v2.csv', w1=0.7, w2=0.3)
    print("Ensemble complete: submission_final_ensemble_v9_v2.csv")
