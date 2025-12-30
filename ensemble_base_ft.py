import json
import os

def load_jsonl(path):
    data = {}
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            data[str(obj['eval_id'])] = obj
    return data

def weighted_rrf_ensemble(base_file, ft_file, output_file, w_base=0.7, w_ft=0.3, k_rrf=60):
    base_data = load_jsonl(base_file)
    ft_data = load_jsonl(ft_file)
    
    all_ids = sorted(set(base_data.keys()) | set(ft_data.keys()), key=lambda x: int(x))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for eid in all_ids:
            base_obj = base_data.get(eid, {})
            ft_obj = ft_data.get(eid, {})
            
            base_topk = base_obj.get('topk', [])
            ft_topk = ft_obj.get('topk', [])
            
            # 게이팅 처리 (둘 다 비어있으면 빈 리스트)
            if not base_topk and not ft_topk:
                res = {
                    "eval_id": int(eid),
                    "topk": [],
                    "answer": base_obj.get('answer', ft_obj.get('answer', "검색이 필요하지 않은 질문입니다."))
                }
            else:
                # Weighted RRF Score 계산
                scores = {}
                
                # Base Ranks
                for rank, docid in enumerate(base_topk, 1):
                    scores[docid] = scores.get(docid, 0) + w_base * (1.0 / (k_rrf + rank))
                
                # FT Ranks
                for rank, docid in enumerate(ft_topk, 1):
                    scores[docid] = scores.get(docid, 0) + w_ft * (1.0 / (k_rrf + rank))
                
                # 점수순 정렬
                sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                final_topk = [doc for doc, score in sorted_docs[:5]]
                
                # 결과 구성 (메타데이터는 Base 우선)
                res = {
                    "eval_id": int(eid),
                    "standalone_query": base_obj.get('standalone_query', ft_obj.get('standalone_query', "")),
                    "topk": final_topk,
                    "answer": base_obj.get('answer', ft_obj.get('answer', ""))
                }
            
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    BASE_FILE = "submission_v9_sota.csv"
    FT_FILE = "submission_bge_m3_finetuned_v9.csv"
    OUTPUT_FILE = "submission_ensemble_base0.8_ft0.2.csv"
    
    print(f"🚀 앙상블 생성 중: {BASE_FILE} (0.8) + {FT_FILE} (0.2)")
    weighted_rrf_ensemble(BASE_FILE, FT_FILE, OUTPUT_FILE, w_base=0.8, w_ft=0.2)
    print(f"✅ 앙상블 완료: {OUTPUT_FILE}")
