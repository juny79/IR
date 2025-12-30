"""
submission_33 vs submission_weighted_rrf 비교 분석
"""
import json

def load_submissions(filepath):
    """JSONL 형식 submission 파일 로드"""
    results = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                results[data['eval_id']] = data
    return results

# 파일 로드
s33 = load_submissions('/root/IR/submission_33_ready_4_tk80_cp80_h200_w542.csv')
wrrf = load_submissions('/root/IR/submission_weighted_rrf.csv')

print("📊 파일 비교 분석")
print(f"submission_33: {len(s33)}개")
print(f"submission_weighted_rrf: {len(wrrf)}개")

# 샘플 비교 (처음 5개)
print("\n" + "="*100)
print("샘플 비교 (처음 5개 평가):")
print("="*100)

for i, (eval_id, data33) in enumerate(list(s33.items())[:5]):
    wrrf_data = wrrf.get(eval_id)
    
    if wrrf_data:
        query33 = data33.get('standalone_query', '')
        query_wrrf = wrrf_data.get('standalone_query', '')
        topk33 = data33.get('topk', [])
        topk_wrrf = wrrf_data.get('topk', [])
        
        print(f"\n[eval_id: {eval_id}]")
        print(f"  Query (s33): {query33[:60]}")
        print(f"  Query (wrrf): {query_wrrf[:60]}")
        print(f"  TopK (s33): {len(topk33)} {'non-empty' if topk33 else 'empty'}")
        print(f"  TopK (wrrf): {len(topk_wrrf)} {'non-empty' if topk_wrrf else 'empty'}")
        
        # topk 일치도
        if topk33 and topk_wrrf:
            matches = len(set(topk33) & set(topk_wrrf))
            print(f"  TopK 일치도: {matches}/{len(topk33)} = {matches/len(topk33)*100:.1f}%")

print("\n" + "="*100)
print("전체 통계:")
print("="*100)

# Empty topk 비교
empty33 = sum(1 for d in s33.values() if not d.get('topk', []))
empty_wrrf = sum(1 for d in wrrf.values() if not d.get('topk', []))

print(f"Empty topk (s33): {empty33}/220 ({empty33/220*100:.1f}%)")
print(f"Empty topk (wrrf): {empty_wrrf}/220 ({empty_wrrf/220*100:.1f}%)")

# TopK 크기 분포
topk_sizes_33 = [len(d.get('topk', [])) for d in s33.values()]
topk_sizes_wrrf = [len(d.get('topk', [])) for d in wrrf.values()]

avg_topk33 = sum(topk_sizes_33) / len(topk_sizes_33)
avg_topk_wrrf = sum(topk_sizes_wrrf) / len(topk_sizes_wrrf)

print(f"\n평균 TopK 크기:")
print(f"  submission_33: {avg_topk33:.2f}")
print(f"  weighted_rrf: {avg_topk_wrrf:.2f}")
