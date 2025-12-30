import json

def cleanup_submission(input_path, output_path):
    results = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                eval_id = data.get('eval_id')
                if eval_id is not None:
                    # 중복 발생 시 나중에 나온 데이터로 덮어쓰기 (최신 결과 유지)
                    results[eval_id] = data
            except Exception as e:
                print(f"Error parsing line: {e}")

    # eval_id 순으로 정렬
    sorted_ids = sorted(results.keys())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for eid in sorted_ids:
            f.write(json.dumps(results[eid], ensure_ascii=False) + '\n')
    
    print(f"Cleanup complete. Total unique IDs: {len(results)}")
    return len(results)

if __name__ == "__main__":
    cleanup_submission('/root/IR/submission_v3_ensemble.csv', '/root/IR/submission_v3_final.csv')
