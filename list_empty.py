import json

def list_empty(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if not obj.get('topk'):
                print(f"ID {obj['eval_id']}: {obj.get('standalone_query', 'N/A')}")

if __name__ == "__main__":
    print("Empty results in v9_sota:")
    list_empty('/root/IR/submission_v9_sota.csv')
