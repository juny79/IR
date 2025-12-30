import json

def get_queries(ids, eval_path):
    queries = {}
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if str(obj['eval_id']) in ids:
                # Get the last user message
                msg = obj['msg']
                if isinstance(msg, list):
                    queries[str(obj['eval_id'])] = msg[-1]['content']
                else:
                    queries[str(obj['eval_id'])] = str(msg)
    return queries

if __name__ == "__main__":
    empty_ids = ["276", "261", "283", "32", "94", "90", "220", "245", "229", "247", "67", "57", "2", "227", "301", "222", "83", "64", "103", "218"]
    queries = get_queries(empty_ids, '/root/IR/data/eval.jsonl')
    for eid in empty_ids:
        print(f"ID {eid}: {queries.get(eid)}")
