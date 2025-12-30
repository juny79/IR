import json
from retrieval.es_connector import es

def get_doc_content(docid):
    try:
        res = es.search(index="test", query={"term": {"docid": docid}}, _source=["content"])
        if res['hits']['hits']:
            return res['hits']['hits'][0]['_source']['content']
    except:
        pass
    return "Content not found."

def generate_judge_report(diffs_path, output_path):
    with open(diffs_path, 'r', encoding='utf-8') as f:
        diffs = json.load(f)
    
    report = []
    for d in diffs:
        print(f"Processing ID {d['eval_id']}...")
        v9_content = get_doc_content(d['v9_top1'])
        v3_content = get_doc_content(d['v3_top1'])
        
        report.append({
            'eval_id': d['eval_id'],
            'query': d['query'],
            'v9': {
                'docid': d['v9_top1'],
                'content': v9_content[:500] + "..."
            },
            'v3': {
                'docid': d['v3_top1'],
                'content': v3_content[:500] + "..."
            }
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_judge_report('/root/IR/v9_v3_diffs.json', '/root/IR/judge_report.json')
