import json
from models.openai_client import openai_client

def llm_choose_best(messages, candidates):
    system_prompt = '당신은 검색 결과의 정확도를 판별하는 전문가입니다. 질문에 대해 가장 정확하고, 직접적이며, 완결된 답변을 제공하는 문서를 하나만 골라주세요. 반드시 JSON 형식으로 {"best_candidate_index": 0} 로 답변하세요.'
    user_prompt = f'대화 맥락:\n{json.dumps(messages, ensure_ascii=False, indent=2)}\n\n'
    for i, (doc_id, doc_content) in enumerate(candidates):
        user_prompt += f'Candidate {i} (ID: {doc_id}):\n{doc_content[:2000]}\n\n'
    resp = openai_client._call_with_retry(prompt=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], temperature=0, response_format={'type': 'json_object'})
    return json.loads(resp).get('best_candidate_index', 0)

with open('data/eval.jsonl', 'r') as f:
    eval_data = {json.loads(l)['eval_id']: json.loads(l)['msg'] for l in f if l.strip()}
with open('data/documents.jsonl', 'r') as f:
    docs = {json.loads(l)['docid']: json.loads(l)['content'] for l in f if l.strip()}

print("ID 271:")
c271 = [('c528c66d-07cc-4fc1-976d-631b76dddc58', docs['c528c66d-07cc-4fc1-976d-631b76dddc58']), ('0598d1c1-f304-47c2-927c-6076838a69e8', docs['0598d1c1-f304-47c2-927c-6076838a69e8'])]
print(f'Winner Index: {llm_choose_best(eval_data[271], c271)}')

print("ID 303:")
c303 = [('5973b08a-ef7d-431c-ac58-d31d4a57d63c', docs['5973b08a-ef7d-431c-ac58-d31d4a57d63c']), ('6b6971ff-885f-48cf-adca-46bbd01041e6', docs['6b6971ff-885f-48cf-adca-46bbd01041e6'])]
print(f'Winner Index: {llm_choose_best(eval_data[303], c303)}')
