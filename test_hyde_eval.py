#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from models.llm_client import LLMClient

llm = LLMClient()

# eval.jsonl에서 샘플 질문 추출
with open('./data/eval.jsonl', 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f][:10]

print('=== 실제 평가 질문에 대한 HyDE 생성 ===\n')

empty_count = 0
for i, q in enumerate(questions, 1):
    standalone = llm.analyze_query(q['msg'])
    if standalone.tool_calls:
        query = json.loads(standalone.tool_calls[0].function.arguments)['standalone_query']
    else:
        query = q['msg'][0]['content']
    
    hypo = llm.generate_hypothetical_answer(query)
    
    if not hypo:
        empty_count += 1
        print(f'{i}. [ID {q["eval_id"]}] {query}')
        print(f'   ❌ 가상 답변 생성 실패\n')
    else:
        print(f'{i}. [ID {q["eval_id"]}] {query}')
        print(f'   ✅ 가상 답변 ({len(hypo)}자): {hypo[:100]}...\n')

print(f'\n총 {len(questions)}개 중 {empty_count}개 생성 실패 ({empty_count/len(questions)*100:.1f}%)')
