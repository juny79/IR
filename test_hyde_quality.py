#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from models.llm_client import LLMClient
import time

llm = LLMClient()

test_queries = [
    '광합성이란?',
    '파이썬 리스트와 튜플의 차이',
    '프랑스 혁명',
    '양자역학',
    '태양계의 행성'
]

print('=== HyDE 가상 답변 품질 테스트 ===\n')
for i, query in enumerate(test_queries, 1):
    print(f'{i}. 원본 질문: {query}')
    hypo = llm.generate_hypothetical_answer(query)
    print(f'   가상 답변 ({len(hypo)}자): {hypo}')
    print(f'   결합 쿼리 ({len(query) + len(hypo)}자):')
    print(f'   {query}')
    print(f'   {hypo}\n')
    time.sleep(2)
