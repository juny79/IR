#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from models.embedding_client import embedding_client
import numpy as np

# 원본 쿼리 vs HyDE 결합 쿼리 임베딩 비교
queries = [
    ('광합성이란?', '광합성이란?\n광합성은 식물, 조류, 일부 세균이 빛에너지를 이용하여 이산화탄소와 물로부터 유기물(포도당)을 합성하는 과정입니다.'),
    ('파이썬', '파이썬\n파이썬은 1991년 귀도 반 로섬이 개발한 고급 프로그래밍 언어로, 읽기 쉬운 문법과 강력한 라이브러리를 특징으로 합니다.'),
]

print('=== 원본 vs HyDE 임베딩 비교 ===\n')

for original, hyde in queries:
    emb_orig = embedding_client.get_query_embedding(original)
    emb_hyde = embedding_client.get_query_embedding(hyde)
    
    # 코사인 유사도
    similarity = np.dot(emb_orig, emb_hyde) / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_hyde))
    
    print(f'원본: "{original}" ({len(original)}자)')
    print(f'HyDE: "{hyde[:50]}..." ({len(hyde)}자)')
    print(f'임베딩 유사도: {similarity:.4f}')
    print(f'임베딩 차이: {np.linalg.norm(emb_orig - emb_hyde):.4f}\n')
