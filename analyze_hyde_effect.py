#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from models.llm_client import llm_client
from retrieval.es_connector import sparse_retrieve, dense_retrieve
from models.embedding_client import embedding_client
import json

# 테스트 질문 선택
with open('./data/eval.jsonl', 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f][:5]

print('=== HyDE가 Sparse/Dense에 미치는 영향 분석 ===\n')

for i, q in enumerate(questions, 1):
    # Standalone query 추출
    analysis = llm_client.analyze_query(q['msg'])
    if analysis.tool_calls:
        query = json.loads(analysis.tool_calls[0].function.arguments)['standalone_query']
    else:
        query = q['msg'][0]['content']
    
    # HyDE 생성
    hypo = llm_client.generate_hypothetical_answer(query)
    if not hypo:
        continue
        
    enhanced_query = f"{query}\n{hypo}"
    
    print(f'{i}. [ID {q["eval_id"]}] "{query}" ({len(query)}자)')
    print(f'   HyDE: "{hypo[:60]}..." ({len(hypo)}자)\n')
    
    # Sparse Search 비교
    sparse_orig = sparse_retrieve(query, 5)
    sparse_hyde = sparse_retrieve(enhanced_query, 5)
    
    sparse_orig_ids = [hit['_source']['docid'] for hit in sparse_orig['hits']['hits']]
    sparse_hyde_ids = [hit['_source']['docid'] for hit in sparse_hyde['hits']['hits']]
    sparse_overlap = len(set(sparse_orig_ids) & set(sparse_hyde_ids))
    
    print(f'   Sparse Search:')
    print(f'     원본 Top-3: {sparse_orig_ids[:3]}')
    print(f'     HyDE Top-3: {sparse_hyde_ids[:3]}')
    print(f'     중복: {sparse_overlap}/5개 ({sparse_overlap/5*100:.0f}%)')
    
    # Dense Search 비교
    dense_orig = dense_retrieve(query, 5, "embeddings_sbert")
    dense_hyde = dense_retrieve(enhanced_query, 5, "embeddings_sbert")
    
    dense_orig_ids = [hit['_source']['docid'] for hit in dense_orig['hits']['hits']]
    dense_hyde_ids = [hit['_source']['docid'] for hit in dense_hyde['hits']['hits']]
    dense_overlap = len(set(dense_orig_ids) & set(dense_hyde_ids))
    
    print(f'   Dense Search:')
    print(f'     원본 Top-3: {dense_orig_ids[:3]}')
    print(f'     HyDE Top-3: {dense_hyde_ids[:3]}')
    print(f'     중복: {dense_overlap}/5개 ({dense_overlap/5*100:.0f}%)\n')
    
    # 임베딩 유사도
    import numpy as np
    emb_orig = embedding_client.get_query_embedding(query)
    emb_hyde = embedding_client.get_query_embedding(enhanced_query)
    similarity = np.dot(emb_orig, emb_hyde) / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_hyde))
    
    print(f'   임베딩 유사도: {similarity:.4f}\n')
    print('-' * 80 + '\n')
