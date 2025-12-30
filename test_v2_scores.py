import torch
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import json

def test_v2_scores():
    model = BGEM3FlagModel('/root/IR/finetuned_bge_m3_v2', use_fp16=True)
    query = '혼합물의 특성에 대해 알려줘.'
    docs = [
        '이 혼합물은 땅콩, 해바라기 씨, 건포도, 아몬드, 초콜릿 조각으로 구성되어 있습니다. 이 혼합물이 혼합물인 이유는 구성 요소가 원래의 성질을 유지하기 때문입니다.',
        '혼합물과 용액은 물리적 현상으로서, 두 가지 다른 물질이 결합하여 형성됩니다. 혼합물은 두 개 이상의 물질이 섞여 있으며, 각각의 물질은 그 자체로 존재합니다.'
    ]

    q_vecs = model.encode([query], return_dense=True, return_sparse=True)
    d_vecs = model.encode(docs, return_dense=True, return_sparse=True)

    # Dense Score
    dense_scores = q_vecs['dense_vecs'] @ d_vecs['dense_vecs'].T
    # Sparse Score
    sparse_scores = model.compute_lexical_matching_score(q_vecs['lexical_weights'], d_vecs['lexical_weights'])

    print(f'Dense: {dense_scores}')
    print(f'Sparse: {sparse_scores}')
    
    # Hybrid (0.5:0.5)
    hybrid = 0.5 * dense_scores[0] + 0.5 * sparse_scores[0]
    print(f'Hybrid: {hybrid}')

if __name__ == "__main__":
    test_v2_scores()
