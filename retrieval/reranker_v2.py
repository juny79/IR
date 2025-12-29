from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import torch

class RerankerV2:
    def __init__(self, cross_model_name='BAAI/bge-reranker-v2-m3', bi_model_path='/root/IR/finetuned_bge_m3_v2'):
        print(f"Loading Cross-Encoder: {cross_model_name}...")
        self.cross_encoder = CrossEncoder(cross_model_name, max_length=512)
        
        print(f"Loading Fine-tuned Bi-Encoder: {bi_model_path}...")
        self.bi_encoder = BGEM3FlagModel(bi_model_path, use_fp16=True)
        
        print("✅ RerankerV2 (Ensemble) 로딩 완료")
    
    def rerank(self, query, documents, top_k=5, batch_size=32):
        """
        Cross-Encoder와 Fine-tuned Bi-Encoder 점수를 앙상블하여 재정렬
        """
        if not documents:
            return []
        
        doc_ids = [doc[0] for doc in documents]
        doc_contents = [doc[1] for doc in documents]
        
        # 1. Cross-Encoder Scores
        pairs = [[query, content[:512]] for content in doc_contents]
        cross_scores = self.cross_encoder.predict(
            pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        
        # 2. Bi-Encoder Scores (Dense + Sparse Hybrid)
        q_vecs = self.bi_encoder.encode([query], return_dense=True, return_sparse=True)
        d_vecs = self.bi_encoder.encode(doc_contents, return_dense=True, return_sparse=True)
        
        dense_scores = (q_vecs['dense_vecs'] @ d_vecs['dense_vecs'].T)[0]
        sparse_scores = self.bi_encoder.compute_lexical_matching_score(q_vecs['lexical_weights'], d_vecs['lexical_weights'])[0]
        
        # Bi-Encoder Hybrid Score (0.5:0.5)
        bi_scores = 0.5 * dense_scores + 0.5 * sparse_scores
        
        # 3. Ensemble (Normalization & Weighted Sum)
        # Cross-Encoder 점수는 보통 -10 ~ 10 사이, Bi-Encoder 점수는 0 ~ 1 사이 (Dense) + Sparse
        # 간단하게 Rank 기반 앙상블(RRF)을 하거나, 점수 스케일을 맞춰서 합산
        
        # 점수 정규화 (Min-Max)
        def normalize(s):
            if len(s) <= 1: return s
            s_min, s_max = s.min(), s.max()
            if s_max == s_min: return np.zeros_like(s)
            return (s - s_min) / (s_max - s_min)
        
        norm_cross = normalize(cross_scores)
        norm_bi = normalize(bi_scores)
        
        # 최종 점수 (Cross 0.7 : Bi 0.3)
        final_scores = 0.7 * norm_cross + 0.3 * norm_bi
        
        sorted_indices = np.argsort(final_scores)[::-1][:top_k]
        return [doc_ids[idx] for idx in sorted_indices]

# 싱글톤 인스턴스
reranker_v2 = RerankerV2()
