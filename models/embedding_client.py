import numpy as np
from sentence_transformers import SentenceTransformer
# from langchain_upstage import UpstageEmbeddings # Solar Pro2/Upstage 가정
# from google import genai # Gemini 가정

class EmbeddingClient:
    def __init__(self):
        # SBERT (베이스라인 모델)
        self.sbert_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        # self.solar_embedder = UpstageEmbeddings() # 초기화 필요
        # self.gemini_embedder = genai.Client().embeddings # 초기화 필요

    def get_embedding(self, sentences, model_name="sbert"):
        if model_name == "sbert":
            return self.sbert_model.encode(sentences)
        # elif model_name == "solar":
        #     # return self.solar_embedder.embed_documents(sentences)
        #     pass
        # elif model_name == "gemini":
        #     # return self.gemini_embedder.embed_documents(sentences)
        #     pass
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_query_embedding(self, query, model_name="sbert"):
        # 단일 쿼리 임베딩 생성 (dense_retrieve에 사용)
        return self.get_embedding([query], model_name=model_name)[0]

embedding_client = EmbeddingClient()