import numpy as np
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

class EmbeddingClient:
    def __init__(self):
        # SBERT (베이스라인 모델)
        self.sbert_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        
        # Upstage API 키
        self.upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        self.upstage_url = "https://api.upstage.ai/v1/solar/embeddings"
        
        print("✅ Embedding 모델 로딩 완료:")
        print("   - SBERT: snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        print("   - Upstage: solar-embedding-1-large-query")

    def get_embedding(self, sentences, model_name="sbert"):
        if model_name == "sbert":
            return self.sbert_model.encode(sentences)
        elif model_name == "upstage":
            # Upstage API 호출
            headers = {
                "Authorization": f"Bearer {self.upstage_api_key}",
                "Content-Type": "application/json"
            }
            
            embeddings = []
            for text in sentences:
                payload = {
                    "model": "solar-embedding-1-large-passage",
                    "input": text
                }
                
                response = requests.post(self.upstage_url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    embeddings.append(result['data'][0]['embedding'])
                else:
                    print(f"Upstage API 오류: {response.status_code}")
                    # 실패 시 zero vector 반환
                    embeddings.append([0.0] * 4096)
            
            return np.array(embeddings)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_query_embedding(self, query, model_name="sbert"):
        # 단일 쿼리 임베딩 생성 (dense_retrieve에 사용)
        if model_name == "upstage":
            headers = {
                "Authorization": f"Bearer {self.upstage_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "solar-embedding-1-large-query",  # query용 모델
                "input": query
            }
            
            response = requests.post(self.upstage_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return np.array(result['data'][0]['embedding'])
            else:
                print(f"Upstage API 오류: {response.status_code}")
                return np.zeros(4096)
        else:
            return self.get_embedding([query], model_name=model_name)[0]

embedding_client = EmbeddingClient()