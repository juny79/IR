import numpy as np
import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pickle
import hashlib
from pathlib import Path

load_dotenv()

class EmbeddingClient:
    def __init__(self):
        # SBERT (베이스라인 모델)
        self.sbert_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        
        # Upstage API 키
        self.upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        self.upstage_url = "https://api.upstage.ai/v1/solar/embeddings"
        
        # Gemini API 키 (Phase 3-3)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        
        # BGE-M3 V3 (Fine-tuned)
        self.bge_m3_v3_model = None
        self.bge_m3_v3_path = "/root/IR/finetuned_bge_m3_v3"
        
        # 쿼리 임베딩 캐시 (비용 절감)
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "query_embeddings.pkl"
        self.cache = self._load_cache()
        
        print("✅ Embedding 모델 로딩 완료:")
        print("   - SBERT: snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        print("   - Upstage: solar-embedding-1-large-query")
        print("   - Gemini: text-embedding-004")
        print(f"   - 캐시: {len(self.cache)} 쿼리 로드됨")
    
    def _load_cache(self):
        """캐시 파일 로드"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """캐시 파일 저장"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            # Interpreter shutdown 시점에는 builtins/open 등이 정리되며 NameError가 날 수 있음.
            # 이 경우는 무해하므로 로그를 남기지 않는다.
            if isinstance(e, NameError) and "open" in str(e):
                return
            print(f"⚠️ 캐시 저장 실패: {e}")
    
    def _get_cache_key(self, query, model_name):
        """캐시 키 생성 (모델명 + 쿼리 해시)"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{model_name}_{query_hash}"

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
        elif model_name == "gemini":
            # Gemini Embedding API 호출 (Phase 3-3)
            embeddings = []
            for text in sentences:
                try:
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
                except Exception as e:
                    print(f"Gemini API 오류: {e}")
                    # 실패 시 zero vector 반환 (768차원)
                    embeddings.append([0.0] * 768)
            
            return np.array(embeddings)
        elif model_name == "bge_m3_v3":
            if self.bge_m3_v3_model is None:
                from FlagEmbedding import BGEM3FlagModel
                self.bge_m3_v3_model = BGEM3FlagModel(self.bge_m3_v3_path, use_fp16=True)
            
            output = self.bge_m3_v3_model.encode(sentences, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            return output['dense_vecs']
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_query_embedding(self, query, model_name="sbert"):
        """
        단일 쿼리 임베딩 생성 (캐싱 적용)
        - Gemini/Upstage API 호출 시 캐시 확인 후 없으면 API 호출
        - SBERT는 로컬 모델이므로 캐싱 생략
        """
        # SBERT는 캐싱 없이 바로 생성 (로컬 모델, 빠름)
        if model_name == "sbert":
            return self.get_embedding([query], model_name=model_name)[0]
        
        if model_name == "bge_m3_v3":
            return self.get_embedding([query], model_name=model_name)[0]
        
        # API 기반 모델은 캐시 확인
        cache_key = self._get_cache_key(query, model_name)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 캐시 미스: API 호출
        embedding = None
        if model_name == "upstage":
            headers = {
                "Authorization": f"Bearer {self.upstage_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "solar-embedding-1-large-query",
                "input": query
            }
            
            response = requests.post(self.upstage_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result['data'][0]['embedding'])
            else:
                print(f"Upstage API 오류: {response.status_code}")
                embedding = np.zeros(4096)
        
        elif model_name == "gemini":
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query"
                )
                embedding = np.array(result['embedding'])
            except Exception as e:
                print(f"Gemini API 오류: {e}")
                embedding = np.zeros(768)
        
        # 캐시 저장 (API 호출 성공 시)
        if embedding is not None:
            self.cache[cache_key] = embedding
            # 100개마다 디스크에 저장 (성능 최적화)
            if len(self.cache) % 100 == 0:
                self._save_cache()
        
        return embedding
    
    def __del__(self):
        """객체 소멸 시 캐시 저장"""
        try:
            # Interpreter shutdown 단계에서는 builtins/open 등이 정리되어 NameError가 날 수 있음.
            # 종료 시점 에러 로그를 남기지 않기 위해 조용히 무시한다.
            self._save_cache()
        except Exception:
            pass

embedding_client = EmbeddingClient()