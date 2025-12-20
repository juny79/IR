from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
        """
        다국어 Reranker 초기화
        bge-reranker-v2-m3: 한국어를 포함한 다국어 지원
        """
        print(f"Reranker 모델 로딩 중: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        print("✅ Reranker 로딩 완료")
    
    def rerank(self, query, documents, top_k=5, batch_size=32):
        """
        질문과 문서들을 Cross-Encoder로 재정렬 (배치 처리 최적화)
        
        Args:
            query: 검색 질문
            documents: [(doc_id, content), ...] 리스트
            top_k: 최종 반환할 상위 문서 개수
            batch_size: 배치 처리 크기 (기본 32)
            
        Returns:
            [doc_id, ...] 상위 top_k개의 문서 ID 리스트
        """
        if not documents:
            return []
        
        # 질문-문서 쌍 생성
        pairs = [[query, doc[1][:512]] for doc in documents]  # 512자로 제한하여 속도 향상
        
        # Cross-Encoder로 관련성 점수 계산 (배치 처리)
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # 점수 기준 내림차순 인덱스
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        # 상위 top_k개의 문서 ID만 반환
        top_docs = [documents[idx][0] for idx in sorted_indices]
        
        return top_docs

# 싱글톤 인스턴스
reranker = Reranker()
