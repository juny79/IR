import json
import os
from models.solar_client import SolarClient
from dotenv import load_dotenv

load_dotenv()

solar_client = SolarClient(model_name="solar-pro")

def test_solar_rerank():
    messages = [{"role": "user", "content": "지구의 대기 성분 중 가장 많은 것은?"}]
    candidates = [
        ("id1", "지구 대기의 약 78%는 질소입니다."),
        ("id2", "지구 대기의 약 21%는 산소입니다."),
        ("id3", "지구 대기에는 이산화탄소도 포함되어 있습니다.")
    ]
    
    print("Testing Solar Rerank...")
    try:
        # Manually calling the logic from eval_rag_bge_m3_v7.py
        system_prompt = """당신은 한국어 과학 지식 검색 전문가입니다. 
사용자의 대화 맥락과 검색된 3개의 문서 후보(Candidate)가 주어집니다.
질문에 대해 가장 정확하고, 직접적인 해답을 포함하고 있으며, 문맥상 가장 자연스러운 문서를 하나만 선택하세요.

반드시 JSON 형식으로 {"best_index": 0} 와 같이 답변하세요. (0, 1, 2 중 선택)"""

        candidate_text = ""
        for i, (doc_id, content) in enumerate(candidates):
            candidate_text += f"Candidate {i}:\n{content}\n\n"
            
        history = f"사용자: {messages[0]['content']}\n"
        user_prompt = f"## 대화 맥락:\n{history}\n\n## 검색 후보:\n{candidate_text}"
        
        resp = solar_client._call_with_retry(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        print(f"Response: {resp}")
        parsed = json.loads(resp)
        print(f"Parsed Index: {parsed.get('best_index')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_solar_rerank()
