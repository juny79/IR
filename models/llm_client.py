import os
import json
import time
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, model_name=None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env 파일에 없습니다.")

        # 기본 모델은 env로 오버라이드 가능
        # NOTE: google.generativeai SDK에서는 보통 "models/..." 형태의 모델명을 사용.
        # 예: GEMINI_MODEL_ID=models/gemini-3-flash-preview
        if model_name is None:
            model_name = os.getenv("GEMINI_MODEL_ID", "models/gemini-3-flash-preview")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Phase 8: 게이팅 정밀도 향상 프롬프트 (Chain of Thought + 신뢰도)
        self.persona_query = """
## Role: 과학 지식 질문 분류 전문가

당신의 임무는 사용자의 질문이 "전문 과학 지식 검색이 필요한 질문"인지 정확히 판단하는 것입니다.

## 판단 기준

### 과학 지식 질문 (search 도구 호출 필요) ✅
- 과학적 개념, 원리, 현상에 대한 설명 요구
- 생물학, 화학, 물리학, 지구과학 등 전문 용어 포함
- 학술적이고 객관적인 문체
- 예시:
  * "광합성의 과정은 어떻게 되나요?"
  * "DNA의 이중나선 구조를 설명해주세요"
  * "뉴턴의 제3법칙이란?"
  * "미토콘드리아의 역할은?"

### 일상 대화 (search 도구 호출 불필요) ❌
- 인사, 감정 표현, 의견 문의
- 일상적 대화, 주관적 질문
- 감성적이고 개인적인 문체
- 예시:
  * "안녕하세요"
  * "오늘 날씨 어때요?"
  * "기분이 좋아요"
  * "뭐 하고 있어?"

## 분석 절차 (Chain of Thought)
1. 질문에 과학 전문 용어가 있는가?
2. 객관적 설명/정보를 요구하는가?
3. 학술적 맥락인가?

위 3가지 중 2개 이상 해당하면 과학 질문입니다.
        """
        
    def _call_with_retry(self, func, *args, max_retries=15, initial_wait=5, **kwargs):
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except exceptions.ResourceExhausted:
                wait = initial_wait * (2 ** i)
                print(f"Rate Limit! {wait}초 대기 중...")
                time.sleep(wait)
            except Exception as e:
                print(f"오류: {e}")
                raise
        return None

    def analyze_query(self, messages):
        # Phase 8: Chain of Thought + 신뢰도 점수 추가
        tools = [{"function_declarations": [{
            "name": "search",
            "description": "과학 지식 데이터베이스 검색 (전문적이고 객관적인 과학 지식이 필요할 때만 사용)",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "rationale": {
                        "type": "STRING",
                        "description": "이 질문이 과학 지식 검색이 필요한 이유 (사고 과정)"
                    },
                    "standalone_query": {
                        "type": "STRING",
                        "description": "검색에 최적화된 독립적 과학 질문"
                    },
                    "confidence": {
                        "type": "NUMBER",
                        "description": "과학 지식 질문일 확률 (0.0~1.0, 0.7 이상일 때만 검색 권장)"
                    }
                },
                "required": ["rationale", "standalone_query", "confidence"]
            }
        }]}]
        
        # Few-shot 예시를 포함한 프롬프트 생성
        prompt = f"""{self.persona_query}

## 사용자 질문
{str(messages)}

## 지침
- 과학 질문이면 search 도구를 호출하세요 (rationale, standalone_query, confidence 모두 필수)
- 일상 대화면 도구를 호출하지 말고 친절하게 직접 답하세요
- confidence는 0.0~1.0 사이 값으로, 확신이 없으면 낮게 설정하세요
"""
        
        response = self._call_with_retry(self.model.generate_content, prompt, tools=tools)
        
        # OpenAI 호환 스타일로 변환하여 반환
        class MockCall:
            def __init__(self, args): self.function = type('obj', (object,), {'arguments': json.dumps(args)})
        
        tool_calls = []
        response_text = None
        
        if response and hasattr(response, 'candidates') and response.candidates:
            try:
                if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            tool_calls.append(MockCall(dict(part.function_call.args)))
            except:
                pass
            
            # text 속성에 안전하게 접근
            try:
                if not tool_calls and hasattr(response, 'text'):
                    response_text = response.text
            except:
                pass
        
        return type('obj', (object,), {'content': response_text, 'tool_calls': tool_calls})

    def generate_answer(self, messages, context):
        prompt = f"참고자료:\n{context}\n\n질문: {messages[-1]['content']}"
        response = self._call_with_retry(self.model.generate_content, prompt)
        return response.text if response else "답변을 생성할 수 없습니다."
    
    def generate_hypothetical_answer(self, query):
        """
        HyDE: 질문에 대한 가상의 이상적인 답변 생성
        짧은 질문을 풍부한 문서 형태로 확장하여 검색 정확도 향상
        
        Args:
            query: 검색 질문
            
        Returns:
            가상 답변 (200-300자)
        """
        prompt = f"""다음 질문에 대한 이상적인 답변을 200자 이내로 간결하게 작성하세요.
전문 용어와 핵심 개념을 포함하되, 자연스러운 문장으로 작성하세요.

질문: {query}

답변:"""
        
        try:
            response = self._call_with_retry(
                self.model.generate_content, 
                prompt,
                max_retries=3,
                initial_wait=2
            )
            if response and hasattr(response, 'text'):
                return response.text.strip()
            return ""
        except Exception as e:
            print(f"HyDE 생성 실패: {e}")
            return ""

llm_client = LLMClient()