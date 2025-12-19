import os
import json
import time
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, model_name="gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env 파일에 없습니다.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # MAP 극대화용 프롬프트
        self.persona_query = """
        ## Role: 과학 상식 전문가
        - 과학 질문이면 'search' 도구를 호출하여 'standalone_query'를 생성하라.
        - 일상 인사나 비과학 질문이면 도구를 호출하지 말고 직접 답하라. (MAP 확보 핵심)
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
        tools = [{"function_declarations": [{
            "name": "search",
            "description": "과학 지식 검색",
            "parameters": {"type": "OBJECT", "properties": {"standalone_query": {"type": "STRING"}}, "required": ["standalone_query"]}
        }]}]
        
        response = self._call_with_retry(self.model.generate_content, str(messages), tools=tools)
        
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

llm_client = LLMClient()


llm_client = LLMClient()