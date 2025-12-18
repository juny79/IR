import os
from openai import OpenAI
# from google import genai # Gemini API 클라이언트
# from anthropic import Anthropic # Claude API 클라이언트

class LLMClient:
    def __init__(self, model="gpt-3.5-turbo-1106"): # 실제 최고 성능 모델로 교체 (예: gemini-3.0-pro)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
        self.model = model
        
        # Function Calling 도구 정의 (기존 베이스라인 유지)
        self.tools = [
            # ... (search function 정의) ...
        ]
        
        # MAP 극대화를 위한 의도 분석 프롬프트 (최신 모델에 최적화)
        self.persona_function_calling = """
## Role: 과학 상식 전문가 (Query Analyst)

## Instruction
- 사용자의 대화 기록을 분석하여 **'과학 지식에 관한 주제'**로 질문한 경우에만 search API를 호출해야 한다.
- **일상 대화 (chit-chat)**, 사적인 질문, 단순 인사 등 과학 상식과 관련되지 않은 나머지 대화 메시지에는 **절대 search API를 호출하지 않고**, 적절하고 간결한 답변을 생성한다. **(MAP 극대화 핵심)**
- 멀티턴 대화인 경우, 이전 문맥을 통합하여 **가장 명확하고 검색에 적합한 하나의 'standalone_query'**를 생성한다.
"""
        self.persona_qa = """
## Role: 과학 상식 전문가 (Answer Generator)

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""
    
    def analyze_query(self, messages):
        # 1. 의도 분석 및 쿼리 생성
        msg = [{"role": "system", "content": self.persona_function_calling}] + messages
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=msg,
                tools=self.tools,
                temperature=0,
                timeout=10
            )
            return result.choices[0].message
        except Exception:
            return None

    def generate_answer(self, messages, retrieved_context):
        # 3. 최종 QA
        content = f"검색된 참고 자료:\n{retrieved_context}"
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": self.persona_qa}] + messages
        try:
            qa_result = self.client.chat.completions.create(
                model=self.model,
                messages=msg,
                temperature=0,
                timeout=30
            )
            return qa_result.choices[0].message.content
        except Exception:
            return "정보가 부족해서 답을 할 수 없습니다."

llm_client = LLMClient(model="gpt-4-turbo-2024-04-09") # 최신 GPT-4 모델 예시