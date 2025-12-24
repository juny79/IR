"""
Solar-pro2 LLM Client for HyDE (Hypothetical Document Embeddings)
Upstage Solar-pro2는 한국어 성능이 우수한 LLM으로, HyDE 쿼리 확장에 최적화

Phase 7C: 이중 게이팅 검증 (Cross-check) 추가
- Gemini 1차 판단 후, Solar Pro 2가 "정말 과학 검색이 필요한가?" 2차 검증
"""

import os
import time
import requests
import pickle
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class SolarClient:
    def __init__(self, model_name="solar-pro"):
        """
        Solar-pro2 클라이언트 초기화 (HyDE 캐싱 포함)
        
        Args:
            model_name: "solar-pro" (Upstage의 최신 모델)
        """
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 .env 파일에 없습니다.")
        
        self.api_url = "https://api.upstage.ai/v1/solar/chat/completions"
        self.model = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # HyDE 캐싱 설정
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.hyde_cache_file = self.cache_dir / "hyde_cache.pkl"
        self.hyde_cache = self._load_hyde_cache()
    
    def _load_hyde_cache(self):
        """HyDE 캐시 파일 로드"""
        if self.hyde_cache_file.exists():
            try:
                with open(self.hyde_cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_hyde_cache(self):
        """HyDE 캐시 파일 저장"""
        try:
            with open(self.hyde_cache_file, 'wb') as f:
                pickle.dump(self.hyde_cache, f)
        except Exception as e:
            print(f"⚠️ HyDE 캐시 저장 실패: {e}")
    
    def _get_hyde_cache_key(self, query):
        """HyDE 캐시 키 생성"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _call_with_retry(self, prompt, max_retries=5, initial_wait=2, temperature=0.3, max_tokens=300):
        """
        Rate Limit 및 오류 처리를 위한 재시도 로직
        
        Args:
            prompt: 프롬프트 텍스트
            max_retries: 최대 재시도 횟수
            initial_wait: 초기 대기 시간(초)
            
        Returns:
            API 응답 텍스트
        """
        for i in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                elif response.status_code == 429:
                    # Rate Limit
                    wait = initial_wait * (2 ** i)
                    print(f"⚠️ Solar API Rate Limit! {wait}초 대기 중...")
                    time.sleep(wait)
                else:
                    print(f"❌ Solar API 오류: {response.status_code} - {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"⏱️ Solar API Timeout! 재시도 {i+1}/{max_retries}")
                time.sleep(initial_wait)
            except Exception as e:
                print(f"❌ Solar API 예외: {e}")
                return None
        
        print(f"❌ Solar API 최대 재시도 초과")
        return None
    
    def generate_hypothetical_answer(self, query):
        """
        HyDE: 질문에 대한 가설적 이상 답변 생성 (Solar-pro2 최적화, 캐싱 포함)
        
        Solar-pro2의 한국어 성능을 활용하여 고품질 가설 답변 생성:
        - 전문 용어 풍부
        - 핵심 개념 포함
        - 200-300자의 상세한 설명 (Phase 4D: 원래 길이)
        - 캐싱으로 비용 절감 (동일 쿼리 시 API 호출 없음)
        
        Args:
            query: 검색 질문
            
        Returns:
            가설 답변 (200-300자)
        """
        # 캐시 확인
        cache_key = self._get_hyde_cache_key(query)
        if cache_key in self.hyde_cache:
            return self.hyde_cache[cache_key]
        
        # ⭐ 최적화: 200자의 간결한 가설 답변 프롬프트
        prompt = f"""당신은 과학 백과사전 전문가입니다. 다음 질문에 대한 전문적인 설명을 작성하세요.

요구사항:
1. 정확히 150-200자의 간결하고 핵심적인 설명
2. 전문 용어와 핵심 개념만 포함 (노이즈 최소화)
3. 백과사전 스타일의 정확하고 구체적인 문장
4. 관련 키워드와 동의어 포함

질문: {query}

전문적인 설명:"""
        
        try:
            result = self._call_with_retry(prompt, max_retries=3, temperature=0.3, max_tokens=300)
            
            if result:
                # 답변 정제: 불필요한 접두사 제거
                result = result.strip()
                if result.startswith("답변:"):
                    result = result[3:].strip()
                if result.startswith("전문적인 설명:"):
                    result = result[9:].strip()
                if result.startswith("설명:"):
                    result = result[3:].strip()
                
                # ⭐ 최적화: 길이 제한 (200자)
                if len(result) > 200:
                    result = result[:200]
                
                # 캐시 저장 (API 호출 성공 시)
                self.hyde_cache[cache_key] = result
                # 20개마다 디스크에 저장
                if len(self.hyde_cache) % 20 == 0:
                    self._save_hyde_cache()
                
                return result
            else:
                return None
                
        except Exception as e:
            print(f"❌ Solar HyDE 생성 실패: {e}")
            return None

    def _extract_json_object(self, text):
        """Solar 응답에서 JSON 객체를 최대한 안전하게 추출"""
        if not text:
            return None

        cleaned = text.strip()
        # 코드펜스 제거
        if '```' in cleaned:
            # ```json ... ``` 또는 ``` ... ```
            parts = cleaned.split('```')
            # 가장 길이가 긴 블록을 후보로
            cleaned = max((p.strip() for p in parts if p.strip()), key=len, default=cleaned)
            # 선행 'json' 라벨 제거
            if cleaned.lower().startswith('json'):
                cleaned = cleaned[4:].strip()

        # 첫 '{' ~ 마지막 '}' 범위 추출
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = cleaned[start:end + 1].strip()
        import json
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def analyze_query_and_hyde(self, messages, hyde_max_chars=200):
        """Gemini 없이 Solar만으로 과학/비과학 판별 + standalone_query + HyDE를 한 번에 생성.

        Returns:
            dict: {
                "is_science": bool,
                "confidence": float,
                "standalone_query": str,
                "hyde": str,
                "direct_answer": str
            }
        """
        # 메시지에서 대화 맥락 텍스트 구성 (멀티턴 follow-up 오판 방지)
        conversation_text = ""
        last_user_text = ""
        try:
            if isinstance(messages, list) and messages:
                lines = []
                for m in messages:
                    role = str(m.get('role', 'user'))
                    content = str(m.get('content', ''))
                    lines.append(f"{role}: {content}")
                    if role == 'user':
                        last_user_text = content
                conversation_text = "\n".join(lines).strip()
            else:
                conversation_text = str(messages)
                last_user_text = conversation_text
        except Exception:
            conversation_text = str(messages)
            last_user_text = conversation_text

        # 캐시
        cache_key = f"analyze_{hashlib.md5((conversation_text + '|' + str(hyde_max_chars)).encode()).hexdigest()}"
        if not hasattr(self, 'analyze_cache'):
            self.analyze_cache = {}
        if cache_key in self.analyze_cache:
            return self.analyze_cache[cache_key]

        # Few-shot + 엄격 JSON 출력 유도
        prompt = f"""당신은 '검색 필요 여부 판단 + 검색용 독립쿼리 정규화 + HyDE(가설 문서)' 생성기입니다.

    아래 대화를 보고, 마지막 사용자 질문이 "문서 검색(코퍼스 기반 답변)이 필요한 지식/설명/정보 질문"인지 판단하세요.

    중요: 이 프로젝트의 코퍼스는 과학에만 한정되지 않을 수 있으므로, 역사/사회/기술/프로그래밍/상식 등 '설명형 지식 질문'은 기본적으로 검색 대상으로 분류하세요.
규칙:
- is_science=false이면(=검색 불필요):

출력 스키마:
{{
- is_science=true이면(=검색 필요):
  \"confidence\": 0.0~1.0,
  \"standalone_query\": \"...\", 
  \"hyde\": \"...\", 
  \"direct_answer\": \"...\"
}}

규칙:
- is_science=false이면:
  - standalone_query는 원문 질문을 그대로 넣기
  - hyde는 빈 문자열
  - direct_answer는 1~2문장 짧은 친절 답변
- is_science=true이면:
  - standalone_query는 검색에 적합한 과학 질문(불필요한 감탄/잡담 제거)
  - hyde는 {hyde_max_chars}자 이내, 백과사전 스타일 150~{hyde_max_chars}자, 핵심 개념/전문용어/동의어 포함
  - direct_answer는 빈 문자열

판단 예시:
Q: "너 정말 똑똑하다!"
A: {{"is_science": false, "confidence": 0.95, "standalone_query": "너 정말 똑똑하다!", "hyde": "", "direct_answer": "고마워요! 무엇을 도와드릴까요?"}}

Q: "오늘 날씨 어때?"
A: {{"is_science": false, "confidence": 0.9, "standalone_query": "오늘 날씨 어때?", "hyde": "", "direct_answer": "제가 실시간 날씨는 확인할 수 없지만, 지역을 알려주면 일반적인 확인 방법을 안내할게요."}}

Q: "python에서 lambda 함수를 언제 사용할 수 있어?"
A: {{"is_science": true, "confidence": 0.85, "standalone_query": "Python에서 lambda(익명 함수)를 언제/어떤 상황에서 사용하는가?", "hyde": "(150~{hyde_max_chars}자 내 기술 설명)", "direct_answer": ""}}

Q: "나이와 코호트 차이를 다 고려하는 디자인 방식은?"
A: {{"is_science": true, "confidence": 0.8, "standalone_query": "나이 효과와 코호트 효과를 함께 고려하는 연구/분석/설계 방법론은 무엇인가?", "hyde": "(150~{hyde_max_chars}자 내 설명)", "direct_answer": ""}}

Q: "광합성이란?"
A: {{"is_science": true, "confidence": 0.9, "standalone_query": "광합성의 정의와 과정은 무엇인가?", "hyde": "(150~{hyde_max_chars}자 내 과학 설명)", "direct_answer": ""}}

Q: "식물 높이 성장 메커니즘"
A: {{"is_science": true, "confidence": 0.85, "standalone_query": "식물의 키(높이) 성장을 조절하는 생리학적 메커니즘은 무엇인가?", "hyde": "(150~{hyde_max_chars}자 내 과학 설명)", "direct_answer": ""}}

중요:
- 아래에는 단일 문장이 아니라 '대화 전체'가 주어질 수 있습니다.
- 마지막 사용자 발화가 "그 이유가 뭐야?" 처럼 짧더라도, 이전 맥락이 과학 주제면 과학 질문으로 분류하세요.
- standalone_query는 반드시 '대화 맥락을 반영'해 독립적으로 완성된 질문으로 만드세요.

대화:
{conversation_text}
"""

        # 분류는 결정적이어야 하므로 temperature=0
        result_text = self._call_with_retry(prompt, max_retries=3, initial_wait=2, temperature=0.0, max_tokens=450)

        parsed = self._extract_json_object(result_text)
        if not parsed:
            # 2차 시도: JSON만 강제하는 짧은 프롬프트
            strict_prompt = f"""반드시 JSON 객체만 출력하세요. 추가 텍스트/설명/코드블록 금지.

스키마:
{{"is_science": true/false, "confidence": 0.0~1.0, "standalone_query": "...", "hyde": "...", "direct_answer": "..."}}

규칙:
- is_science=false이면 hyde는 빈 문자열, direct_answer는 1~2문장.
- is_science=true이면 direct_answer는 빈 문자열, hyde는 {hyde_max_chars}자 이내.

대화:
{conversation_text}
"""
            result_text = self._call_with_retry(strict_prompt, max_retries=2, initial_wait=2, temperature=0.0, max_tokens=450)
            parsed = self._extract_json_object(result_text)

        if not parsed:
            # 파싱 실패 fallback:
            # - 명백한 일상 대화만 비과학 처리
            # - 그 외는 과학으로 처리(과학 질문을 비과학으로 놓쳐 topk=[] 되는 것을 방지)
            non_science_markers = [
                "안녕", "반가", "고마", "감사", "사랑", "좋아", "싫어", "힘들", "우울", "기분",
                "날씨", "오늘", "내일", "밥", "뭐해", "그만", "ㅋㅋ", "ㅎㅎ",
            ]
            is_clearly_non_science = any(m in (last_user_text or "") for m in non_science_markers)

            if is_clearly_non_science:
                parsed = {
                    "is_science": False,
                    "confidence": 0.0,
                    "standalone_query": last_user_text,
                    "hyde": "",
                    "direct_answer": "질문을 조금 더 구체적으로 알려주시면 도와드릴게요."
                }
            else:
                parsed = {
                    "is_science": True,
                    "confidence": 0.0,
                    "standalone_query": last_user_text,
                    "hyde": "",
                    "direct_answer": ""
                }

        # 필드 정규화
        try:
            parsed.setdefault("is_science", False)
            parsed.setdefault("confidence", 0.0)
            parsed.setdefault("standalone_query", last_user_text)
            parsed.setdefault("hyde", "")
            parsed.setdefault("direct_answer", "")
            parsed["confidence"] = float(parsed["confidence"]) if parsed["confidence"] is not None else 0.0
            parsed["standalone_query"] = str(parsed["standalone_query"] or last_user_text).strip()
            parsed["hyde"] = str(parsed["hyde"] or "").strip()
            parsed["direct_answer"] = str(parsed["direct_answer"] or "").strip()
            if len(parsed["hyde"]) > hyde_max_chars:
                parsed["hyde"] = parsed["hyde"][:hyde_max_chars]
        except Exception:
            pass

        # 파싱 실패 fallback은 캐시하지 않음(다음 호출에서 정상 JSON을 받을 기회 유지)
        if float(parsed.get("confidence", 0.0) or 0.0) > 0.0:
            self.analyze_cache[cache_key] = parsed
        return parsed
    
    def verify_science_query(self, query):
        """
        ⭐ Phase 7C: 이중 게이팅 검증 (Cross-check)
        Gemini가 1차로 "과학 질문"이라고 판단한 후, Solar Pro 2가 2차 검증
        
        목적: "일상 대화를 과학 질문으로 오판"하는 것을 방지
        - 오판 시 해당 쿼리의 MAP 점수가 0점이 됨
        
        Args:
            query: Gemini가 추출한 standalone_query
            
        Returns:
            dict: {
                "is_science": bool,    # 과학 질문 여부
                "confidence": str,     # "high", "medium", "low"
                "reason": str          # 판단 이유
            }
        """
        # 캐시 키 생성 (검증용 별도 캐시)
        cache_key = f"verify_{hashlib.md5(query.encode()).hexdigest()}"
        
        # 검증 캐시 확인 (별도 속성)
        if not hasattr(self, 'verify_cache'):
            self.verify_cache = {}
        
        if cache_key in self.verify_cache:
            return self.verify_cache[cache_key]
        
        prompt = f"""당신은 과학 질문 분류 전문가입니다. 다음 질문이 "전문 과학 지식 검색이 필요한 질문"인지 판단하세요.

## 판단 기준
1. **과학 질문 (검색 필요)**: 과학적 개념, 원리, 현상, 실험에 대한 전문적 설명이 필요한 질문
   - 예: "광합성 과정은?", "DNA 구조는?", "뉴턴의 제3법칙은?"
   
2. **비과학/일상 질문 (검색 불필요)**: 인사, 감정, 의견, 일상 대화
   - 예: "안녕?", "오늘 날씨 어때?", "기분이 어때?"

## 질문
{query}

## 응답 형식 (JSON)
{{"is_science": true/false, "confidence": "high/medium/low", "reason": "판단 이유"}}

응답:"""
        
        try:
            result = self._call_with_retry(prompt, max_retries=2)
            
            if result:
                result = result.strip()
                # JSON 파싱 시도
                import json
                try:
                    # JSON 블록 추출
                    if '```json' in result:
                        result = result.split('```json')[1].split('```')[0]
                    elif '```' in result:
                        result = result.split('```')[1].split('```')[0]
                    
                    parsed = json.loads(result.strip())
                    
                    # 캐시 저장
                    self.verify_cache[cache_key] = parsed
                    return parsed
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 기본값 (보수적: 과학으로 판단)
                    default = {"is_science": True, "confidence": "low", "reason": "파싱 실패"}
                    return default
            
            return {"is_science": True, "confidence": "low", "reason": "API 실패"}
            
        except Exception as e:
            print(f"❌ Solar 검증 실패: {e}")
            return {"is_science": True, "confidence": "low", "reason": str(e)}
    
    def generate_multi_query(self, query):
        """
        ⭐ Phase 7D: 멀티 쿼리 생성 (Query Expansion)
        단일 가설 답변 대신 3가지 핵심 키워드 조합 생성
        BM25 Sparse 검색의 재현율(Recall) 향상
        
        Args:
            query: 원본 검색 질문
            
        Returns:
            list: 3가지 검색 쿼리 리스트
        """
        # 캐시 키 생성
        cache_key = f"multi_{hashlib.md5(query.encode()).hexdigest()}"
        
        if not hasattr(self, 'multi_query_cache'):
            self.multi_query_cache = {}
        
        if cache_key in self.multi_query_cache:
            return self.multi_query_cache[cache_key]
        
        prompt = f"""당신은 검색 쿼리 최적화 전문가입니다. 다음 질문에 대해 검색 성능을 높이는 3가지 핵심 키워드 조합을 생성하세요.

## 요구사항
1. 각 키워드 조합은 3-5개 단어로 구성
2. 전문 용어, 동의어, 관련 개념 포함
3. 질문의 다른 측면을 반영

## 질문
{query}

## 응답 형식
1. [키워드 조합 1]
2. [키워드 조합 2]
3. [키워드 조합 3]

응답:"""
        
        try:
            result = self._call_with_retry(prompt, max_retries=2)
            
            if result:
                result = result.strip()
                # 키워드 조합 파싱
                queries = []
                lines = result.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                        # 번호 제거
                        q = line[2:].strip().strip('[]').strip()
                        if q:
                            queries.append(q)
                
                # 최소 1개 이상이면 캐시 저장
                if queries:
                    self.multi_query_cache[cache_key] = queries[:3]  # 최대 3개
                    return queries[:3]
            
            return [query]  # 실패 시 원본 쿼리만 반환
            
        except Exception as e:
            print(f"❌ 멀티 쿼리 생성 실패: {e}")
            return [query]
    
    def generate_hypothetical_answer_batch(self, queries):
        """
        여러 질문에 대한 배치 HyDE 생성 (대량 평가 시 사용)
        
        Args:
            queries: 질문 리스트
            
        Returns:
            가설 답변 리스트
        """
        results = []
        for i, query in enumerate(queries):
            if i > 0 and i % 10 == 0:
                print(f"🔄 Solar HyDE 진행: {i}/{len(queries)}")
                time.sleep(1)  # Rate Limit 방지
            
            result = self.generate_hypothetical_answer(query)
            results.append(result if result else query)  # 실패 시 원본 쿼리 사용
        
        return results
    
    def generate_answer(self, messages, context):
        """
        참고자료를 기반으로 최종 답변 생성 (RAG)
        
        Args:
            messages: 대화 히스토리 (list of dict with 'role' and 'content')
            context: 검색된 문서 컨텍스트
            
        Returns:
            최종 답변
        """
        # 마지막 사용자 메시지 추출
        user_question = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_question = msg.get('content')
                break
        
        if not user_question:
            return "질문을 찾을 수 없습니다."
        
        prompt = f"""당신은 과학 전문 AI 어시스턴트입니다. 제공된 참고자료를 바탕으로 사용자의 질문에 정확하고 체계적으로 답변하세요.

참고자료:
{context}

사용자 질문: {user_question}

답변 작성 지침:
1. 참고자료의 내용을 **최우선**으로 활용
2. 참고자료에 명시된 전문 용어, 개념, 수치를 정확히 인용
3. 참고자료에 없는 내용은 절대 추측하거나 창작하지 말 것
4. 구조화된 형식(번호, 항목, 단계 등)으로 가독성 향상
5. 자연스럽고 이해하기 쉬운 한국어로 작성
6. 참고자료가 부족하면 "참고자료에 명시되지 않았습니다"라고 명시

답변:"""
        
        try:
            result = self._call_with_retry(prompt, max_retries=3)
            
            if result:
                # 답변 정제
                result = result.strip()
                if result.startswith("답변:"):
                    result = result[3:].strip()
                
                return result
            else:
                return "죄송합니다. 답변 생성에 실패했습니다."
                
        except Exception as e:
            print(f"❌ Solar 답변 생성 실패: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."


# 전역 Solar 클라이언트 인스턴스
solar_client = SolarClient()


if __name__ == "__main__":
    # 테스트 코드
    print("=== Solar-pro2 HyDE 테스트 ===\n")
    
    test_queries = [
        "광합성이란?",
        "DNA의 구조는?",
        "뉴턴의 제3법칙을 설명하세요."
    ]
    
    for query in test_queries:
        print(f"📝 질문: {query}")
        hyde = solar_client.generate_hypothetical_answer(query)
        print(f"✨ HyDE 답변: {hyde}\n")
        print(f"📏 길이: {len(hyde) if hyde else 0}자\n")
        print("-" * 80)    
    # 캐시 저장
    solar_client._save_hyde_cache()
    print(f"\n✅ HyDE 캐시 {len(solar_client.hyde_cache)}개 저장 완료")