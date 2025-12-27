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
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class SolarClient:
    def __init__(self, model_name=None):
        """
        Solar-pro2 클라이언트 초기화 (HyDE 캐싱 포함)
        
        Args:
            model_name: Upstage Solar 모델 ID. None이면 환경변수/기본값 사용.
                - SOLAR_MODEL_ID env가 설정되어 있으면 그 값을 사용
                - 아니면 기본값 "solar-pro" 사용
        """
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("UPSTAGE_API_KEY가 .env 파일에 없습니다.")
        
        self.api_url = "https://api.upstage.ai/v1/solar/chat/completions"

        if model_name is None:
            model_name = os.getenv("SOLAR_MODEL_ID") or "solar-pro"
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
    
    def _call_with_retry(self, prompt, max_retries=5, initial_wait=2, temperature=0.3, max_tokens=300, response_format=None, use_cache=True):
        """
        Rate Limit 및 오류 처리를 위한 재시도 로직
        """
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_format:
            payload["response_format"] = response_format

        # 일반 호출 캐싱 추가
        cache_key = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"solar_{cache_key}.json"

        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)["content"]
            except:
                pass

        for i in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    if use_cache:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump({"payload": payload, "content": content}, f, ensure_ascii=False, indent=2)
                    
                    return content
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
        # 목표: 대회 기준 "검색 불필요"(일상 대화/감정/인사/잡담/AI 메타/의견) 케이스를 더 잘 잡아 topk=[]로 반환.
        prompt = f"""당신은 '의도(intent) 분류 + 검색 필요 여부 판단 + 검색용 독립쿼리 정규화 + HyDE(가설 문서)' 생성기입니다.

아래 대화를 보고, 마지막 사용자 발화가 "과학/기술/지식에 대한 객관적 설명을 위해 문서 검색이 필요한 질문"인지 판단하세요.

의도(intent) 라벨:
- science_knowledge: 과학/기술 개념/원리/현상/정의/비교/메커니즘 등 객관적 설명 요구
- chitchat: 인사/감정/농담/칭찬/잡담/관계 대화
- assistant_meta: "너는 누구야", "할 수 있어?" 같은 AI 자체/기능 질문
- opinion: 주관적 의견/선호/추천(코퍼스 근거 없이도 답 가능한 형태)
- other: 그 외

출력은 반드시 JSON 1개만. (추가 텍스트/코드블록 금지)

스키마:
{{
    "intent": "science_knowledge|chitchat|assistant_meta|opinion|other",
    "is_science": true/false,
    "confidence": 0.0~1.0,
    "standalone_query": "...",
    "hyde": "...",
    "direct_answer": "..."
}}

규칙:
- is_science=false(=검색 불필요)인 경우:
    - standalone_query: 원문 질문 그대로
    - hyde: ""
    - direct_answer: 1~2문장 짧고 친절한 답
- is_science=true(=검색 필요)인 경우:
    - standalone_query: 검색에 적합하게 정규화(불필요한 감탄/잡담 제거)
    - hyde: {hyde_max_chars}자 이내, 백과사전 스타일 150~{hyde_max_chars}자, 핵심 개념/전문용어/동의어 포함
    - direct_answer: ""

예시:
입력: "안녕 ㅋㅋ"
출력: {{"intent":"chitchat","is_science":false,"confidence":0.95,"standalone_query":"안녕 ㅋㅋ","hyde":"","direct_answer":"안녕하세요! 무엇을 도와드릴까요?"}}

입력: "너는 누구야?"
출력: {{"intent":"assistant_meta","is_science":false,"confidence":0.9,"standalone_query":"너는 누구야?","hyde":"","direct_answer":"저는 질문에 답하고 정보를 정리해드리는 AI 어시스턴트예요."}}

입력: "광합성이 뭐야?"
출력: {{"intent":"science_knowledge","is_science":true,"confidence":0.9,"standalone_query":"광합성의 정의와 과정은 무엇인가?","hyde":"(150~{hyde_max_chars}자 내 과학 설명)","direct_answer":""}}

중요:
- 마지막 발화가 짧아도 이전 대화 맥락이 과학 주제면 science_knowledge로 분류하세요.

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
                # greetings / chit-chat
                "안녕", "반가", "잘 지내", "좋은 아침", "좋은밤", "잘자",
                # thanks / compliment
                "고마", "감사", "땡큐", "최고", "대단", "멋지",
                # emotions
                "힘들", "우울", "기분", "짜증", "행복", "슬퍼",
                # daily-life
                "날씨", "오늘", "내일", "밥", "점심", "저녁", "뭐해",
                # laughter / slang
                "ㅋㅋ", "ㅎㅎ", "ㄷㄷ", "ㅇㅇ",
                # assistant meta
                "너는 누구", "넌 누구", "할 수 있어", "할수있어", "가능해",
            ]
            is_clearly_non_science = any(m in (last_user_text or "") for m in non_science_markers)

            if is_clearly_non_science:
                parsed = {
                    "is_science": False,
                    "confidence": 0.95,
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
            parsed.setdefault("intent", "other")
            parsed.setdefault("is_science", False)
            parsed.setdefault("confidence", 0.0)
            parsed.setdefault("standalone_query", last_user_text)
            parsed.setdefault("hyde", "")
            parsed.setdefault("direct_answer", "")
            parsed["intent"] = str(parsed.get("intent") or "other").strip()
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
        단일 가설 답변 대신 여러 개의 검색용 쿼리 변형을 생성
        BM25 Sparse 검색의 재현율(Recall) 향상
        
        Args:
            query: 원본 검색 질문
            
        환경변수:
            - MULTI_QUERY_COUNT: 생성할 쿼리 개수 (기본 3, 최대 6)

        Returns:
            list: 검색 쿼리 리스트
        """
        # 생성 개수(실험 레버)
        try:
            n = int(os.getenv("MULTI_QUERY_COUNT", "3"))
        except Exception:
            n = 3
        n = max(1, min(6, n))

        # 캐시 키 생성 (개수/프롬프트 버전 포함)
        cache_key = f"multi_v2_{n}_{hashlib.md5(query.encode()).hexdigest()}"
        
        if not hasattr(self, 'multi_query_cache'):
            self.multi_query_cache = {}
        
        if cache_key in self.multi_query_cache:
            return self.multi_query_cache[cache_key]
        
        prompt = f"""당신은 검색 쿼리 최적화 전문가입니다.
    다음 질문에 대해 BM25 검색 재현율을 높이는 '검색용 쿼리 변형'을 {n}개 생성하세요.

    ## 생성 규칙
    - 각 쿼리는 3~12개의 핵심 토큰(단어/구)로 구성 (문장형이어도 괜찮지만 군더더기 없이)
    - 원문의 의미를 유지하되, 서로 다른 관점/표현으로 다양화
    - 가능한 경우 다음 요소를 분산 포함:
      1) 핵심 전문용어(정의/원리)
      2) 동의어/유사어/대체 표기
      3) 관련 개념/메커니즘/원인-결과
      4) 영문 용어(가능하면 괄호로 병기)
    - 너무 일반적인 단어만 나열하지 말 것(예: "설명", "방법")

    ## 질문
    {query}

    ## 출력 형식(JSON만)
    {{"queries": ["...", "...", "..."]}}
    """
        
        try:
            result = self._call_with_retry(prompt, max_retries=2)
            
            if result:
                result = result.strip()

                # JSON 우선 파싱
                try:
                    json_str = result
                    if '```json' in json_str:
                        json_str = json_str.split('```json', 1)[1].split('```', 1)[0]
                    elif '```' in json_str:
                        json_str = json_str.split('```', 1)[1].split('```', 1)[0]

                    parsed = json.loads(json_str.strip())
                    cand = parsed.get('queries') if isinstance(parsed, dict) else None
                    queries = []
                    if isinstance(cand, list):
                        for q in cand:
                            if not isinstance(q, str):
                                continue
                            q = q.strip()
                            if not q:
                                continue
                            queries.append(q)
                except Exception:
                    # 레거시 텍스트 파싱 fallback
                    queries = []
                    for line in result.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        if line[0].isdigit() and line[1:2] == '.':
                            q = line[2:].strip().strip('[]').strip()
                            if q:
                                queries.append(q)

                # 후처리: 중복 제거 + 길이/공백 정리
                uniq = []
                seen = set()
                for q in queries:
                    norm = ' '.join(q.split())
                    key = norm.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(norm)

                if uniq:
                    uniq = uniq[:n]
                    self.multi_query_cache[cache_key] = uniq
                    return uniq

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