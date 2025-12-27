import os
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv

class GeminiClient:
    def __init__(self, model_name="gemini-3-flash-preview"): # 사용자가 요청한 모델명으로 나중에 변경 가능
        # Ensure .env is loaded even when this module is imported directly.
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY)가 .env 파일에 없습니다.")

        # Some SDK error messages reference GOOGLE_API_KEY; keep them aligned.
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def _call_with_retry(self, prompt, max_retries=5, initial_wait=2, temperature=0, max_tokens=500, response_format=None):
        # Gemini는 response_format을 다르게 처리하지만, 여기서는 텍스트로 받고 파싱하는 방식으로 통일
        def _safe_response_text(resp):
            if resp is None:
                return None
            # Prefer explicit parts if present (avoids response.text quick-accessor errors)
            try:
                if hasattr(resp, "candidates") and resp.candidates:
                    cand = resp.candidates[0]
                    if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
                        parts = []
                        for p in cand.content.parts:
                            t = getattr(p, "text", None)
                            if t:
                                parts.append(t)
                        if parts:
                            return "".join(parts)
            except Exception:
                pass

            # Fallback to .text accessor if safe
            try:
                return getattr(resp, "text", None)
            except Exception:
                return None

        for i in range(max_retries):
            try:
                # messages 형식 변환
                contents = []
                if isinstance(prompt, list):
                    for m in prompt:
                        role = "user" if m["role"] == "user" else "model"
                        contents.append({"role": role, "parts": [m["content"]]})
                else:
                    contents = [{"role": "user", "parts": [prompt]}]

                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                
                if response_format and response_format.get("type") == "json_object":
                    generation_config["response_mime_type"] = "application/json"

                response = self.model.generate_content(
                    contents,
                    generation_config=generation_config
                )
                text = _safe_response_text(response)
                # If Gemini returns no usable text parts (often blocked/empty), don't retry-loop.
                if text is None:
                    return None
                return text
            except Exception as e:
                msg = str(e)
                lower = msg.lower()

                # The legacy google.generativeai SDK sometimes raises an exception when the
                # candidate has no Parts (blocked/empty), e.g.:
                # "Invalid operation: The `response.text` quick accessor requires ... finish_reason is 2"
                # Retrying in those cases just burns time; callers can safely fallback.
                non_retriable_signals = (
                    ("response.text" in lower and "quick accessor" in lower)
                    or ("no valid part" in lower)
                    or ("requires the response to contain" in lower and "part" in lower)
                    or ("finish_reason" in lower and " 2" in lower)
                    or ("finish_reason" in lower and "is 2" in lower)
                    or ("candidate" in lower and "finish_reason" in lower)
                    or ("blocked" in lower)
                    or ("safety" in lower and "finish" in lower)
                    or ("invalid operation" in lower and "finish_reason" in lower)
                    or ("invalid operation" in lower)
                )

                if non_retriable_signals:
                    print(f"⚠️ Gemini API 오류(비재시도): {msg}")
                    return None

                # Stability-first: for this project we prefer finishing the run over spending
                # time on retries. Callers can fallback to non-LLM ranking.
                print(f"⚠️ Gemini API 오류(즉시 fallback): {msg}")
                return None
        return None

gemini_client = GeminiClient()
