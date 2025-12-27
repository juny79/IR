import os
import json
import hashlib
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, model="gpt-4o", cache_dir="/root/IR/cache/llm"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            print(f"✅ OpenAIClient initialized with key: {self.api_key[:5]}***")
        else:
            print("❌ OpenAIClient initialized with NO KEY!")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, payload):
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.md5(payload_str.encode("utf-8")).hexdigest()

    def _call_with_retry(self, prompt, max_retries=5, initial_wait=2, temperature=0, max_tokens=500, response_format=None, use_cache=True):
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

        cache_key = self._get_cache_key(payload)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)["content"]
            except:
                pass

        for i in range(max_retries):
            try:
                response = self.client.chat.completions.create(**payload)
                content = response.choices[0].message.content
                
                if use_cache:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"payload": payload, "content": content}, f, ensure_ascii=False, indent=2)
                
                return content
            except Exception as e:
                print(f"⚠️ OpenAI API 오류 (재시도 {i+1}/{max_retries}): {e}")
                time.sleep(initial_wait * (2 ** i))
        return None

openai_client = OpenAIClient()
