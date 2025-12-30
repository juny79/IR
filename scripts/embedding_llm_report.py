import os
import sys
import io
import math
import json
import contextlib
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _safe_bool_env(name: str) -> str:
    return "set" if os.getenv(name) else "unset"


def _l2_norm(vec) -> float:
    try:
        s = 0.0
        for x in vec:
            s += float(x) * float(x)
        return math.sqrt(s)
    except Exception:
        return float("nan")


def _is_all_zero(vec, eps: float = 1e-12) -> bool:
    try:
        for x in vec:
            if abs(float(x)) > eps:
                return False
        return True
    except Exception:
        return False


def main():
    # Load .env explicitly (avoid stack-frame edge cases)
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / f"EMBEDDING_LLM_REPORT_{ts}.md"

    # LLM (chat) model ids
    gemini_llm_default = "models/gemini-3-flash-preview"
    gemini_llm_env = os.getenv("GEMINI_MODEL_ID")
    gemini_llm_effective = gemini_llm_env or gemini_llm_default

    solar_chat_effective = None
    solar_env = os.getenv("SOLAR_MODEL_ID")
    try:
        # Import solar_client without printing secrets
        from models.solar_client import solar_client
        solar_chat_effective = getattr(solar_client, "model", None)
    except Exception as e:
        solar_chat_effective = f"<unavailable: {type(e).__name__}>"

    # Embedding models (hard-coded in EmbeddingClient)
    gemini_embedding_model = "models/text-embedding-004"
    upstage_query_model = "solar-embedding-1-large-query"
    upstage_passage_model = "solar-embedding-1-large-passage"

    # Import embedding client with stdout suppressed (it prints model info)
    embedding_client = None
    import_note = None
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            from models.embedding_client import embedding_client as _ec
            embedding_client = _ec
        except Exception as e:
            import_note = f"EmbeddingClient import failed: {type(e).__name__}: {str(e)[:120]}"

    tests = []

    def add_test(name: str, fn):
        try:
            vec = fn()
            # Normalize return shapes
            if hasattr(vec, "tolist"):
                vec_list = vec.tolist()
            else:
                vec_list = list(vec)
            dim = len(vec_list)
            norm = _l2_norm(vec_list)
            tests.append({
                "name": name,
                "ok": True,
                "dim": dim,
                "l2_norm": norm,
                "all_zero": _is_all_zero(vec_list),
            })
        except Exception as e:
            tests.append({
                "name": name,
                "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })

    if embedding_client is not None:
        # SBERT is local
        add_test("SBERT (sentence_transformers) query", lambda: embedding_client.get_query_embedding("광합성이란?", model_name="sbert"))

        # Gemini embeddings
        add_test("Gemini embedding retrieval_query", lambda: embedding_client.get_query_embedding("광합성이란?", model_name="gemini"))
        add_test("Gemini embedding retrieval_document", lambda: embedding_client.get_embedding(["광합성은 빛 에너지를 화학 에너지로 바꾸는 과정이다."], model_name="gemini")[0])

        # Upstage embeddings
        add_test("Upstage embedding query", lambda: embedding_client.get_query_embedding("광합성이란?", model_name="upstage"))
        add_test("Upstage embedding passage", lambda: embedding_client.get_embedding(["광합성은 식물이 빛을 이용해 포도당을 만드는 과정이다."], model_name="upstage")[0])

    # Write report (no secrets)
    lines = []
    lines.append(f"# Embedding/LLM Config Report ({ts})\n")

    lines.append("## Environment\n")
    lines.append(f"- GEMINI_API_KEY: {_safe_bool_env('GEMINI_API_KEY')}")
    lines.append(f"- UPSTAGE_API_KEY: {_safe_bool_env('UPSTAGE_API_KEY')}")
    lines.append(f"- ES_PASSWORD: {_safe_bool_env('ES_PASSWORD')}\n")

    lines.append("## Chat/Analysis LLM Models\n")
    lines.append(f"- Gemini LLM effective: `{gemini_llm_effective}` (GEMINI_MODEL_ID env: {'<unset>' if not gemini_llm_env else gemini_llm_env})")
    lines.append(f"- Solar chat effective: `{solar_chat_effective}` (SOLAR_MODEL_ID env: {'<unset>' if not solar_env else solar_env})\n")

    lines.append("## Embedding Models (Configured in Code)\n")
    lines.append(f"- Gemini embedding model: `{gemini_embedding_model}`")
    lines.append(f"- Upstage query embedding model: `{upstage_query_model}`")
    lines.append(f"- Upstage passage embedding model: `{upstage_passage_model}`\n")

    lines.append("## Health Checks\n")
    if import_note:
        lines.append(f"- EmbeddingClient: ❌ {import_note}")
    elif not tests:
        lines.append("- No tests executed.")
    else:
        for t in tests:
            if t.get("ok"):
                lines.append(
                    f"- {t['name']}: ✅ dim={t['dim']} l2_norm={t['l2_norm']:.4f} all_zero={t['all_zero']}"
                )
            else:
                lines.append(f"- {t['name']}: ❌ {t.get('error')}")

    lines.append("\n## Notes\n")
    lines.append("- ‘최신 LLM’과 ‘임베딩 모델’은 별개입니다. 이 프로젝트에서 임베딩은 Gemini `text-embedding-004` / Upstage `solar-embedding-1-large-*`로 호출됩니다.")
    lines.append("- Gemini/Upstage API 호출 실패 시 코드가 zero-vector로 fallback 하므로, 위 health check에서 all_zero=true가 나오면 ‘모델/키/호출 실패’를 의미할 가능성이 큽니다.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
