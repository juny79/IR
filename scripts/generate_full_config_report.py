import os
import sys
import io
import json
import contextlib
import math
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_set(name: str) -> str:
    return "set" if os.getenv(name) else "unset"


def _env(name: str, default: str = "<unset>") -> str:
    v = os.getenv(name)
    return default if v is None else v


def _env_bool(name: str, default: bool = False) -> str:
    v = os.getenv(name)
    if v is None:
        return str(default).lower()
    return v.strip().lower()


def _mask_value(v: str) -> str:
    # Never print secrets. For safety, if a value looks like a key, mask it.
    if not v:
        return v
    low = v.lower()
    if any(k in low for k in ["key", "token", "secret", "password", "sk-"]):
        return "<masked>"
    return v


def main():
    # Load .env explicitly (do not rely on stack frames)
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / f"FULL_CONFIG_REPORT_{ts}.md"

    # Optional health checks (will make a small number of API calls)
    run_health_checks = (os.getenv("RUN_HEALTH_CHECKS", "0").strip().lower() in ("1", "true", "t", "yes", "y", "on"))

    # Effective model ids
    gemini_llm_default = "models/gemini-3-flash-preview"
    gemini_llm_env = os.getenv("GEMINI_MODEL_ID")
    gemini_llm_effective = gemini_llm_env or gemini_llm_default

    solar_chat_effective = "<unavailable>"
    try:
        from models.solar_client import solar_client
        solar_chat_effective = str(getattr(solar_client, "model", "<unknown>"))
    except Exception as e:
        solar_chat_effective = f"<unavailable: {type(e).__name__}>"

    # Embedding model ids (as implemented)
    sbert_model = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    gemini_embedding_model = "models/text-embedding-004"
    upstage_query_model = "solar-embedding-1-large-query"
    upstage_passage_model = "solar-embedding-1-large-passage"

    # Try to import EmbeddingClient quietly (it prints info)
    embedding_import = "ok"
    embedding_client_obj = None
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            from models.embedding_client import embedding_client as _ec
            embedding_client_obj = _ec
        except Exception as e:
            embedding_import = f"failed: {type(e).__name__}: {str(e)[:120]}"

    # Key env toggles to report (safe values only)
    toggles = [
        # Retrieval/fusion
        "USE_RRF",
        "RRF_K",
        "TOP_K_RETRIEVE",
        "CANDIDATE_POOL_SIZE",
        "USE_MULTI_EMBEDDING",
        "USE_GEMINI_ONLY",
        "VOTING_WEIGHTS_JSON",
        "DENSE_EMBEDDING_FIELDS",
        "DENSE_K_PER_FIELD",
        "DENSE_K_PER_FIELD_MAP",
        # Gating/no-search
        "USE_SOLAR_ANALYZER",
        "USE_GATING",
        "NO_SEARCH_CONFIDENCE_THRESHOLD",
        "ENABLE_INTENT_NO_SEARCH",
        "NO_SEARCH_STRICT_THRESHOLD",
        "NO_SEARCH_HEURISTIC_THRESHOLD",
        "NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF",
        # Query expansion
        "HYDE_MAX_LENGTH",
        "USE_MULTI_QUERY",
        "MULTI_QUERY_COUNT",
        # Model overrides
        "GEMINI_MODEL_ID",
        "SOLAR_MODEL_ID",
    ]

    # Also capture effective defaults from eval_rag (when env vars are unset)
    eval_rag_defaults = {}
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            import eval_rag as _eval
            for k in [
                "VOTING_WEIGHTS",
                "USE_MULTI_EMBEDDING",
                "USE_GEMINI_ONLY",
                "TOP_K_RETRIEVE",
                "USE_RRF",
                "RRF_K",
                "USE_GATING",
                "HYDE_MAX_LENGTH",
                "USE_SOLAR_ANALYZER",
                "USE_MULTI_QUERY",
                "NO_SEARCH_CONFIDENCE_THRESHOLD",
                "CANDIDATE_POOL_SIZE",
                "ENABLE_INTENT_NO_SEARCH",
                "NO_SEARCH_STRICT_THRESHOLD",
                "NO_SEARCH_HEURISTIC_THRESHOLD",
                "NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF",
            ]:
                if hasattr(_eval, k):
                    eval_rag_defaults[k] = getattr(_eval, k)
        except Exception as e:
            eval_rag_defaults = {"_error": f"{type(e).__name__}: {str(e)[:160]}"}

    def l2_norm(vec) -> float:
        try:
            s = 0.0
            for x in vec:
                s += float(x) * float(x)
            return math.sqrt(s)
        except Exception:
            return float("nan")

    def is_all_zero(vec, eps: float = 1e-12) -> bool:
        try:
            for x in vec:
                if abs(float(x)) > eps:
                    return False
            return True
        except Exception:
            return False

    health = []
    if run_health_checks:
        # Embedding health checks: 1 call each
        if embedding_client_obj is None:
            health.append({"name": "EmbeddingClient import", "ok": False, "error": embedding_import})
        else:
            # SBERT
            try:
                v = embedding_client_obj.get_query_embedding("광합성이란?", model_name="sbert")
                v = v.tolist() if hasattr(v, "tolist") else list(v)
                health.append({"name": "SBERT query", "ok": True, "dim": len(v), "l2_norm": l2_norm(v), "all_zero": is_all_zero(v)})
            except Exception as e:
                health.append({"name": "SBERT query", "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})

            # Gemini embedding (query/doc)
            try:
                v = embedding_client_obj.get_query_embedding("광합성이란?", model_name="gemini")
                v = v.tolist() if hasattr(v, "tolist") else list(v)
                health.append({"name": "Gemini embedding retrieval_query", "ok": True, "dim": len(v), "l2_norm": l2_norm(v), "all_zero": is_all_zero(v)})
            except Exception as e:
                health.append({"name": "Gemini embedding retrieval_query", "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})

            try:
                v = embedding_client_obj.get_embedding(["광합성은 빛 에너지를 화학 에너지로 바꾸는 과정이다."], model_name="gemini")[0]
                v = v.tolist() if hasattr(v, "tolist") else list(v)
                health.append({"name": "Gemini embedding retrieval_document", "ok": True, "dim": len(v), "l2_norm": l2_norm(v), "all_zero": is_all_zero(v)})
            except Exception as e:
                health.append({"name": "Gemini embedding retrieval_document", "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})

            # Upstage embedding (query/passage)
            try:
                v = embedding_client_obj.get_query_embedding("광합성이란?", model_name="upstage")
                v = v.tolist() if hasattr(v, "tolist") else list(v)
                health.append({"name": "Upstage embedding query", "ok": True, "dim": len(v), "l2_norm": l2_norm(v), "all_zero": is_all_zero(v)})
            except Exception as e:
                health.append({"name": "Upstage embedding query", "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})

            try:
                v = embedding_client_obj.get_embedding(["광합성은 식물이 빛을 이용해 포도당을 만드는 과정이다."], model_name="upstage")[0]
                v = v.tolist() if hasattr(v, "tolist") else list(v)
                health.append({"name": "Upstage embedding passage", "ok": True, "dim": len(v), "l2_norm": l2_norm(v), "all_zero": is_all_zero(v)})
            except Exception as e:
                health.append({"name": "Upstage embedding passage", "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})

        # Solar chat ping (1 call)
        try:
            from models.solar_client import solar_client
            txt = solar_client.generate_hypothetical_answer("광합성이란?")
            health.append({"name": "Solar chat ping (HyDE)", "ok": True, "response_chars": len(txt or "")})
        except Exception as e:
            health.append({"name": "Solar chat ping (HyDE)", "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})

        # Gemini chat ping (1 call) - optional, can be expensive; keep lightweight
        try:
            from models.llm_client import llm_client
            # A single very short generation without tools
            resp = llm_client.model.generate_content("ping")
            text = getattr(resp, "text", "") if resp is not None else ""
            health.append({"name": "Gemini chat ping", "ok": True, "response_chars": len(text or "")})
        except Exception as e:
            health.append({"name": "Gemini chat ping", "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    # Build markdown
    lines = []
    lines.append(f"# Full Config Report ({ts})\n")

    lines.append("## Secrets (presence only)\n")
    for k in ["GEMINI_API_KEY", "UPSTAGE_API_KEY", "ES_PASSWORD", "OPENAI_API_KEY"]:
        lines.append(f"- {k}: {_is_set(k)}")

    lines.append("\n## Effective Model IDs\n")
    lines.append(f"- Gemini LLM effective: `{gemini_llm_effective}` (GEMINI_MODEL_ID env: {'<unset>' if not gemini_llm_env else gemini_llm_env})")
    lines.append(f"- Solar chat effective: `{solar_chat_effective}` (SOLAR_MODEL_ID env: {'<unset>' if not os.getenv('SOLAR_MODEL_ID') else os.getenv('SOLAR_MODEL_ID')})")
    lines.append(f"- EmbeddingClient import: `{embedding_import}`\n")

    lines.append("## Embedding Models (as coded)\n")
    lines.append(f"- SBERT: `{sbert_model}`")
    lines.append(f"- Gemini embedding: `{gemini_embedding_model}`")
    lines.append(f"- Upstage embedding (query): `{upstage_query_model}`")
    lines.append(f"- Upstage embedding (passage): `{upstage_passage_model}`\n")

    lines.append("## Runtime Toggles / Hyperparameters\n")
    for name in toggles:
        v = os.getenv(name)
        if v is None:
            lines.append(f"- {name}: <unset>")
        else:
            lines.append(f"- {name}: `{_mask_value(v)}`")

    lines.append("\n## Effective Defaults (from eval_rag.py)\n")
    if eval_rag_defaults.get("_error"):
        lines.append(f"- failed_to_import_eval_rag: `{eval_rag_defaults['_error']}`")
    else:
        for k, v in eval_rag_defaults.items():
            # keep compact JSON for lists
            if isinstance(v, (list, dict)):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v)
            lines.append(f"- {k}: `{v_str}`")

    # Reproducible command that pins effective defaults explicitly.
    # This helps reproduce the exact behavior even if code defaults change later.
    lines.append("\n## Repro Commands\n")
    lines.append("- 아래 커맨드는 ‘현재 효과적 기본값’을 env로 명시해 재현성을 높입니다. (비밀키/패스워드는 포함하지 않음)")

    env_pairs = []
    # Model IDs
    env_pairs.append(("GEMINI_MODEL_ID", gemini_llm_effective))
    if isinstance(solar_chat_effective, str) and not solar_chat_effective.startswith("<unavailable"):
        env_pairs.append(("SOLAR_MODEL_ID", solar_chat_effective))

    # Map eval_rag defaults back to env names
    if not eval_rag_defaults.get("_error"):
        # Voting weights
        if "VOTING_WEIGHTS" in eval_rag_defaults:
            env_pairs.append(("VOTING_WEIGHTS_JSON", json.dumps(eval_rag_defaults["VOTING_WEIGHTS"], ensure_ascii=False)))

        def _b(v):
            return "true" if bool(v) else "false"

        mapping = {
            "USE_MULTI_EMBEDDING": ("USE_MULTI_EMBEDDING", _b(eval_rag_defaults.get("USE_MULTI_EMBEDDING"))),
            "USE_GEMINI_ONLY": ("USE_GEMINI_ONLY", _b(eval_rag_defaults.get("USE_GEMINI_ONLY"))),
            "TOP_K_RETRIEVE": ("TOP_K_RETRIEVE", str(eval_rag_defaults.get("TOP_K_RETRIEVE"))),
            "USE_RRF": ("USE_RRF", _b(eval_rag_defaults.get("USE_RRF"))),
            "RRF_K": ("RRF_K", str(eval_rag_defaults.get("RRF_K"))),
            "USE_GATING": ("USE_GATING", _b(eval_rag_defaults.get("USE_GATING"))),
            "HYDE_MAX_LENGTH": ("HYDE_MAX_LENGTH", str(eval_rag_defaults.get("HYDE_MAX_LENGTH"))),
            "USE_SOLAR_ANALYZER": ("USE_SOLAR_ANALYZER", _b(eval_rag_defaults.get("USE_SOLAR_ANALYZER"))),
            "USE_MULTI_QUERY": ("USE_MULTI_QUERY", _b(eval_rag_defaults.get("USE_MULTI_QUERY"))),
            "NO_SEARCH_CONFIDENCE_THRESHOLD": ("NO_SEARCH_CONFIDENCE_THRESHOLD", str(eval_rag_defaults.get("NO_SEARCH_CONFIDENCE_THRESHOLD"))),
            "CANDIDATE_POOL_SIZE": ("CANDIDATE_POOL_SIZE", str(eval_rag_defaults.get("CANDIDATE_POOL_SIZE"))),
            "ENABLE_INTENT_NO_SEARCH": ("ENABLE_INTENT_NO_SEARCH", _b(eval_rag_defaults.get("ENABLE_INTENT_NO_SEARCH"))),
            "NO_SEARCH_STRICT_THRESHOLD": ("NO_SEARCH_STRICT_THRESHOLD", str(eval_rag_defaults.get("NO_SEARCH_STRICT_THRESHOLD"))),
            "NO_SEARCH_HEURISTIC_THRESHOLD": ("NO_SEARCH_HEURISTIC_THRESHOLD", str(eval_rag_defaults.get("NO_SEARCH_HEURISTIC_THRESHOLD"))),
            "NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF": ("NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF", str(eval_rag_defaults.get("NO_SEARCH_OVERRIDE_SCIENCE_MAX_CONF"))),
        }
        for _, (k, v) in mapping.items():
            if v is not None and v != "None":
                env_pairs.append((k, v))

    # If user already sets some optional dense controls, include them too
    for name in ["DENSE_EMBEDDING_FIELDS", "DENSE_K_PER_FIELD", "DENSE_K_PER_FIELD_MAP", "MULTI_QUERY_COUNT"]:
        if os.getenv(name) is not None:
            env_pairs.append((name, os.getenv(name)))

    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for k, v in env_pairs:
        if k in seen:
            continue
        seen.add(k)
        dedup.append((k, v))

    def shell_quote(s: str) -> str:
        # Minimal safe quoting for markdown display
        s = "" if s is None else str(s)
        return "'" + s.replace("'", "'\\''") + "'"

    env_str = " ".join([f"{k}={shell_quote(_mask_value(str(v)))}" for k, v in dedup])
    lines.append("\nExample: sanity run (1 item)\n")
    lines.append("```bash")
    lines.append(f"{env_str} EVAL_LIMIT=1 SUBMISSION_FILE='submission_sanity.jsonl' python3 main.py")
    lines.append("```")

    lines.append("\nExample: full submission\n")
    lines.append("```bash")
    lines.append(f"{env_str} SUBMISSION_FILE='submission.csv' python3 main.py")
    lines.append("```")

    lines.append("\n## Health Checks\n")
    lines.append(f"- RUN_HEALTH_CHECKS: `{str(run_health_checks).lower()}`")
    if not run_health_checks:
        lines.append("- Not executed. Re-run with `RUN_HEALTH_CHECKS=1` to perform 1 API call per provider/model.")
    else:
        for r in health:
            if r.get("ok"):
                if "dim" in r:
                    lines.append(f"- {r['name']}: ✅ dim={r['dim']} l2_norm={r['l2_norm']:.4f} all_zero={r['all_zero']}")
                else:
                    lines.append(f"- {r['name']}: ✅ response_chars={r.get('response_chars', 0)}")
            else:
                lines.append(f"- {r['name']}: ❌ {r.get('error')}")

    lines.append("\n## Notes\n")
    lines.append("- 이 리포트는 설정/모델 ID 확인용이며, API 호출 결과(성능/점수)를 포함하지 않습니다.")
    lines.append("- 대회 규칙상 topk=[] 정답 케이스(21개)가 존재하므로, 게이팅 변경은 ‘추가로 비울 케이스’가 과학 질문을 침범하지 않게 보수적으로 진행해야 합니다.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
