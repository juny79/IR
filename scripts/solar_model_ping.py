import os
import sys
from pathlib import Path

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    model = os.getenv("SOLAR_MODEL_ID") or "<unset>"
    print(f"[Ping] SOLAR_MODEL_ID env: {model}")

    # One minimal call to verify model identifier is accepted by Upstage Solar API.
    try:
        from models.solar_client import solar_client
        # HyDE call: one request
        txt = solar_client.generate_hypothetical_answer("광합성이란?")
        if txt:
            print(f"[Ping] Solar call OK. response_chars={len(txt)}")
        else:
            print("[Ping] Solar call returned empty response (no exception).")
    except Exception as e:
        print(f"[Ping] Solar call FAILED: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
