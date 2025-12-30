import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import models.*` works even when
# executed as `python scripts/print_model_config.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    # Avoid printing any secrets; only show model identifiers.
    gemini_env = os.getenv("GEMINI_MODEL_ID")
    gemini_selected = gemini_env or "models/gemini-3-flash-preview"
    print(f"[Model] Gemini model id: {gemini_selected} (GEMINI_MODEL_ID={'<unset>' if not gemini_env else gemini_env})")

    try:
        from models.solar_client import solar_client
        print(f"[Model] Solar model id: {getattr(solar_client, 'model', '<unknown>')}")
        solar_env = os.getenv("SOLAR_MODEL_ID")
        print(f"[Model] SOLAR_MODEL_ID env: {'<unset>' if not solar_env else solar_env}")
    except Exception as e:
        print(f"[Model] Solar model id: <unavailable> ({str(e)[:120]})")


if __name__ == "__main__":
    main()
