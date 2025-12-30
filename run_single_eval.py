import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run main.py to generate a submission JSONL file")
    parser.add_argument("--mode", default="submit", help="Kept for compatibility; ignored")
    parser.add_argument("--output", required=False, help="Output submission JSONL path")
    parser.add_argument("--eval", dest="eval_file", required=False, help="Eval JSONL path")
    parser.add_argument("--limit", required=False, type=int, help="Eval limit (0 = all)")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.output:
        env["SUBMISSION_FILE"] = args.output
    if args.eval_file:
        env["EVAL_FILE"] = args.eval_file
    if args.limit is not None:
        env["EVAL_LIMIT"] = str(args.limit)

    result = subprocess.run([sys.executable, "main.py"], env=env, capture_output=False, text=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
