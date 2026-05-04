"""
Batch runner for polynomial.main_polynomial with automatic exp_name sequencing and retries.

Usage example:
    python run_batch_polynomial.py \
        --exp-prefix poly_x-7 \
        --start 3 --end 15 --suffix 1 --max-retries 3 \
        -- --game_dir games_descriptions/polynomial_game \
           --output_dir polynomial/outputs/output/cooperative \
           --temp 1 \
           --azure \
           --azure_openai_api <KEY> \
           --azure_openai_endpoint https://ai-pentesting-models.openai.azure.com/ \
           --reuse_faiss \
           --result "-7" \
           --min_answers 16

Notes:
- exp_name is auto-set to "<exp-prefix>_<idx>.<suffix>" for idx in [start, end].
- Any --exp_name provided in the trailing args is stripped so this script controls it.
- If a run fails (non-zero exit), the same index is retried up to --max-retries times.
"""

import argparse
import subprocess
import sys
from typing import List


def strip_exp_name(args: List[str]) -> List[str]:
    """Remove any --exp_name occurrences from the provided args."""
    cleaned: List[str] = []
    skip_next = False
    for arg in args:
        if arg == "--":
            continue
        if skip_next:
            skip_next = False
            continue
        if arg == "--exp_name":
            skip_next = True
            continue
        if arg.startswith("--exp_name="):
            continue
        cleaned.append(arg)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Batch runner for polynomial.main_polynomial")
    parser.add_argument("--exp-prefix", default="", help="Prefix for exp_name (e.g., poly_x-7). If empty, exp_name will be <idx>.<suffix>.")
    parser.add_argument("--suffix", default="1", help="Suffix after the dot in exp_name (default: 1)")
    parser.add_argument("--start", type=int, default=1, help="Starting index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="Ending index (inclusive)")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per run on failure")
    parser.add_argument(
        "main_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to polynomial.main_polynomial (prefix with --)",
    )
    args = parser.parse_args()

    base_args = strip_exp_name(args.main_args)

    for idx in range(args.start, args.end + 1):
        if args.exp_prefix:
            exp_name = f"{args.exp_prefix}_{idx}.{args.suffix}"
        else:
            exp_name = f"{idx}.{args.suffix}"
        cmd = [sys.executable, "-m", "polynomial.main_polynomial", "--exp_name", exp_name] + base_args
        attempt = 0
        while True:
            attempt += 1
            print(f"\n=== Running {exp_name} (attempt {attempt}) ===")
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print(f"Run {exp_name} succeeded.")
                break
            if attempt >= args.max_retries:
                print(f"Run {exp_name} failed after {attempt} attempt(s); moving on.")
                break
            print(f"Run {exp_name} failed (exit {result.returncode}); retrying...")


if __name__ == "__main__":
    main()
