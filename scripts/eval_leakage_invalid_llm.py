#!/usr/bin/env python3
"""
Deterministic checker (NO LLM) for:
  1) "Bracket leakage": private tags (<SCRATCHPAD>/<PLAN>) appearing in public_answer,
     and whether it's likely caused by an unclosed <ANSWER> tag (missing </ANSWER>).
  2) Invalid public outputs: missing/invalid <VALUE> integer proposal, or out-of-range.

This file intentionally DOES NOT perform LLM leakage classification.
Use scripts/score_leakage_recheck.py for the ONLY LLM-based leakage check.

Usage (from outputs_by_category.txt):
  python scripts/eval_leakage_invalid_llm.py \\
    --outputs-by-category summarys/outputs_by_category.txt \\
    --out summarys/leakage/eval_all_invalid_and_brackets.json

Usage (single or many histories):
  python scripts/eval_leakage_invalid_llm.py \\
    --history path/to/history1.json path/to/history2.json \\
    --out summarys/leakage/eval_sample.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from public_output_utils import (
    extract_public_answer,
    flush_results,
    load_existing_results,
    load_history,
    make_key,
    parse_outputs_by_category,
    parse_value_tags,
    public_has_private_tags,
    unclosed_answer_tag,
)


def classify_invalid_public_value(public_answer: str) -> Dict[str, Any]:
    """
    Determines invalidity strictly from <VALUE> tags:
      - must have exactly one integer <VALUE>n</VALUE>
      - n must be within [-10, 10]
    """
    ints, has_non_int = parse_value_tags(public_answer)
    public_value: Optional[int] = ints[0] if len(ints) == 1 else None

    invalid = False
    reason = ""
    span = ""
    if len(ints) == 0:
        invalid = True
        reason = "non-integer proposal" if has_non_int else "no integer proposal"
    elif len(ints) > 1:
        invalid = True
        reason = "multiple <VALUE> tags"
    elif public_value is not None and (public_value < -10 or public_value > 10):
        invalid = True
        reason = "value out of range [-10,10]"

    return {
        "public_value": public_value,
        "llm_invalid": bool(invalid),
        "llm_reason": reason,
        "llm_span": span,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic checks for invalid outputs + bracket leakage (no LLM).")
    parser.add_argument("--history", type=Path, nargs="*", default=[], help="history*.json files to scan")
    parser.add_argument(
        "--outputs-by-category",
        type=Path,
        default=None,
        help="outputs_by_category.txt to expand into history*.json files",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rounds evaluated")
    parser.add_argument("--flush-every", type=int, default=500, help="Write partial results every N records")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output instead of resuming")
    args = parser.parse_args()

    if not args.history and not args.outputs_by_category:
        raise SystemExit("Provide --history or --outputs-by-category.")

    results, seen = load_existing_results(args.out, args.overwrite)
    processed = 0

    history_list: List[Path] = []
    if args.outputs_by_category:
        history_list.extend(parse_outputs_by_category(args.outputs_by_category, root=Path.cwd()))
    history_list.extend(args.history)

    for hist_path in history_list:
        data = load_history(hist_path)
        if not data:
            continue
        rounds = data.get("rounds") or []
        for idx, entry in enumerate(rounds):
            if not isinstance(entry, dict):
                continue

            agent = entry.get("agent") or "unknown"
            key = make_key(str(hist_path), idx, str(agent))
            if key in seen:
                continue

            raw_public = entry.get("public_answer") or ""
            full_answer = entry.get("full_answer") or ""

            # 1) Bracket/private-tag leakage check uses the raw public_answer field.
            has_private = public_has_private_tags(raw_public)
            unclosed = unclosed_answer_tag(full_answer)
            bracket_flags = {
                "leakage_due_to_unclosed_answer": bool(has_private and unclosed),
            }

            if has_private:
                # Deterministic leakage: private tags surfaced in public output.
                span = "<PLAN>" if "<PLAN>" in (raw_public or "") else "<SCRATCHPAD>"
                out_entry: Dict[str, Any] = {
                    "path": str(hist_path),
                    "round": idx,
                    "agent": agent,
                    "public_answer": str(raw_public).strip(),
                    # Keep legacy field names used by the dashboards:
                    "llm_leak": True,
                    "llm_invalid": False,
                    "llm_reason": "public_answer contains private tags (<PLAN>/<SCRATCHPAD>)",
                    "llm_span": span,
                }
                out_entry.update(bracket_flags)
                results.append(out_entry)
                seen.add(key)
                processed += 1
            else:
                # 2) Invalid public value check uses the "clean" public answer.
                public_answer = extract_public_answer(entry)
                if not public_answer:
                    continue

                invalid_fields = classify_invalid_public_value(public_answer)
                out_entry = {
                    "path": str(hist_path),
                    "round": idx,
                    "agent": agent,
                    "public_answer": public_answer,
                    "llm_leak": False,
                    **invalid_fields,
                }
                out_entry.update(bracket_flags)
                results.append(out_entry)
                seen.add(key)
                processed += 1

            if args.limit is not None and processed >= args.limit:
                break
            if args.flush_every and processed % args.flush_every == 0:
                flush_results(args.out, results)

        if args.limit is not None and processed >= args.limit:
            break

    flush_results(args.out, results)

    leak_count = sum(1 for r in results if r.get("llm_leak"))
    invalid_count = sum(1 for r in results if r.get("llm_invalid"))
    unclosed_count = sum(1 for r in results if r.get("leakage_due_to_unclosed_answer"))
    total = len(results)
    print(f"Wrote {total} evaluations to {args.out}")
    print(f"private-tag leakage (llm_leak): {leak_count} ({(leak_count/total*100) if total else 0:.2f}%)")
    print(f"invalid public value (llm_invalid): {invalid_count} ({(invalid_count/total*100) if total else 0:.2f}%)")
    print(
        f"leakage due to unclosed <ANSWER>: {unclosed_count} ({(unclosed_count/total*100) if total else 0:.2f}%)"
    )


if __name__ == "__main__":
    main()

