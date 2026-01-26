#!/usr/bin/env python3
"""
Deterministic (no-LLM) check for invalid public deal proposals:
  - Requires exactly one integer inside <VALUE>...</VALUE>.
  - Integer must be within [-10, 10].

This script does NOT run any LLM.
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
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for invalid public <VALUE> proposals (no LLM).")
    parser.add_argument("--history", type=Path, nargs="*", default=[], help="history*.json files to scan")
    parser.add_argument(
        "--outputs-by-category",
        type=Path,
        default=None,
        help="outputs_by_category.txt to expand into history*.json files",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rounds evaluated")
    parser.add_argument("--flush-every", type=int, default=500, help="Write partial results every N rounds")
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

            public_answer = extract_public_answer(entry)
            if not public_answer:
                continue

            ints, has_non_int = parse_value_tags(public_answer)
            public_value: Optional[int] = ints[0] if len(ints) == 1 else None
            invalid = False
            reason = ""
            if len(ints) == 0:
                invalid = True
                reason = "no integer proposal" if not has_non_int else "non-integer proposal"
            elif len(ints) > 1:
                invalid = True
                reason = "multiple <VALUE> tags"
            elif public_value is not None and (public_value < -10 or public_value > 10):
                invalid = True
                reason = "value out of range [-10,10]"

            out_entry: Dict[str, Any] = {
                "path": str(hist_path),
                "round": idx,
                "agent": agent,
                "public_answer": public_answer,
                "public_value": public_value,
                "invalid_public_value": bool(invalid),
                "invalid_reason": reason,
            }
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

    invalid_count = sum(1 for r in results if r.get("invalid_public_value"))
    total = len(results)
    print(f"Wrote {total} records to {args.out}")
    print(f"invalid_public_value: {invalid_count} ({(invalid_count/total*100) if total else 0:.2f}%)")


if __name__ == "__main__":
    main()

