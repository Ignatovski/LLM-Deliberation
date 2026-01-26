#!/usr/bin/env python3
"""
Deterministic (no-LLM) check for bracket/tag leakage:
  - Detects when private tags (<SCRATCHPAD>/<PLAN>) appear in *public_answer*.
  - Flags whether this is likely due to an unclosed <ANSWER> block (missing </ANSWER>).

This script does NOT run any LLM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from public_output_utils import (
    flush_results,
    load_existing_results,
    load_history,
    make_key,
    parse_outputs_by_category,
    public_has_private_tags,
    unclosed_answer_tag,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for private-tag leakage in public answers (no LLM).")
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
            raw_public = entry.get("public_answer") or ""
            if not isinstance(raw_public, str) or not raw_public.strip():
                continue
            agent = entry.get("agent") or "unknown"
            key = make_key(str(hist_path), idx, str(agent))
            if key in seen:
                continue

            has_private = public_has_private_tags(raw_public)
            unclosed = unclosed_answer_tag(entry.get("full_answer") or "")
            out_entry: Dict[str, Any] = {
                "path": str(hist_path),
                "round": idx,
                "agent": agent,
                "public_answer": raw_public.strip(),
                "public_has_private_tags": bool(has_private),
                "leakage_due_to_unclosed_answer": bool(has_private and unclosed),
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

    private_tag_count = sum(1 for r in results if r.get("public_has_private_tags"))
    unclosed_count = sum(1 for r in results if r.get("leakage_due_to_unclosed_answer"))
    total = len(results)
    print(f"Wrote {total} records to {args.out}")
    print(f"public_has_private_tags: {private_tag_count} ({(private_tag_count/total*100) if total else 0:.2f}%)")
    print(f"leakage_due_to_unclosed_answer: {unclosed_count} ({(unclosed_count/total*100) if total else 0:.2f}%)")


if __name__ == "__main__":
    main()

