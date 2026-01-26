#!/usr/bin/env python3
"""
Extract public answers from runs and flag potential leakage snippets for review.

Usage:
  python scripts/extract_leakage_candidates.py --summary viewer/metrics_summary.json --out summarys/leakage_candidates.json

It reads metrics_summary.json to locate history files, scans public answers, applies a
keyword-based heuristic (same as build_metrics_summary.py), and writes a JSON list of
candidate snippets with run metadata. This is meant for manual/LLM review; it does not
call any API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


KEYWORDS = [
    "my utility",
    "utility function",
    "my threshold",
    "acceptance threshold",
    "tau",
    "baseline",
    "fallback",
    "my score",
    "coefficient",
    "a_i",
    "b_i",
    "c_i",
]


def is_leaky(text: str) -> bool:
    txt = text.lower()
    return any(k in txt for k in KEYWORDS)


def load_summary(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("runs", [])


def extract_from_history(hist_path: Path, run_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        data = json.loads(hist_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rounds = data.get("rounds") or []
    candidates: List[Dict[str, Any]] = []
    for idx, entry in enumerate(rounds):
        pa = entry.get("public_answer")
        agent = entry.get("agent")
        if not isinstance(pa, str) or not agent:
            continue
        hit = is_leaky(pa)
        if hit:
            candidates.append(
                {
                    "path": str(hist_path),
                    "run_id": run_meta.get("group") or run_meta.get("path"),
                    "category": run_meta.get("category"),
                    "variant": run_meta.get("variant"),
                    "model_mix": run_meta.get("model_mix"),
                    "round": idx,
                    "agent": agent,
                    "public_answer": pa,
                    "keyword_hit": True,
                }
            )
    return candidates


def resolve_history_path(path_str: str) -> Path | None:
    p = Path(path_str)
    if p.exists():
        return p
    alt = Path(".") / p
    if alt.exists():
        return alt
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract leakage candidates from public answers.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("viewer/metrics_summary.json"),
        help="Path to metrics_summary.json (default: viewer/metrics_summary.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("summarys/leakage_candidates.json"),
        help="Output JSON file for candidates (default: summarys/leakage_candidates.json).",
    )
    args = parser.parse_args()

    runs = load_summary(args.summary)
    all_candidates: List[Dict[str, Any]] = []
    for run in runs:
        hist_path = resolve_history_path(run.get("path", ""))
        if not hist_path:
            continue
        all_candidates.extend(extract_from_history(hist_path, run))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_candidates, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_candidates)} candidates to {args.out}")


if __name__ == "__main__":
    main()
