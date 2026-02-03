#!/usr/bin/env python3
"""
Summarize "final x repeat" behavior for unsuccessful games (no agreement at all).

Definition (per run):
  - final_x: x in the last trace entry
  - prior_count: how many times that same x appeared earlier in the run (trace[:-1])
  - is_repeat: prior_count > 0

By default this script reads viewer/metrics_summary.json and then loads each run's
history file via the stored "path" field, so it works even if metrics_summary.json
doesn't already include the derived fields.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


def is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x and x not in (float("inf"), float("-inf"))


def canon_x(x: Any) -> Optional[int | float]:
    if not is_finite(x):
        return None
    xf = float(x)
    if abs(xf - round(xf)) < 1e-6:
        return int(round(xf))
    return xf


def compute_final_repeat_from_history(history_path: Path) -> tuple[Optional[int | float], Optional[int]]:
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except OSError:
        return None, None
    except json.JSONDecodeError:
        return None, None
    trace = data.get("polynomial_trace") or []
    if not isinstance(trace, list) or len(trace) < 2:
        return None, None
    final = canon_x((trace[-1] or {}).get("x"))
    if final is None:
        return None, None
    prior = 0
    for entry in trace[:-1]:
        if canon_x((entry or {}).get("x")) == final:
            prior += 1
    return final, prior


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "0.0%"
    return f"{(100.0 * n / d):.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        type=Path,
        default=Path("viewer/metrics_summary.json"),
        help="Path to metrics_summary.json (default: viewer/metrics_summary.json).",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repo root to resolve run history paths stored in metrics (default: .).",
    )
    ap.add_argument(
        "--by",
        choices=["none", "mix", "category", "mix+category"],
        default="mix+category",
        help="Break down stats by group (default: mix+category).",
    )
    args = ap.parse_args()

    obj = json.loads(args.metrics.read_text(encoding="utf-8"))
    runs = obj.get("runs") or []
    if not isinstance(runs, list):
        raise SystemExit("metrics file does not contain a 'runs' list")

    groups: dict[str, list[tuple[Optional[int | float], Optional[int]]]] = defaultdict(list)
    missing = 0
    total_unsuccess = 0

    for r in runs:
        if not isinstance(r, dict):
            continue
        # "no agreement at all" => any_success == False
        if r.get("any_success") is True:
            continue
        total_unsuccess += 1

        rel = r.get("path")
        if not isinstance(rel, str):
            missing += 1
            continue
        history_path = (args.root / rel).resolve()
        final, prior = compute_final_repeat_from_history(history_path)
        if final is None or prior is None:
            missing += 1
            continue

        mix = r.get("model_mix") or "unknown_mix"
        cat = r.get("category") or "unknown_category"
        if args.by == "none":
            key = "all"
        elif args.by == "mix":
            key = str(mix)
        elif args.by == "category":
            key = str(cat)
        else:
            key = f"{mix} | {cat}"
        groups[key].append((final, prior))

    print(f"Unsuccessful runs (any_success=false): {total_unsuccess}")
    if missing:
        print(f"Missing/unreadable histories: {missing}")
    print("")

    for key in sorted(groups.keys()):
        rows = groups[key]
        n = len(rows)
        repeats = sum(1 for _, prior in rows if prior and prior > 0)
        prior_sum = sum(prior for _, prior in rows if isinstance(prior, int))
        avg_prior = (prior_sum / n) if n else 0.0
        hist = Counter(prior for _, prior in rows if isinstance(prior, int))

        print(f"[{key}] n={n} repeats={repeats} ({pct(repeats, n)}) avg_prior_count={avg_prior:.2f}")
        # Compact histogram: show counts for 0..5 and 6+
        buckets = {k: hist.get(k, 0) for k in range(0, 6)}
        buckets["6+"] = sum(v for k, v in hist.items() if k >= 6)
        print("  prior_count_hist:", ", ".join(f"{k}:{v}" for k, v in buckets.items()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

