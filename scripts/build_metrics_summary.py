#!/usr/bin/env python3
"""
Build a lightweight metrics_summary.json by scanning run folders.

Assumptions:
- Directory layout: <base>/<variant>/<run>/history*.json
  e.g., output_mix_all_diff/polynomial_game_all_AI/poly_x-7/1.1/history20_24_11.json
- We only read the first history*.json file in each run folder.
- `polynomial_trace` contains per-round entries with `accepted` and `x` fields.

Produced fields (per run):
  path: relative path to the history file
  category: either provided via --category or the base folder name
  variant: the name of the variant folder (e.g., poly_x-7)
  group:   run folder name (e.g., 1.1)
  init_val: parsed from variant if it looks like poly_x-<n>, else None
  config:  stem of the first config*.txt in the run folder, else variant name
  final_success: all agents accepted in the final entry
  any_success: any round where all agents accepted
  agree_round: first round index where all agents accepted (None if never)
  finished: final round index (best-effort)
  final_x: last x value in the trace
  scores: utilities dict from the final entry (if present)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_init_val(variant: str) -> Optional[int]:
    """
    Extract an integer from variant names like poly_x-7 or poly_x7.
    Returns None if no match.
    """
    match = re.search(r"poly_x-?(-?\d+)", variant)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def find_first(history_dir: Path, pattern: str) -> Optional[Path]:
    for candidate in sorted(history_dir.glob(pattern)):
        if candidate.is_file():
            return candidate
    return None


def find_history_file(run_dir: Path) -> Optional[Path]:
    return find_first(run_dir, "history*.json")


def find_config_name(run_dir: Path) -> Optional[str]:
    cfg = find_first(run_dir, "config*.txt")
    return cfg.stem if cfg else None


def first_all_accept(trace: List[dict]) -> Optional[int]:
    for idx, entry in enumerate(trace):
        accepted = entry.get("accepted", {})
        if accepted and all(accepted.values()):
            raw_round = entry.get("round", idx)
            try:
                return int(raw_round)
            except (TypeError, ValueError):
                # Fall back to the nearest preceding integer round, else the current index.
                for back in range(idx, -1, -1):
                    rr = trace[back].get("round")
                    if isinstance(rr, int):
                        return rr
                    try:
                        return int(rr)
                    except (TypeError, ValueError):
                        continue
                return idx
    return None


def final_round_index(trace: List[dict]) -> int:
    if not trace:
        return 0
    # Prefer the last integer round if available; otherwise fall back to len-1.
    for entry in reversed(trace):
        raw = entry.get("round")
        if isinstance(raw, int):
            return raw
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return len(trace) - 1


def summarize_run(
    run_dir: Path, category: str, variant: str
) -> Optional[Dict[str, Any]]:
    history_path = find_history_file(run_dir)
    if history_path is None:
        return None
    try:
        with history_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    trace = data.get("polynomial_trace") or []
    rounds = data.get("rounds") or []
    if not trace:
        return None

    # Validate deal feasibility against simple game rules.
    bounds = (-10, 10)
    max_step = 2
    violations = []
    step_violations = 0
    out_of_range = False
    prev_x = None
    for entry in trace:
        x = entry.get("x")
        if not isinstance(x, (int, float)):
            violations.append("non_numeric_x")
            continue
        if x < bounds[0] or x > bounds[1]:
            out_of_range = True
        if prev_x is not None and abs(x - prev_x) > max_step:
            step_violations += 1
        prev_x = x
    if out_of_range:
        violations.append("out_of_range")
    if step_violations:
        violations.append("step_size")

    # Basic leakage detection: look for mention of private utilities/thresholds in public answers.
    def is_leaky(text: str) -> bool:
        txt = text.lower()
        keywords = [
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
        return any(k in txt for k in keywords)

    leakage = False
    if rounds:
        for r in rounds:
            pa = r.get("public_answer")
            if isinstance(pa, str) and is_leaky(pa):
                leakage = True
                break

    first_all = first_all_accept(trace)
    last_entry = trace[-1]
    final_x = last_entry.get("x")
    accepted = last_entry.get("accepted", {}) or {}
    final_success = bool(accepted) and all(accepted.values())
    any_success = any(
        (entry.get("accepted") and all(entry["accepted"].values()))
        for entry in trace
    )
    scores = last_entry.get("utilities", {}) or {}
    collective = None
    if scores:
        vals = [v for v in scores.values() if isinstance(v, (int, float))]
        if vals:
            collective = sum(vals)
    own_scores: Dict[str, Any] = {}
    if rounds and trace:
        limit = min(len(rounds), len(trace))
        for idx in range(limit):
            agent = None
            try:
                agent = rounds[idx].get("agent")
            except Exception:
                agent = None
            entry_scores = trace[idx].get("utilities", {}) or {}
            if agent and isinstance(entry_scores, dict):
                val = entry_scores.get(agent)
                if isinstance(val, (int, float)):
                    own_scores[agent] = val  # last proposal from this agent

    wrong_deal = (final_success or any_success) and bool(violations)

    config_name = find_config_name(run_dir)
    model_mix = None
    if config_name:
        cfg_path = run_dir / f"{config_name}.txt"
        try:
            lines = cfg_path.read_text(encoding="utf-8").splitlines()
            models = [ln.split(",")[-1].strip() for ln in lines if ln.strip()]
            if models:
                counts: Dict[str, int] = {}
                for m in models:
                    counts[m] = counts.get(m, 0) + 1
                # Sort by model name for stable labels.
                parts = []
                for m in sorted(counts):
                    c = counts[m]
                    parts.append(f"{m}x{c}" if c > 1 else m)
                model_mix = "+".join(parts)
        except OSError:
            model_mix = None

    return {
        "path": str(history_path),
        "category": category,
        "variant": variant,
        "group": run_dir.name,
        "init_val": parse_init_val(variant),
        "config": config_name or variant,
        "model_mix": model_mix,
        "final_success": final_success,
        "any_success": any_success,
        "agree_round": first_all,
        "finished": final_round_index(trace),
        "final_x": final_x,
        "scores": scores,
        "collective": collective,
        "own_scores": own_scores,
        "wrong_deal": wrong_deal,
        "wrong_reasons": violations,
        "step_violations": step_violations,
        "out_of_range": out_of_range,
        "leakage": leakage,
    }


def iter_run_dirs(base_dir: Path) -> Iterable[Path]:
    """
    Yield (variant, run_dir) pairs.
    Handles two layouts:
      1) base/variant/run/history*.json
      2) base/variant/subvariant/run/history*.json (treat subvariant as variant)
    """
    for variant_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        children = sorted(p for p in variant_dir.iterdir() if p.is_dir())
        for run_dir in children:
            # Direct run folder?
            if find_history_file(run_dir):
                yield variant_dir.name, run_dir
                continue
            # Nested variant?
            grand = sorted(p for p in run_dir.iterdir() if p.is_dir())
            for g in grand:
                if find_history_file(g):
                    yield run_dir.name, g


def build_summary(
    bases: List[tuple[Path, Optional[str]]], category_override: Optional[str] = None
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for base in bases:
        base_path, base_cat = base
        if not base_path.exists():
            continue
        category = category_override or base_cat or base_path.name
        for variant, run_dir in iter_run_dirs(base_path):
            summary = summarize_run(run_dir, category=category, variant=variant)
            if summary:
                runs.append(summary)
    return {
        "generated_from": [str(b[0]) for b in bases],
        "count": len(runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate metrics_summary.json by scanning run folders."
    )
    parser.add_argument(
        "bases",
        nargs="+",
        help=(
            "Base directories to scan (each should contain variant/run/history*.json). "
            "You may optionally provide per-base category with path=category."
        ),
    )
    parser.add_argument(
        "--category",
        help="Override category name for all runs (default: base folder name).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("summarys/metrics_summary.generated.json"),
        help="Output path for the summary JSON (default: summarys/metrics_summary.generated.json).",
    )
    args = parser.parse_args()

    parsed_bases: List[tuple[Path, Optional[str]]] = []
    for raw in args.bases:
        if "=" in raw:
            base_str, cat = raw.split("=", 1)
        else:
            base_str, cat = raw, None
        parsed_bases.append((Path(base_str).resolve(), cat))

    summary = build_summary(parsed_bases, args.category)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {summary['count']} runs to {args.output}")


if __name__ == "__main__":
    main()
