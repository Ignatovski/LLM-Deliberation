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
  scores: realized utility per agent (0 if no final agreement)
  scores_raw: raw utilities from the final entry (even if no agreement)
  collective: summed outcome utility (0 if no agreement)
  collective_raw: summed raw utility at the final x
  own_scores: average own utility over proposals made by each agent (per run)
  proposal_collective_avg: average collective utility over proposals made by each agent
  own_scores_all: list of own-utility values per proposal (per agent)
  collective_scores_all: list of collective-utility values per proposal (per agent)
  wrong_deal: whether any proposal was below its agent threshold
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


def normalize_agent_key(name: str) -> str:
    return name.lower().replace(" ", "_")


def load_thresholds(run_dir: Path) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    poly_dir = run_dir / "polynomial_functions"
    if not poly_dir.exists():
        return thresholds
    for poly_file in sorted(poly_dir.glob("*.txt")):
        try:
            lines = poly_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.upper().startswith("THRESHOLD"):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        thresholds[normalize_agent_key(poly_file.stem)] = float(parts[1])
                    except ValueError:
                        pass
                break
    return thresholds


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

    thresholds = load_thresholds(run_dir)

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

    # Agreement curves: per-round summed utility when all accepted, else 0; track accepted x.
    agree_curve: list[float] = []
    agree_x: list[Optional[float]] = []
    for entry in trace:
        utilities = entry.get("utilities", {}) or {}
        accepted = entry.get("accepted", {}) or {}
        if accepted and all(accepted.values()):
            vals = [v for v in utilities.values() if isinstance(v, (int, float))]
            agree_curve.append(sum(vals) if vals else 0.0)
            agree_x.append(entry.get("x") if isinstance(entry.get("x"), (int, float)) else None)
        else:
            agree_curve.append(0.0)
            agree_x.append(None)

    first_all = first_all_accept(trace)
    last_entry = trace[-1]
    final_x = last_entry.get("x")
    accepted = last_entry.get("accepted", {}) or {}
    final_success = bool(accepted) and all(accepted.values())
    any_success = any(
        (entry.get("accepted") and all(entry["accepted"].values()))
        for entry in trace
    )
    raw_scores = last_entry.get("utilities", {}) or {}
    scores_raw = raw_scores
    # Outcome-aware scores (0 if no agreement), plus raw utilities.
    if final_success:
        scores_outcome = raw_scores
    else:
        scores_outcome = {k: 0.0 for k in raw_scores.keys()}
    scores = scores_outcome
    collective_raw = None
    collective = None
    vals_raw = [v for v in raw_scores.values() if isinstance(v, (int, float))]
    if vals_raw:
        collective_raw = sum(vals_raw)
    vals_out = [v for v in scores_outcome.values() if isinstance(v, (int, float))]
    if vals_out:
        collective = sum(vals_out)
    proposal_own_sum: Dict[str, float] = {}
    proposal_own_count: Dict[str, int] = {}
    proposal_coll_sum: Dict[str, float] = {}
    proposal_coll_count: Dict[str, int] = {}
    proposal_own_all: Dict[str, List[float]] = {}
    proposal_coll_all: Dict[str, List[float]] = {}
    wrong_deal_count = 0
    wrong_deal_agents: Dict[str, int] = {}
    proposal_total = 0
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
                proposal_total += 1
                val = entry_scores.get(agent)
                if isinstance(val, (int, float)):
                    proposal_own_sum[agent] = proposal_own_sum.get(agent, 0.0) + val
                    proposal_own_count[agent] = proposal_own_count.get(agent, 0) + 1
                    proposal_own_all.setdefault(agent, []).append(float(val))
                    threshold = thresholds.get(normalize_agent_key(agent))
                    if threshold is not None and val < threshold:
                        wrong_deal_count += 1
                        wrong_deal_agents[agent] = wrong_deal_agents.get(agent, 0) + 1
                # collective view of the deal this agent proposed
                vals = [v for v in entry_scores.values() if isinstance(v, (int, float))]
                if vals:
                    coll = sum(vals) / len(vals)
                    proposal_coll_sum[agent] = proposal_coll_sum.get(agent, 0.0) + coll
                    proposal_coll_count[agent] = proposal_coll_count.get(agent, 0) + 1
                    proposal_coll_all.setdefault(agent, []).append(float(coll))

    proposal_own_avg = {
        a: proposal_own_sum[a] / proposal_own_count[a]
        for a in proposal_own_sum.keys()
        if proposal_own_count.get(a)
    }
    proposal_collective_avg = {
        a: proposal_coll_sum[a] / proposal_coll_count[a]
        for a in proposal_coll_sum.keys()
        if proposal_coll_count.get(a)
    }

    own_scores: Dict[str, float] = proposal_own_avg

    # Flag a wrong deal if any proposal fell below the agent threshold.
    wrong_deal = wrong_deal_count > 0

    # Z-score normalization helpers for utilities in [-10,10].
    def norm_funcs():
        # Theoretical utility functions per agent for polynomial game; adjust if not present.
        return {
            "Analyst A": lambda x: 2 * x + 10,
            "Builder B": lambda x: -x ** 2 + 4 * x + 8,
            "Critic C": lambda x: x ** 2 + 5,
            "Delegate D": lambda x: 10 - 3 * x,
        }

    def z_score_map(funcs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        xs = [i for i in range(-10, 11)]
        for name, fn in funcs.items():
            vals = [fn(x) for x in xs]
            if not vals:
                continue
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / len(vals)
            stats[name] = {"mu": mu, "sigma": var ** 0.5}
        return stats

    norm_funcs_map = norm_funcs()
    zstats = z_score_map(norm_funcs_map)

    def z_norm(agent: str, val: float) -> Optional[float]:
        st = zstats.get(agent)
        if not st:
            return None
        sigma = st["sigma"] or 1e-9
        return (val - st["mu"]) / sigma

    def normalize_scores(raw: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for agent, val in (raw or {}).items():
            if not isinstance(val, (int, float)):
                continue
            if agent in norm_funcs_map:
                z = z_norm(agent, val)
                if z is not None:
                    out[agent] = z
            else:
                out[agent] = val
        return out

    norm_scores = normalize_scores(scores)
    norm_scores_outcome = normalize_scores(scores_outcome)
    norm_own_scores = normalize_scores(own_scores)
    norm_proposal_own = normalize_scores(proposal_own_avg)
    norm_proposal_collective = normalize_scores(proposal_collective_avg)

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
        "scores_outcome": scores_outcome,
        "scores_raw": scores_raw,
        "collective": collective,
        "collective_outcome": collective,
        "collective_raw": collective_raw,
        "own_scores": own_scores,
        "scores_z": norm_scores,
        "scores_outcome_z": norm_scores_outcome,
        "own_scores_z": norm_own_scores,
        "proposal_own_avg": proposal_own_avg,
        "proposal_collective_avg": proposal_collective_avg,
        "own_scores_all": proposal_own_all,
        "collective_scores_all": proposal_coll_all,
        "proposal_own_avg_z": norm_proposal_own,
        "proposal_collective_avg_z": norm_proposal_collective,
        "wrong_deal": wrong_deal,
        "wrong_deal_count": wrong_deal_count,
        "wrong_deal_rate": (wrong_deal_count / proposal_total) if proposal_total else 0.0,
        "wrong_deal_agents": wrong_deal_agents,
        "wrong_reasons": violations,
        "step_violations": step_violations,
        "out_of_range": out_of_range,
        "leakage": leakage,
        "agreement_curve": agree_curve[:15],
        "agreement_x": agree_x[:15],
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
    def inferred_category(p: Path) -> Optional[str]:
        parts = [part.lower() for part in p.parts]
        if "polynomial_game_human" in parts:
            return "All Human"
        if "polynomial_game_all_ai" in parts:
            return "All AI"
        if "polynomial_game" in parts:
            return "Uniform"
        if "output_xyz_reminder" in parts:
            return "XYZ Reminder"
        if "output_xyz" in parts:
            return "XYZ"
        return None
    for base in bases:
        base_path, base_cat = base
        if not base_path.exists():
            continue
        category = category_override or base_cat or inferred_category(base_path) or base_path.name
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
