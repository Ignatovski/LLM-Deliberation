#!/usr/bin/env python3
"""
Build a compact per-run dynamics summary from an existing metrics_summary.json.

Why:
  - BaselineOverview focuses on final/any success, but we also want to analyze
    "behavior over time": are proposals moving toward the best group outcome,
    and what happens after the first unanimous agreement.

Inputs:
  - A metrics summary JSON produced by scripts/build_metrics_summary.py (or similar),
    containing `runs[]` with a `path` to each history*.json.

Outputs:
  - A JSON with per-run traces/derived metrics that can be consumed by a viewer HTML page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polynomial.core.polynomial_utils import evaluate_all_agents, load_polynomial  # noqa: E402


def _repo_root() -> Path:
    return ROOT


def resolve_history_path(raw: str) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    # Many summaries store absolute paths under the repo root.
    root = _repo_root()
    try:
        if p.is_absolute() and str(p).endswith(".json") and root.name in p.parts:
            idx = p.parts.index(root.name)
            rel = Path(*p.parts[idx + 1 :])
            candidate = root / rel
            if candidate.exists():
                return candidate
    except ValueError:
        pass
    # Try treating as repo-relative.
    candidate = root / raw.lstrip("/")
    if candidate.exists():
        return candidate
    return None


def normalize_path_for_web(path: Path) -> str:
    root = _repo_root()
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def load_profiles_for_history(history_path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    poly_dir = history_path.parent / "polynomial_functions"
    if not poly_dir.exists():
        return None
    # load_polynomial expects game_dir + poly_dir_name
    return load_polynomial(str(history_path.parent), "polynomial_functions")


def best_x_values(
    profiles: Dict[str, Dict[str, Any]],
) -> Tuple[int, Optional[int]]:
    # Determine domain from profiles (fallback [-10, 10]).
    lows: List[int] = []
    highs: List[int] = []
    for prof in profiles.values():
        dom = prof.get("domain")
        if isinstance(dom, (list, tuple)) and len(dom) == 2:
            lows.append(int(dom[0]))
            highs.append(int(dom[1]))
    low = min(lows) if lows else -10
    high = max(highs) if highs else 10

    best_any_x = low
    best_any_score = float("-inf")
    best_feasible_x: Optional[int] = None
    best_feasible_score = float("-inf")

    for x in range(low, high + 1):
        utilities, accepted = evaluate_all_agents(profiles, x)
        total = float(sum(utilities.values()))
        if total > best_any_score:
            best_any_score = total
            best_any_x = x
        if all(bool(v) for v in accepted.values()):
            if total > best_feasible_score:
                best_feasible_score = total
                best_feasible_x = x
    return best_any_x, best_feasible_x


def safe_sum(d: Dict[str, Any]) -> float:
    total = 0.0
    for v in d.values():
        try:
            total += float(v)
        except Exception:
            continue
    return total


def compute_run_dynamics(
    run_entry: Dict[str, Any],
    best_feasible_override: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    history_path = resolve_history_path(run_entry.get("path", ""))
    if history_path is None:
        return None

    history = json.loads(history_path.read_text())
    trace = history.get("polynomial_trace") or []
    if not isinstance(trace, list) or not trace:
        return None

    best_any_x: Optional[int] = None
    best_feasible_x: Optional[int] = best_feasible_override

    if best_feasible_x is None:
        profiles = load_profiles_for_history(history_path)
        if profiles:
            best_any_x, best_feasible_x = best_x_values(profiles)
        # If there is no feasible x (rare), fall back to unconstrained best.
        if best_feasible_x is None and best_any_x is not None:
            best_feasible_x = best_any_x

    x_trace: List[int] = []
    collective_trace: List[float] = []
    agreement_trace: List[bool] = []
    accept_count_trace: List[int] = []
    distance_trace: List[Optional[int]] = []

    for step in trace:
        x = step.get("x")
        if not isinstance(x, int):
            try:
                x = int(x)
            except Exception:
                x = None
        if x is None:
            continue

        utilities = step.get("utilities") or {}
        accepted = step.get("accepted") or {}

        x_trace.append(x)
        collective_trace.append(safe_sum(utilities) if isinstance(utilities, dict) else 0.0)

        if isinstance(accepted, dict) and accepted:
            accept_count = sum(1 for v in accepted.values() if bool(v))
            accept_count_trace.append(accept_count)
            agreement_trace.append(accept_count == len(accepted))
        else:
            accept_count_trace.append(0)
            agreement_trace.append(False)

        if best_feasible_x is None:
            distance_trace.append(None)
        else:
            distance_trace.append(abs(x - int(best_feasible_x)))

    if not x_trace:
        return None

    # Derived per-run metrics
    first_agree_round: Optional[int] = None
    for i, ok in enumerate(agreement_trace):
        if ok:
            first_agree_round = i
            break

    agreements_count = int(sum(1 for ok in agreement_trace if ok))
    final_agreed = bool(agreement_trace[-1])

    agreement_lost_after_first = False
    if first_agree_round is not None:
        agreement_lost_after_first = any(not ok for ok in agreement_trace[first_agree_round + 1 :])

    # Ratios over step-to-step transitions
    deltas = len(x_trace) - 1
    collective_improve_ratio = None
    toward_opt_ratio = None
    if deltas > 0:
        collective_improve_ratio = sum(
            1 for i in range(1, len(collective_trace)) if collective_trace[i] > collective_trace[i - 1]
        ) / deltas
        if all(d is not None for d in distance_trace):
            toward_opt_ratio = sum(
                1
                for i in range(1, len(distance_trace))
                if distance_trace[i] is not None
                and distance_trace[i - 1] is not None
                and distance_trace[i] < distance_trace[i - 1]
            ) / deltas

    # Post-first-agreement drift
    drift_mean = None
    drift_max = None
    if first_agree_round is not None and first_agree_round < len(x_trace) - 1:
        anchor = x_trace[first_agree_round]
        drifts = [abs(x - anchor) for x in x_trace[first_agree_round + 1 :]]
        drift_mean = sum(drifts) / len(drifts) if drifts else 0.0
        drift_max = max(drifts) if drifts else 0

    out: Dict[str, Any] = {
        "path": normalize_path_for_web(history_path),
        "category": run_entry.get("category"),
        "model_mix": run_entry.get("model_mix"),
        "init_val": run_entry.get("init_val"),
        "final_success": bool(run_entry.get("final_success")),
        "any_success": bool(run_entry.get("any_success")),
        "final_x": run_entry.get("final_x"),
        "best_x_any": best_any_x,
        "best_x_feasible": best_feasible_x,
        "x_trace": x_trace,
        "collective_trace": collective_trace,
        "agreement_trace": agreement_trace,
        "accept_count_trace": accept_count_trace,
        "distance_to_best_trace": distance_trace,
        "first_agreement_round": first_agree_round,
        "agreements_count": agreements_count,
        "final_agreed": final_agreed,
        "agreement_lost_after_first": agreement_lost_after_first,
        "collective_improve_ratio": collective_improve_ratio,
        "toward_best_ratio": toward_opt_ratio,
        "post_first_agreement_drift_mean": drift_mean,
        "post_first_agreement_drift_max": drift_max,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-run dynamics summary for behavior analysis.")
    ap.add_argument(
        "--summary",
        default="viewer/metrics_summary.json",
        help="Input metrics summary JSON (contains runs[] with history paths).",
    )
    ap.add_argument(
        "--out",
        default="viewer/dynamics_summary.json",
        help="Output JSON to write (for viewer HTML to load).",
    )
    ap.add_argument(
        "--best-feasible-x",
        type=int,
        default=2,
        help=(
            "Optional: hardcode the best feasible x for ALL runs (skips reading polynomial_functions). "
            "Example: --best-feasible-x 2"
        ),
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_path = Path(args.out)

    data = json.loads(summary_path.read_text())
    runs = data.get("runs") or []
    if not isinstance(runs, list):
        raise SystemExit("Expected `runs` list in summary JSON.")

    out_runs: List[Dict[str, Any]] = []
    missing = 0
    skipped = 0
    for r in runs:
        if not isinstance(r, dict):
            skipped += 1
            continue
        dyn = compute_run_dynamics(r, best_feasible_override=args.best_feasible_x)
        if dyn is None:
            # differentiate missing history vs trace-less
            history_path = resolve_history_path(r.get("path", ""))
            if history_path is None:
                missing += 1
            else:
                skipped += 1
            continue
        out_runs.append(dyn)

    out_obj = {
        "generated_from": str(summary_path).replace("\\", "/"),
        "count": len(out_runs),
        "missing_history": missing,
        "skipped": skipped,
        "runs": out_runs,
    }
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False))
    print(f"Wrote {len(out_runs)} run dynamics to {out_path}")
    if missing or skipped:
        print(f"Missing history: {missing}, skipped (no trace/invalid): {skipped}")


if __name__ == "__main__":
    main()
