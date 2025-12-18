#!/usr/bin/env python3
"""
Plot collective utility for the polynomial game using fixed utility functions:
  f_A(x) = 2x + 10
  f_B(x) = -x^2 + 4x + 8
  f_C(x) = x^2 + 5
  f_D(x) = 10 - 3x

We generate two PNGs in viewer/plots/:
  - collective_final_agree.png: summed utility at final x for each run.
  - collective_first_agree.png: summed utility at the first all-accept x per run.

Both plots include the total utility curve, the feasible band x ∈ [-2, 2],
and reference lines at x = -2 (light red, dotted) and x = 2 (dark red).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path("viewer/metrics_summary.json")
PLOTS_DIR = Path("viewer/plots")
FEASIBLE_RANGE = (-2, 2)


def total_utility(x: float) -> float:
    """Sum of the four utility functions at x."""
    return (2 * x + 10) + (-x ** 2 + 4 * x + 8) + (x ** 2 + 5) + (10 - 3 * x)


def load_summary(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("runs", [])


def first_agree_x(trace: List[dict]) -> Optional[float]:
    for entry in trace:
        acc = entry.get("accepted", {})
        if acc and all(acc.values()):
            return entry.get("x")
    return None


def collect_points(runs: Iterable[Dict]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return (final_points, first_points) as lists of (x, total_utility(x))."""
    final_pts: List[Tuple[float, float]] = []
    first_pts: List[Tuple[float, float]] = []
    for run in runs:
        path = run.get("path")
        if not path:
            continue
        hist_path = Path(path)
        if not hist_path.exists():
            # Try relative to repo root
            alt = Path(".") / hist_path
            if alt.exists():
                hist_path = alt
            else:
                continue
        try:
            with hist_path.open("r", encoding="utf-8") as fh:
                hist = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        trace = hist.get("polynomial_trace") or []
        if not trace:
            continue
        # Final point
        fx = trace[-1].get("x")
        if isinstance(fx, (int, float)):
            final_pts.append((fx, total_utility(fx)))
        # First agreement point
        ax = first_agree_x(trace)
        if isinstance(ax, (int, float)):
            first_pts.append((ax, total_utility(ax)))
    return final_pts, first_pts


def plot_points(points: List[Tuple[float, float]], title: str, outfile: Path) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    xs_curve = np.linspace(-10, 10, 400)
    ys_curve = total_utility(xs_curve)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs_curve, ys_curve, color="#2563eb", label="Σ utilities (theoretical)")
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, color="#0ea5e9", edgecolors="#0f172a", linewidths=0.6, alpha=0.85, label="Observed runs")

    # Feasible band and markers
    ax.axvspan(FEASIBLE_RANGE[0], FEASIBLE_RANGE[1], color="#ef4444", alpha=0.08, label="Feasible band [-2, 2]")
    ax.axvline(FEASIBLE_RANGE[0], color="#ef4444", linestyle=":", linewidth=1.2, label="x = -2")
    ax.axvline(FEASIBLE_RANGE[1], color="#b91c1c", linestyle="-", linewidth=1.4, label="x = 2")

    ax.set_xlabel("x")
    ax.set_ylabel("Total utility (Σ f_i)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot collective utility curves from metrics_summary.json.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="Path to metrics_summary.json (default: viewer/metrics_summary.json).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional category filter (e.g., 'Uniform', 'All AI', 'All Human').",
    )
    parser.add_argument(
        "--model-mix",
        default=None,
        help="Optional model_mix filter (e.g., 'gpt-5x4').",
    )
    args = parser.parse_args()

    runs = load_summary(args.summary)
    if args.category:
        runs = [r for r in runs if r.get("category") == args.category]
    if args.model_mix:
        runs = [r for r in runs if r.get("model_mix") == args.model_mix]

    label_bits = []
    if args.category:
        label_bits.append(args.category)
    if args.model_mix:
        label_bits.append(args.model_mix)
    suffix = " (" + ", ".join(label_bits) + ")" if label_bits else ""

    final_pts, first_pts = collect_points(runs)
    plot_points(
        final_pts,
        f"Collective utility at final state (per run){suffix}",
        PLOTS_DIR / "collective_final_agree.png",
    )
    plot_points(
        first_pts,
        f"Collective utility at first agreement (per run){suffix}",
        PLOTS_DIR / "collective_first_agree.png",
    )
    print(f"Wrote {len(final_pts)} final points and {len(first_pts)} first-agreement points.")


if __name__ == "__main__":
    main()
