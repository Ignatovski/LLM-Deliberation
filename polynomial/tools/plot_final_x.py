import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


RESULT_PRIORITIES = ["results_7.json", "results_-7.json", "results_0.json"]


def pick_results_file(folder: Path) -> Path | None:
    """Pick a results file inside a run folder based on known priorities."""
    for name in RESULT_PRIORITIES:
        candidate = folder / name
        if candidate.exists():
            return candidate
    matches = sorted(folder.glob("results_*.json"))
    return matches[0] if matches else None


def load_final_x(base_dir: Path) -> List[Tuple[str, float]]:
    """
    Collect (label, final_x) pairs from results_*.json files under base_dir.
    Only immediate subdirectories ending with '.1' are considered to match the
    expected naming convention (e.g., 1.1, 2.1, ...).
    """
    points: List[Tuple[str, float]] = []
    def numeric_key(path: Path):
        # Sort subfolders numerically on the prefix before the dot (e.g., "10.1" -> 10),
        # and fall back to the name for non-numeric prefixes.
        name = path.name
        prefix = name.split(".", 1)[0]
        try:
            return (0, float(prefix))
        except ValueError:
            return (1, name)

    for sub in sorted(base_dir.iterdir(), key=numeric_key):
        if not (sub.is_dir() and sub.name.endswith(".1")):
            continue
        results_file = pick_results_file(sub)
        if results_file is None:
            continue
        with results_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        runs = data.get("runs", [])
        for idx, run in enumerate(runs, start=1):
            final_x = run.get("final_x")
            if final_x is None:
                continue
            # Use folder name as the primary label; include run index if multiple per folder.
            label = sub.name if len(runs) == 1 else f"{sub.name}:r{idx}"
            points.append((label, float(final_x)))
    return points


def plot_points(base_dir: Path, output_path: Path) -> bool:
    points = load_final_x(base_dir)
    if not points:
        return False
    # Force a consistent visual scale across plots to spot differences quickly.
    y_min, y_max = -10, 10

    labels, values = zip(*points)
    xs = list(range(1, len(values) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, values, marker="o", linestyle="-", color="#1f77b4")
    ax.set_xticks(xs, labels=labels, rotation=45, ha="right")
    ax.set_xlabel("Runs")
    ax.set_ylabel("final_x")
    ax.set_title(base_dir.name)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.3)

    avg = sum(values) / len(values)
    ax.text(
        0.99,
        0.01,
        f"avg final_x = {avg:.3f}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot final_x across runs for polynomial game outputs."
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Directory containing *.1 subfolders with results_*.json (e.g., .../poly_x7)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the plot (default: <base_dir>/<base_name>_final_x.png)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "If base_dir only contains subfolders (e.g., poly_x0, poly_x7), "
            "plot each of them automatically."
        ),
    )
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    if args.recursive and args.output is not None:
        raise SystemExit("--output cannot be combined with --recursive (ambiguous)")

    output_path = (
        args.output
        if args.output
        else base_dir / f"{base_dir.name}_final_x.png"
    )
    plotted = plot_points(base_dir, output_path)
    if plotted:
        return

    # If nothing was found and the user asked for recursion, try immediate subfolders.
    if not args.recursive:
        raise SystemExit(f"No final_x values found under {base_dir}")

    child_dirs = [p for p in sorted(base_dir.iterdir()) if p.is_dir()]
    any_plotted = False
    for child in child_dirs:
        child_output = child / f"{child.name}_final_x.png"
        if plot_points(child, child_output):
            any_plotted = True
    if not any_plotted:
        raise SystemExit(f"No final_x values found under {base_dir} or its subfolders")


if __name__ == "__main__":
    main()
