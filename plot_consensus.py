import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def find_history_file(run_dir: Path) -> Optional[Path]:
    """Pick the first history*.json file in the run directory."""
    candidates = sorted(run_dir.glob("history*.json"))
    return candidates[0] if candidates else None


def load_trace(history_path: Path) -> List[dict]:
    with history_path.open("r") as fh:
        data = json.load(fh)
    return data.get("polynomial_trace", [])


def first_all_accept(trace: List[dict]) -> Optional[int]:
    """Return the first round index where all agents accepted, or None."""
    for idx, entry in enumerate(trace):
        accepted = entry.get("accepted", {})
        if accepted and all(accepted.values()):
            raw_round = entry.get("round", idx)
            try:
                return int(raw_round)
            except (TypeError, ValueError):
                return idx
    return None


def summarize_run(run_dir: Path) -> Optional[dict]:
    history_path = find_history_file(run_dir)
    if history_path is None:
        return None
    trace = load_trace(history_path)
    if not trace:
        return None

    first_accept = first_all_accept(trace)
    last_entry = trace[-1]
    raw_round = last_entry.get("round", len(trace) - 1)
    if isinstance(raw_round, int):
        final_round = raw_round
    else:
        try:
            final_round = int(raw_round)
        except (TypeError, ValueError):
            final_round = len(trace) - 1
    final_all = False
    accepted = last_entry.get("accepted", {})
    if accepted:
        final_all = all(accepted.values())
    final_x = last_entry.get("x")
    return {
        "run": run_dir.name,
        "first_all_round": first_accept,
        "final_round": final_round,
        "final_all": final_all,
        "final_x": final_x,
    }


def plot_summary(summaries: List[dict], output_path: Path):
    if not summaries:
        raise SystemExit("No runs found to summarize.")

    def sort_key(run_label: str):
        prefix = run_label.split(".", 1)[0]
        try:
            return (0, float(prefix))
        except ValueError:
            return (1, run_label)

    summaries = sorted(summaries, key=lambda s: sort_key(s["run"]))

    labels = [s["run"] for s in summaries]
    x_pos = list(range(len(labels)))

    first_vals = [s["first_all_round"] if s["first_all_round"] is not None else float("nan") for s in summaries]
    final_vals = [s["final_round"] for s in summaries]

    fig, ax = plt.subplots(figsize=(12, 6))

    # First consensus markers
    ax.scatter(
        x_pos,
        first_vals,
        marker="o",
        color="#1f77b4",
        label="First all-accept",
    )

    # Final round markers (color by whether everyone still agrees)
    final_colors = ["#2ca02c" if s["final_all"] else "#d62728" for s in summaries]
    ax.scatter(
        x_pos,
        final_vals,
        marker="s",
        color=final_colors,
        label="Final round (green=all agree, red=not all)",
    )

    # Annotate final_x above final markers
    for xp, yv, s in zip(x_pos, final_vals, summaries):
        fx = s["final_x"]
        if fx is not None:
            ax.text(xp, yv + 0.3, f"x={fx}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_pos, labels, rotation=45, ha="right")
    ax.set_ylabel("Round index")
    ax.set_xlabel("Run")
    ax.set_title("Consensus timing per run")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize consensus timing (first all-accept vs final) across runs."
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Directory containing run folders like 1.1, 2.1, ...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default: <base_dir>/consensus_summary.png)",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    summaries = []
    for sub in sorted(base_dir.iterdir()):
        if not sub.is_dir():
            continue
        # Expect run directories like "1.1", "2.1", etc.; skip aux folders.
        name = sub.name
        if not any(char.isdigit() for char in name) or "." not in name:
            continue
        summary = summarize_run(sub)
        if summary:
            summaries.append(summary)

    output_path = args.output if args.output else base_dir / "consensus_summary.png"
    plot_summary(summaries, output_path)


if __name__ == "__main__":
    main()
