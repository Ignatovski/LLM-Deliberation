# Polynomial visualization helper
# -------------------------------
# This script reads a history produced by polynomial/main_polynomial.py and plots:
#   1) The negotiated x over time.
#   2) Each agent's utility f_i(x) vs. its acceptance threshold.
#   3) A decision strip showing whether x moved up/down/same and which agents accepted per round.
# It is intentionally standalone: point it at any saved history JSON and optionally --save a PNG.

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_trace(history_path: Path) -> Tuple[List[Dict], Dict]:
    """
    Load the JSON file and pull out the polynomial_trace array plus the rest of the log.
    polynomial_trace is appended by polynomial/main_polynomial.py each time record_state(...) runs.
    """
    with history_path.open("r") as f:
        data = json.load(f)
    trace = data.get("polynomial_trace")
    if not trace:
        raise ValueError(
            f"No polynomial_trace found in {history_path}. "
            "Run polynomial/main_polynomial.py with the updated logging."
        )
    return trace, data


def load_thresholds(output_dir: Path) -> Dict[str, float]:
    """
    Reconstruct the thresholds (τ_i) by re-reading the copied config + polynomial_functions files
    that sit next to the history. This mirrors how the main script copies its inputs.
    """
    config_path = output_dir / "config.txt"
    thresholds: Dict[str, float] = {}
    if not config_path.exists():
        return thresholds

    with config_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    name_to_file: Dict[str, str] = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        agent_name, file_name = parts[0], parts[1]
        name_to_file[agent_name] = file_name

    functions_dir = output_dir / "polynomial_functions"
    for agent_name, file_name in name_to_file.items():
        profile_path = functions_dir / f"{file_name}.txt"
        if not profile_path.exists():
            continue
        with profile_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0].upper() == "THRESHOLD":
                    thresholds[agent_name] = float(parts[1])
                    break
    return thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Visualize utilities from a polynomial negotiation run."
    )
    parser.add_argument(
        "--history",
        required=True,
        help="Path to the saved history JSON (output/.../historyHH_MM_SS.json).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the figure (PNG). If omitted, displays interactively.",
    )
    args = parser.parse_args()

    history_path = Path(args.history).expanduser().resolve()
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    trace, data = load_trace(history_path)

    # Build sequential index for plotting and keep human-readable labels.
    steps = list(range(len(trace)))
    labels = [str(entry.get("round", idx)) for idx, entry in enumerate(trace)]
    x_values = [entry.get("x", 0) for entry in trace]

    # Gather utilities per agent.
    utility_series: Dict[str, List[float]] = {}
    for entry in trace:
        utilities = entry.get("utilities", {})
        for agent, value in utilities.items():
            utility_series.setdefault(agent, []).append(value)
        # Ensure all agents have same length by padding with None if missing.
        missing_agents = set(utility_series.keys()) - set(utilities.keys())
        for agent in missing_agents:
            utility_series[agent].append(float("nan"))

    output_dir = history_path.parent
    thresholds = load_thresholds(output_dir)

    agent_names = sorted(utility_series.keys())

    fig, (ax_x, ax_u, ax_state) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 3, 1]}
    )

    ax_x.plot(steps, x_values, marker="o", color="black")
    ax_x.set_ylabel("x")
    ax_x.set_title("Negotiated value of x over time")
    ax_x.grid(True, linestyle="--", alpha=0.4)

    for agent, values in utility_series.items():
        ax_u.plot(steps, values, marker="o", label=agent)
        if agent in thresholds:
            ax_u.axhline(
                thresholds[agent],
                linestyle="--",
                color=ax_u.lines[-1].get_color(),
                alpha=0.5,
                label=f"{agent} threshold",
            )

    ax_u.set_ylabel("Utility f_i(x)")
    ax_u.set_xlabel("Negotiation step")
    ax_u.set_title("Agent utilities vs. thresholds")
    ax_u.grid(True, linestyle="--", alpha=0.4)
    ax_u.legend(loc="best")

    ax_u.set_xticks(steps)
    ax_u.set_xticklabels(labels, rotation=45, ha="right")

    # Decision-strip subplot
    directions = []
    prev_x = None
    for value in x_values:
        if prev_x is None:
            directions.append("•")
        else:
            if value > prev_x:
                directions.append("↑")
            elif value < prev_x:
                directions.append("↓")
            else:
                directions.append("→")
        prev_x = value

    ax_state.set_title("Move direction & acceptance per round")
    ax_state.set_yticks(range(len(agent_names)))
    ax_state.set_yticklabels(agent_names)
    ax_state.set_xlabel("Negotiation step")
    ax_state.set_xlim(-0.5, len(steps) - 0.5)
    ax_state.set_ylim(-0.5, len(agent_names) + 0.5)
    ax_state.grid(True, linestyle=":", alpha=0.3, axis="x")

    # Arrow glyphs above the strip show whether x increased (↑), decreased (↓), held (→), or
    # mark the very first step (•). Makes it easy to spot the negotiation flow.
    for idx, arrow in enumerate(directions):
        ax_state.text(idx, len(agent_names) + 0.1, arrow, ha="center", va="bottom", fontsize=12)

    # Colored squares convey acceptance status per agent/round: green = ACCEPT, red = hold.
    for agent_idx, agent in enumerate(agent_names):
        for step_idx, entry in enumerate(trace):
            accepted = entry.get("accepted", {}).get(agent, False)
            color = "#2ca02c" if accepted else "#d62728"
            ax_state.scatter(step_idx, agent_idx, color=color, s=60, marker="s")

    if args.save:
        save_path = Path(args.save).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
