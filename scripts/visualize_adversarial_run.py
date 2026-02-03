#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_history(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("History JSON must be an object at top-level.")
    if "polynomial_trace" not in data or "rounds" not in data:
        raise ValueError("History JSON missing required keys: 'polynomial_trace' and/or 'rounds'.")
    return data


def extract_series(history: Dict[str, Any]) -> Tuple[List[int], List[int], List[float], List[str], List[Dict[str, bool]]]:
    trace = history.get("polynomial_trace") or []
    rounds = history.get("rounds") or []

    if len(trace) != len(rounds):
        # Some runs may store extra non-proposal rounds; align conservatively.
        n = min(len(trace), len(rounds))
        trace = trace[:n]
        rounds = rounds[:n]

    round_idx: List[int] = []
    xs: List[int] = []
    collective: List[float] = []
    proposers: List[str] = []
    accepted_list: List[Dict[str, bool]] = []

    prev_r: Optional[int] = None
    for idx, (t_entry, r_entry) in enumerate(zip(trace, rounds)):
        raw_round = t_entry.get("round")
        try:
            r = int(raw_round)
        except (TypeError, ValueError):
            # Some histories include a terminal "final" round; place it after the last numeric round.
            r = (prev_r + 1) if prev_r is not None else idx
        x = int(t_entry.get("x"))
        utilities: Dict[str, float] = t_entry.get("utilities") or {}
        accepted: Dict[str, bool] = t_entry.get("accepted") or {}
        proposer = str(r_entry.get("agent") or "")

        round_idx.append(r)
        xs.append(x)
        collective.append(_mean([float(v) for v in utilities.values()]))
        proposers.append(proposer)
        accepted_list.append({str(k): bool(v) for k, v in accepted.items()})
        prev_r = r

    return round_idx, xs, collective, proposers, accepted_list


def first_unanimous_round(accepted_list: List[Dict[str, bool]]) -> Optional[int]:
    for idx, accepted in enumerate(accepted_list):
        if accepted and all(bool(v) for v in accepted.values()):
            return idx
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize one adversarial polynomial run: x over time, collective score per proposal, and acceptance grid."
    )
    ap.add_argument("--history", required=True, help="Path to history*.json")
    ap.add_argument("--save", required=True, help="Output image path (png)")
    ap.add_argument("--best-x", type=int, default=2, help="Reference (best feasible) x to mark (default: 2)")
    ap.add_argument(
        "--show-proposer-distributions",
        action="store_true",
        help="Add per-proposer boxplots for collective score and |x-best_x| (within this single run).",
    )
    args = ap.parse_args()

    history_path = Path(args.history)
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history = load_history(history_path)
    rounds, xs, collective, proposers, accepted_list = extract_series(history)

    best_x = int(args.best_x)
    dist = [abs(x - best_x) for x in xs]
    first_agree_idx = first_unanimous_round(accepted_list)

    # Gather agent ordering from first accepted dict (stable across run).
    agent_names: List[str] = []
    for accepted in accepted_list:
        if accepted:
            agent_names = list(accepted.keys())
            break

    show_dist = bool(args.show_proposer_distributions)
    fig = plt.figure(figsize=(12, 11 if show_dist else 9), constrained_layout=True)
    if show_dist:
        gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.2, 1.0, 1.0])
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.2, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rounds, xs, marker="o", linewidth=2, color="#1f77b4")
    ax1.axhline(best_x, linestyle="--", linewidth=1.5, color="#2ca02c", label=f"best x={best_x}")
    if first_agree_idx is not None:
        ax1.axvline(rounds[first_agree_idx], linestyle=":", linewidth=1.5, color="#ff7f0e", label="first unanimous")
    ax1.set_title("Proposed x over rounds")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("x")
    ax1.set_xticks(rounds)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rounds, collective, marker="o", linewidth=2, color="#17becf", label="collective score (mean utility)")
    ax2_2 = ax2.twinx()
    ax2_2.plot(rounds, dist, marker="s", linewidth=1.5, color="#9467bd", alpha=0.8, label=f"|x-{best_x}|")
    if first_agree_idx is not None:
        ax2.axvline(rounds[first_agree_idx], linestyle=":", linewidth=1.5, color="#ff7f0e")
    ax2.set_title("Per-proposal collective score and distance to best x")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Collective score")
    ax2_2.set_ylabel(f"Distance to x={best_x}")
    ax2.set_xticks(rounds)
    ax2.grid(True, alpha=0.25)
    # Legend combining both axes
    lines = ax2.get_lines() + ax2_2.get_lines()
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    next_row = 2
    if show_dist:
        # Per-proposer distributions within this run (no averaging).
        axd = fig.add_subplot(gs[next_row, 0])
        next_row += 1

        fixed_order = ["Analyst A", "Builder B", "Critic C", "Delegate D"]
        present = [p for p in fixed_order if p in set(proposers)]
        if not present:
            present = sorted(set(proposers))

        coll_by = {p: [] for p in present}
        dist_by = {p: [] for p in present}
        for p, c, d in zip(proposers, collective, dist):
            if p in coll_by:
                coll_by[p].append(float(c))
                dist_by[p].append(float(d))

        data_coll = [coll_by[p] for p in present]
        data_dist = [dist_by[p] for p in present]

        axd.set_title("Within-run distributions (per proposer) — avoids misleading averages")
        axd.grid(True, axis="y", alpha=0.2)

        # Two boxplots on one axis using offsets.
        positions = list(range(1, len(present) + 1))
        offset = 0.18
        bp1 = axd.boxplot(
            data_coll,
            positions=[p - offset for p in positions],
            widths=0.28,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#111", "linewidth": 1.5},
        )
        for b in bp1["boxes"]:
            b.set_facecolor("#c7f0ef")
            b.set_edgecolor("#17becf")

        axd2 = axd.twinx()
        bp2 = axd2.boxplot(
            data_dist,
            positions=[p + offset for p in positions],
            widths=0.28,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#111", "linewidth": 1.5},
        )
        for b in bp2["boxes"]:
            b.set_facecolor("#e8dcff")
            b.set_edgecolor("#9467bd")

        axd.set_xticks(positions)
        axd.set_xticklabels([p.replace(" ", "\n") for p in present])
        axd.set_ylabel("Collective score (mean utility)")
        axd2.set_ylabel(f"Distance to x={best_x}")

        # Legend proxy
        axd.plot([], [], color="#17becf", label="collective score (box)")
        axd.plot([], [], color="#9467bd", label=f"|x-{best_x}| (box, right axis)")
        axd.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[next_row, 0])
    if agent_names:
        # Build acceptance matrix: agents x rounds (1 = accept, 0 = reject).
        mat = []
        for a in agent_names:
            mat.append([1 if accepted.get(a, False) else 0 for accepted in accepted_list])
        ax3.imshow(mat, aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=1)
        ax3.set_yticks(list(range(len(agent_names))))
        ax3.set_yticklabels(agent_names)
        ax3.set_xticks(list(range(len(rounds))))
        ax3.set_xticklabels([str(r) for r in rounds], rotation=0)
        ax3.set_title("Acceptance per round (green=accept, red=reject)")
        ax3.set_xlabel("Round")
        # Annotate proposer initial for each round above the grid.
        for i, proposer in enumerate(proposers):
            label = proposer.split()[-1][0] if proposer else "?"
            ax3.text(i, -0.6, label, ha="center", va="bottom", fontsize=9, color="#333")
        ax3.text(-0.6, -0.6, "Proposer", ha="right", va="bottom", fontsize=9, color="#333")
    else:
        ax3.text(0.5, 0.5, "No acceptance data found.", ha="center", va="center")
        ax3.axis("off")

    fig.suptitle(f"Run: {history_path}", fontsize=11)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
