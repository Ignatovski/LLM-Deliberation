#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Poly:
    coeffs: Tuple[float, ...]  # (a,b) or (a,b,c)

    def eval(self, x: float) -> float:
        if len(self.coeffs) == 2:
            a, b = self.coeffs
            return a * x + b
        if len(self.coeffs) == 3:
            a, b, c = self.coeffs
            return a * x * x + b * x + c
        raise ValueError(f"Unsupported COEFFS length: {len(self.coeffs)}")


def parse_coeffs(path: Path) -> Poly:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("COEFFS"):
            parts = line.split()[1:]
            coeffs = tuple(float(p) for p in parts)
            if len(coeffs) not in (2, 3):
                raise ValueError(f"{path}: COEFFS must have 2 or 3 numbers; got {len(coeffs)}")
            return Poly(coeffs=coeffs)
    raise ValueError(f"{path}: missing COEFFS line")


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def iter_history_files(root: Path) -> List[Path]:
    return sorted(root.rglob("history*.json"))


def load_hist(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_trace(hist: dict) -> Tuple[List[int], List[int], List[Dict[str, float]], List[str]]:
    trace = hist.get("polynomial_trace") or []
    rounds = hist.get("rounds") or []
    n = min(len(trace), len(rounds))
    trace = trace[:n]
    rounds = rounds[:n]

    round_idx: List[int] = []
    xs: List[int] = []
    utilities: List[Dict[str, float]] = []
    proposers: List[str] = []

    prev_r: Optional[int] = None
    for i, (t, r) in enumerate(zip(trace, rounds)):
        raw_round = t.get("round")
        try:
            rr = int(raw_round)
        except (TypeError, ValueError):
            rr = (prev_r + 1) if prev_r is not None else i
        prev_r = rr

        x = int(t.get("x"))
        u = {str(k): float(v) for k, v in (t.get("utilities") or {}).items()}
        proposer = str(r.get("agent") or "")

        round_idx.append(rr)
        xs.append(x)
        utilities.append(u)
        proposers.append(proposer)

    return round_idx, xs, utilities, proposers


def default_agent_polys(game_dir: Path) -> Dict[str, Poly]:
    # Standard polynomial game mapping.
    pf = game_dir / "polynomial_functions"
    mapping = {
        "Analyst A": pf / "analyst_a.txt",
        "Builder B": pf / "builder_b.txt",
        "Critic C": pf / "critic_c.txt",
        "Delegate D": pf / "delegate_d.txt",
    }
    return {k: parse_coeffs(v) for k, v in mapping.items()}


def nice_ylim(max_abs: float) -> float:
    if not math.isfinite(max_abs) or max_abs <= 0:
        return 1.0
    # Round up to a "nice" number.
    exp = 10 ** int(math.floor(math.log10(max_abs)))
    for m in (1, 2, 5, 10):
        if m * exp >= max_abs:
            return float(m * exp)
    return float(10 * exp)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate behavior across many runs without misleading averages: "
            "plots per-proposer distributions of collective regret and %improve/%worsen."
        )
    )
    ap.add_argument("--runs-root", required=True, help="Folder containing many run subfolders with history*.json")
    ap.add_argument("--game-dir", required=True, help="Game directory containing polynomial_functions/*.txt")
    ap.add_argument("--save", required=True, help="Output image path (png)")
    ap.add_argument("--best-x", type=int, default=2, help="Best feasible x reference (default: 2)")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    game_dir = Path(args.game_dir)
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_x = int(args.best_x)
    polys = default_agent_polys(game_dir)
    best_collective = mean([p.eval(best_x) for p in polys.values()])

    histories = iter_history_files(runs_root)
    if not histories:
        raise SystemExit(f"No history*.json found under {runs_root}")

    # Aggregate across all proposals in all runs.
    fixed_order = ["Analyst A", "Builder B", "Critic C", "Delegate D"]
    regret_by: Dict[str, List[float]] = {k: [] for k in fixed_order}
    improve_by: Dict[str, int] = {k: 0 for k in fixed_order}
    worsen_by: Dict[str, int] = {k: 0 for k in fixed_order}
    same_by: Dict[str, int] = {k: 0 for k in fixed_order}

    total_props = 0
    used_histories = 0

    for hpath in histories:
        hist = load_hist(hpath)
        _, xs, utilities_list, proposers = extract_trace(hist)
        if not xs:
            continue
        used_histories += 1

        collective_scores = [mean([float(v) for v in u.values()]) for u in utilities_list]

        prev_collective: Optional[float] = None
        for x, coll, proposer in zip(xs, collective_scores, proposers):
            if proposer not in regret_by:
                continue
            total_props += 1
            regret = best_collective - coll
            regret_by[proposer].append(regret)

            if prev_collective is not None:
                d = coll - prev_collective
                if d > 1e-9:
                    improve_by[proposer] += 1
                elif d < -1e-9:
                    worsen_by[proposer] += 1
                else:
                    same_by[proposer] += 1
            prev_collective = coll

    # Plot
    present = [k for k in fixed_order if regret_by[k]]
    if not present:
        raise SystemExit("No proposer data found (unexpected proposer names?)")

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    data = [regret_by[k] for k in present]
    ax1.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    ax1.set_xticks(range(1, len(present) + 1))
    ax1.set_xticklabels([k.replace(" ", "\n") for k in present])
    ax1.set_title(f"Collective regret per proposer (best x={best_x})")
    ax1.set_ylabel("Regret = collective(best) - collective(proposed)\n(lower is better; 0 = best)")
    ax1.grid(True, axis="y", alpha=0.2)
    y_max = nice_ylim(max(abs(v) for k in present for v in regret_by[k]) * 1.05)
    ax1.set_ylim(-y_max, y_max)

    ax2 = fig.add_subplot(gs[0, 1])
    improves = [improve_by[k] for k in present]
    worsens = [worsen_by[k] for k in present]
    sames = [same_by[k] for k in present]
    totals = [max(1, i + w + s) for i, w, s in zip(improves, worsens, sames)]
    imp_pct = [i / t * 100 for i, t in zip(improves, totals)]
    wor_pct = [w / t * 100 for w, t in zip(worsens, totals)]
    sam_pct = [s / t * 100 for s, t in zip(sames, totals)]

    x = list(range(len(present)))
    ax2.bar(x, imp_pct, label="improve", color="#2ca02c")
    ax2.bar(x, sam_pct, bottom=imp_pct, label="same", color="#c7c7c7")
    ax2.bar(x, wor_pct, bottom=[a + b for a, b in zip(imp_pct, sam_pct)], label="worsen", color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.replace(" ", "\n") for k in present])
    ax2.set_ylim(0, 100)
    ax2.set_title("Direction of change vs previous proposal\n(only within-run deltas)")
    ax2.set_ylabel("Percent of proposals")
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.legend(loc="upper right")

    fig.suptitle(
        f"Across runs: {runs_root} | histories={used_histories} proposals={total_props}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

