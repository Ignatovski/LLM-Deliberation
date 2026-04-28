from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from thesis_plot_style import apply_thesis_style


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "viewer" / "plots" / "thesis" / "cross-Evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLY_METRICS_PATH = ROOT / "viewer" / "metrics_summary.json"
POLY_DYNAMICS_PATH = ROOT / "viewer" / "dynamics_summary.json"
ADV_OBSTRUCTIVE_PATH = ROOT / "summarys" / "metrics_summary.adversarial_obstructive.json"
ADV_TARGETED_PATH = ROOT / "summarys" / "metrics_summary.adversarial_outcome_targeted.json"
CYBER_ROOT = ROOT / "cyber_negotiation_v1" / "games_descriptions" / "cyber_game" / "output"
KNOWING_DOING_GAP_IMG = ROOT / "viewer" / "plots" / "thesis" / "knowing_doing_gap_recent_repeat_distribution.png"

apply_thesis_style(font_size=11, y_grid=True)

COLORS = {
    "GPT-5": "#0072B2",
    "Claude": "#D55E00",
    "Llama": "#009E73",
    "Mixed": "#CC79A7",
    "Neutral": "#4D4D4D",
}

PRIOR_ORDER = ["Uniform", "All Human", "All AI"]
PRIOR_LABELS = ["Uniform", "Human prior\n(All Human)", "LLM prior\n(All AI)"]
MODEL_ORDER = ["GPT-5", "Claude", "Llama", "Mixed"]
CYBER_COND_LABELS = {
    "C1": "Single GPT-5",
    "C2": "Single Claude",
    "C3": "3x GPT-5",
    "C4": "3x Claude",
    "C5": "Mixed",
    "C6": "GPT-5 + LLM prior",
    "C7": "Claude + human prior",
}

SEV_ORDER = {"Compliance": 0, "Info": 1, "Low": 2, "Medium": 3, "High": 4}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [float(v) for v in values if v is not None]
    return mean(numeric) if numeric else None


def rate(values: Iterable[Any]) -> float:
    vals = list(values)
    return sum(1 for v in vals if bool(v)) / len(vals) if vals else float("nan")


def pct_label(value: float) -> str:
    return f"{value * 100:.1f}%"


def setup_rate_axis(ax: plt.Axes, ylim: Tuple[float, float] = (0.0, 1.0)) -> None:
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / f"{name}.png"
    pdf_path = OUT_DIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def normalize_prior(category: str) -> str:
    return {
        "polynomial_game": "Uniform",
        "polynomial_game_all_AI": "All AI",
        "polynomial_game_human": "All Human",
    }.get(category, category)


def model_family(model_mix: str) -> str:
    if model_mix == "gpt-5x4":
        return "GPT-5"
    if model_mix == "claude-sonnet-4-5x4":
        return "Claude"
    if model_mix == "Llama-3.3-70B-Instructx4":
        return "Llama"
    return "Mixed"


def load_poly_runs() -> List[Dict[str, Any]]:
    runs = list(load_json(POLY_METRICS_PATH).get("runs") or [])
    for row in runs:
        row["prior_group"] = normalize_prior(str(row.get("category") or ""))
        row["family"] = model_family(str(row.get("model_mix") or ""))
    return runs


def load_poly_dynamics() -> List[Dict[str, Any]]:
    runs = list(load_json(POLY_DYNAMICS_PATH).get("runs") or [])
    for row in runs:
        row["prior_group"] = normalize_prior(str(row.get("category") or ""))
        row["family"] = model_family(str(row.get("model_mix") or ""))
    return runs


def condition_sort_key(condition_id: str) -> int:
    return int(condition_id[1:]) if len(condition_id) > 1 and condition_id[1:].isdigit() else 999


def committee_type_value(snapshot: Dict[str, Any]) -> str:
    value = snapshot.get("committee_type_label") or snapshot.get("committee_type")
    return str(value) if value is not None else "None"


def committee_severity_value(snapshot: Dict[str, Any]) -> str:
    value = snapshot.get("committee_exact")
    if isinstance(value, dict) and value.get("severity"):
        return str(value.get("severity"))
    value = snapshot.get("committee_exact_severity")
    return str(value) if value is not None else "None"


def count_transitions(values: Sequence[str]) -> int:
    return sum(1 for left, right in zip(values, values[1:]) if left != right)


def load_cyber_latest() -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for metrics_path in sorted(CYBER_ROOT.rglob("metrics_*.json")):
        relative_parts = metrics_path.relative_to(CYBER_ROOT).parts
        if not relative_parts or relative_parts[0] == "llm_evaluator":
            continue
        payload = load_json(metrics_path)
        report = dict(payload.get("run_report") or {})
        if not report:
            continue
        category = relative_parts[0]
        scenario_id = str(report.get("scenario_id") or "")
        condition_id = str(report.get("condition_id") or "")
        run_id = str(report.get("run_id") or metrics_path.stem.replace("metrics_", "", 1))
        if not category or not scenario_id or not condition_id:
            continue
        history_path = metrics_path.parent / f"{run_id}.json"
        if not history_path.exists():
            continue
        history = load_json(history_path)
        if history.get("run_status") != "completed":
            continue

        trajectory = list(report.get("decision_trajectory") or history.get("decision_trajectory") or [])
        type_path = [committee_type_value(snapshot) for snapshot in trajectory]
        severity_path = [committee_severity_value(snapshot) for snapshot in trajectory]
        record = {
            "category": category,
            "scenario_id": scenario_id,
            "condition_id": condition_id,
            "run_id": run_id,
            "metrics_path": metrics_path,
            "report": report,
            "type_transitions": count_transitions(type_path),
            "severity_transitions": count_transitions(severity_path),
        }
        key = (category, scenario_id, condition_id)
        current = latest.get(key)
        if current is None or metrics_path.stat().st_mtime > current["metrics_path"].stat().st_mtime:
            latest[key] = record

    # Some older cybersecurity runs only persisted the completed history JSON
    # and not a sibling metrics file. Fill those gaps without overriding the
    # newer metrics-backed records so cross-evaluation figures keep the original
    # scenario set.
    for history_path in sorted(CYBER_ROOT.rglob("history*.json")):
        relative_parts = history_path.relative_to(CYBER_ROOT).parts
        if not relative_parts or relative_parts[0] == "llm_evaluator":
            continue
        history = load_json(history_path)
        if history.get("run_status") != "completed":
            continue

        category = relative_parts[0]
        scenario_id = str(history.get("scenario_id") or "")
        condition_id = str(
            history.get("condition_id")
            or (history.get("condition") or {}).get("condition_id")
            or ((history.get("condition_aggregate") or {}).get("headline_metrics") or {}).get("condition_id")
            or ""
        )
        run_id = str(history.get("run_id") or history_path.stem)
        if not category or not scenario_id or not condition_id:
            continue

        key = (category, scenario_id, condition_id)
        if key in latest:
            continue

        trajectory = list(history.get("decision_trajectory") or [])
        type_path = [committee_type_value(snapshot) for snapshot in trajectory]
        severity_path = [committee_severity_value(snapshot) for snapshot in trajectory]
        latest[key] = {
            "category": category,
            "scenario_id": scenario_id,
            "condition_id": condition_id,
            "run_id": run_id,
            "metrics_path": history_path,
            "report": {
                "scenario_id": scenario_id,
                "condition_id": condition_id,
                "run_id": run_id,
                "headline_metrics": dict(history.get("metrics") or {}),
                "derived_metrics": dict(history.get("derived_metrics") or {}),
                "decision_trajectory": trajectory,
            },
            "type_transitions": count_transitions(type_path),
            "severity_transitions": count_transitions(severity_path),
        }
    return sorted(latest.values(), key=lambda r: (r["category"], r["scenario_id"], condition_sort_key(r["condition_id"])))


def cyber_condition_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cond[record["condition_id"]].append(record)

    out: Dict[str, Dict[str, float]] = {}
    for cond, rows in by_cond.items():
        exact: List[float] = []
        type_correct: List[float] = []
        severity: List[float] = []
        wrong: List[float] = []
        bias: List[float] = []
        under: List[float] = []
        over: List[float] = []
        type_trans: List[float] = []
        sev_trans: List[float] = []
        for row in rows:
            report = row["report"]
            headline = report.get("headline_metrics") or {}
            derived = report.get("derived_metrics") or {}
            if headline.get("FinalCorrectExact") is not None:
                exact.append(float(headline.get("FinalCorrectExact")))
            if headline.get("FinalCorrectType") is not None:
                type_correct.append(float(headline.get("FinalCorrectType")))
            if headline.get("FinalCorrectSeverity") is not None:
                severity.append(float(headline.get("FinalCorrectSeverity")))
            elif headline.get("SeverityBias") is not None:
                severity.append(1.0 if float(headline.get("SeverityBias")) == 0 else 0.0)
            if derived.get("WrongConsensusExact") is not None:
                wrong.append(float(derived.get("WrongConsensusExact")))
            if headline.get("SeverityBias") is not None:
                b = float(headline.get("SeverityBias"))
                bias.append(b)
                under.append(1.0 if b < 0 else 0.0)
                over.append(1.0 if b > 0 else 0.0)
            type_trans.append(float(row["type_transitions"]))
            sev_trans.append(float(row["severity_transitions"]))
        out[cond] = {
            "exact": mean(exact),
            "type": mean(type_correct),
            "severity": mean(severity),
            "wrong": mean(wrong),
            "bias": mean(bias),
            "under": mean(under),
            "over": mean(over),
            "type_trans": mean(type_trans),
            "severity_trans": mean(sev_trans),
        }
    return out


def cyber_category_exact(records: Sequence[Dict[str, Any]], category: str) -> float:
    vals = []
    for row in records:
        if row["category"] != category:
            continue
        headline = row["report"].get("headline_metrics") or {}
        if headline.get("FinalCorrectExact") is not None:
            vals.append(float(headline.get("FinalCorrectExact")))
    return mean(vals)


def make_transfer_matrix() -> None:
    rows = [
        ("Cooperative improvement under dialogue", "Yes", "Partial", "Partial transfer"),
        ("Prior-framing effect", "Yes", "Yes", "Shared"),
        ("Mixed-model stabilization", "Yes", "No", "Does not transfer"),
        ("Late drift / instability", "Yes", "Partial", "Partial transfer"),
        ("Explicit adversarial manipulation", "Yes", "No", "Polynomial-only"),
        ("Protocol/output failures", "Yes", "Partial", "Partial / weaker in cyber"),
        ("Wrong consensus", "No", "Yes", "Cyber-emergent"),
        ("Severity instability", "No", "Yes", "Cyber-emergent"),
        ("Under-severity bias", "No", "Yes", "Cyber-emergent"),
    ]
    cols = ["Phenomenon", "Polynomial", "Cybersecurity", "Transfer status"]
    text = [[phenomenon, poly, cyber, transfer] for phenomenon, poly, cyber, transfer in rows]

    def cell_color(value: str, col: int) -> str:
        if col == 0:
            return "white"
        if value == "Yes":
            return "#D9D9D9"
        if value == "Partial":
            return "#F2F2F2"
        if value == "No":
            return "white"
        if value == "Shared":
            return "#D9D9D9"
        if value in {"Partial transfer", "Partial / weaker in cyber"}:
            return "#F2F2F2"
        return "#E7E6E6"

    cell_colours = [[cell_color(value, idx) for idx, value in enumerate(row)] for row in text]
    fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=300)
    ax.axis("off")
    table = ax.table(
        cellText=text,
        colLabels=cols,
        cellColours=cell_colours,
        colWidths=[0.42, 0.15, 0.17, 0.26],
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#E7E6E6")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_text_props(ha="left")
    save(fig, "cross_framework_transfer_matrix")


def make_prior_effects(poly_runs: Sequence[Dict[str, Any]], cyber_stats: Dict[str, Dict[str, float]]) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.8, 10.0),
        dpi=300,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    ax = axes[0]
    x = np.arange(len(PRIOR_ORDER))
    prior_families = ["GPT-5", "Claude", "Mixed"]
    for family in prior_families:
        family_rows = [r for r in poly_runs if r["family"] == family]
        final_rates = []
        any_rates = []
        for prior in PRIOR_ORDER:
            rows = [r for r in family_rows if r["prior_group"] == prior]
            final_rates.append(rate(r.get("final_success") for r in rows))
            any_rates.append(rate(r.get("any_success") for r in rows))
        color = COLORS[family]
        ax.plot(x, final_rates, marker="o", markersize=8, color=color, linewidth=2.0)
        ax.plot(x, any_rates, marker="s", markersize=8, color=color, linewidth=2.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(PRIOR_LABELS, fontsize=14)
    ax.set_ylabel("Success rate", fontsize=16, labelpad=10)
    ax.set_title("A. Polynomial Prior Effects", fontsize=18, pad=10)
    setup_rate_axis(ax)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=10)

    model_handles = [
        Line2D([0], [0], color=COLORS[f], marker="o", linewidth=0, label=f)
        for f in prior_families
    ]
    metric_handles = [
        Line2D([0], [0], color="#555555", linewidth=1.8, label="Final success"),
        Line2D([0], [0], color="#555555", linewidth=1.8, linestyle="--", label="Any success"),
    ]
    ax.legend(
        handles=model_handles + metric_handles,
        frameon=False,
        fontsize=12,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.98),
        columnspacing=1.1,
        handlelength=2.5,
    )

    ax = axes[1]
    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    gpt_base = [cyber_stats["C3"][m] for m in metrics]
    gpt_prior = [cyber_stats["C6"][m] for m in metrics]
    claude_base = [cyber_stats["C4"][m] for m in metrics]
    claude_prior = [cyber_stats["C7"][m] for m in metrics]
    gpt_x = np.array([0.0, 1.4, 2.8, 4.2])
    claude_x = gpt_x + 6.4
    width = 0.32

    ax.bar(gpt_x - width / 2, gpt_base, width=width, color=COLORS["GPT-5"], edgecolor="#333333", linewidth=0.6)
    ax.bar(gpt_x + width / 2, gpt_prior, width=width, color=COLORS["GPT-5"], edgecolor="#333333", linewidth=0.6, hatch="//")
    ax.bar(claude_x - width / 2, claude_base, width=width, color=COLORS["Claude"], edgecolor="#333333", linewidth=0.6)
    ax.bar(claude_x + width / 2, claude_prior, width=width, color=COLORS["Claude"], edgecolor="#333333", linewidth=0.6, hatch="//")

    ax.set_xticks(np.concatenate([gpt_x, claude_x]))
    ax.set_xticklabels(labels + labels, fontsize=13)
    ax.set_ylabel("Rate", fontsize=16, labelpad=10)
    ax.set_title("B. Cybersecurity Prior Comparison", fontsize=18, pad=10)
    setup_rate_axis(ax)
    ax.axvline((gpt_x[-1] + claude_x[0]) / 2, color="#BFBFBF", linewidth=0.8)
    ax.set_xlim(-0.8, claude_x[-1] + 0.8)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=10)
    ax.text(gpt_x.mean(), -0.13, "GPT-5 (C3/C6)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12)
    ax.text(claude_x.mean(), -0.13, "Claude (C4/C7)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12)
    prior_handles = [
        Patch(facecolor="white", edgecolor="#333333", label="No prior"),
        Patch(facecolor="white", edgecolor="#333333", hatch="//", label="With prior"),
    ]
    ax.legend(
        handles=prior_handles,
        frameon=False,
        title="Committee",
        title_fontsize=14,
        fontsize=13,
        loc="upper right",
    )
    save(fig, "cross_prior_effects_comparison")


def make_mixed_model_transfer(poly_runs: Sequence[Dict[str, Any]], cyber_stats: Dict[str, Dict[str, float]]) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.8, 11.8),
        dpi=300,
        gridspec_kw={"height_ratios": [1.15, 1.0, 0.8]},
    )

    ax = axes[0]
    xs = np.arange(len(MODEL_ORDER))
    prior_styles = {
        "Uniform": ("-", "Uniform"),
        "All Human": ("--", "Human prior"),
        "All AI": (":", "LLM prior"),
    }
    for prior in PRIOR_ORDER:
        linestyle, _ = prior_styles[prior]
        vals = []
        for family in MODEL_ORDER:
            rows = [r for r in poly_runs if r["family"] == family and r["prior_group"] == prior]
            vals.append(rate(r.get("any_success") for r in rows))
        ax.plot(xs, vals, color="#6B7280", linestyle=linestyle, linewidth=1.1, zorder=1)
        for x, y, family in zip(xs, vals, MODEL_ORDER):
            marker = "D" if family == "Mixed" else "s"
            ax.scatter(
                [x],
                [y],
                s=54,
                marker=marker,
                color=COLORS[family],
                edgecolors="#333333",
                linewidths=0.5,
                zorder=3,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(MODEL_ORDER, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylabel("Any success rate", fontsize=16, labelpad=10)
    ax.set_title("A. Polynomial any-success by prior", fontsize=18, pad=10)
    setup_rate_axis(ax, (0.0, 1.0))

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=("D" if family == "Mixed" else "s"),
            color="none",
            markerfacecolor=COLORS[family],
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            markersize=5.5,
            linestyle="None",
            label=family,
        )
        for family in MODEL_ORDER
    ]
    prior_handles = [
        Line2D([0], [0], color="#6B7280", linestyle=prior_styles[prior][0], linewidth=1.1, label=prior_styles[prior][1])
        for prior in PRIOR_ORDER
    ]
    ax.legend(
        handles=model_handles + prior_handles,
        frameon=False,
        fontsize=12,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        columnspacing=1.2,
        handlelength=2.5,
    )

    ax = axes[1]
    metrics = ["exact", "severity", "wrong", "under"]
    labels = ["Exact", "Severity", "Wrong", "Under"]
    width = 0.34
    hom = {m: mean([cyber_stats["C3"][m], cyber_stats["C4"][m]]) for m in metrics}
    mix = {m: cyber_stats["C5"][m] for m in metrics}
    xs = np.array([0.0, 1.5, 3.0, 4.5])
    ax.bar(xs - width / 2, [hom[m] for m in metrics], width=width, color=COLORS["Neutral"], edgecolor="#333333", linewidth=0.6, label="Homogeneous (C3, C4)")
    ax.bar(xs + width / 2, [mix[m] for m in metrics], width=width, color=COLORS["Mixed"], edgecolor="#333333", linewidth=0.6, label="Mixed (C5)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_xlim(-0.8, xs[-1] + 0.8)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylabel("Rate", fontsize=16, labelpad=10)
    ax.set_title("B. Cybersecurity rates", fontsize=18, pad=10)
    setup_rate_axis(ax)
    ax.legend(frameon=False, fontsize=13, loc="upper right")

    ax = axes[2]
    hom_bias = mean([cyber_stats["C3"]["bias"], cyber_stats["C4"]["bias"]])
    mix_bias = cyber_stats["C5"]["bias"]
    ax.bar(
        np.arange(2),
        [hom_bias, mix_bias],
        color=[COLORS["Neutral"], COLORS["Mixed"]],
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["Homogeneous\n(C3, C4)", "Mixed\n(C5)"], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylabel("Severity bias", fontsize=16, labelpad=10)
    ax.set_title("C. Bias", fontsize=18, pad=10)
    ax.set_ylim(-1.0, 0.25)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, "cross_mixed_model_transfer")


def final_repeat_rate(failed_runs: Sequence[Dict[str, Any]]) -> float:
    repeat_flags = []
    for row in failed_runs:
        trace = row.get("x_trace") or []
        if len(trace) < 2:
            continue
        final_x = trace[-1]
        repeat_flags.append(final_x in trace[:-1])
    return rate(repeat_flags)


def make_instability_modes(poly_dyn: Sequence[Dict[str, Any]], cyber_stats: Dict[str, Dict[str, float]]) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.8, 9.6),
        dpi=300,
        gridspec_kw={"height_ratios": [1.05, 0.95]},
    )

    ax = axes[0]
    recent_window = 6
    counts: Counter[int] = Counter()
    for row in poly_dyn:
        if row.get("final_success"):
            continue
        trace = list(row.get("x_trace") or [])
        if not trace:
            continue
        final_x = trace[-1]
        recent = trace[:-1][-recent_window:]
        counts[recent.count(final_x)] += 1
    xs = list(range(0, max(counts) + 1))
    ys = [counts.get(x, 0) for x in xs]
    bars = ax.bar(xs, ys, color=COLORS["Neutral"], edgecolor="#333333", linewidth=0.8)
    ax.set_xlabel("Occurrences of final x in recent public answers", fontsize=16, labelpad=8)
    ax.set_ylabel("Failed runs (count)", fontsize=16, labelpad=10)
    ax.set_xticks(xs)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylim(0, max(ys) * 1.12)
    for bar, value in zip(bars, ys):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(ys) * 0.025,
            str(value),
            ha="center",
            va="bottom",
            fontsize=12,
        )
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("A. Polynomial knowing-doing gap", fontsize=18, pad=10)

    ax = axes[1]
    conds = ["C3", "C4", "C5", "C6", "C7"]
    xs = np.array([0.0, 1.4, 2.8, 4.2, 5.6])
    width = 0.32
    type_vals = [cyber_stats[c]["type_trans"] for c in conds]
    sev_vals = [cyber_stats[c]["severity_trans"] for c in conds]
    ax.bar(xs - width / 2, type_vals, width=width, color="white", edgecolor="#333333", linewidth=0.8, label="Type transitions")
    ax.bar(xs + width / 2, sev_vals, width=width, color=COLORS["Neutral"], edgecolor="#333333", linewidth=0.8, hatch="//", label="Severity transitions")
    ax.set_xticks(xs)
    ax.set_xticklabels(conds, fontsize=14)
    ax.set_xlim(-0.8, xs[-1] + 0.8)
    ax.set_ylim(0.0, 1.14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylabel("Mean transitions", fontsize=16, labelpad=10)
    ax.set_title("B. Cybersecurity judgment transitions", fontsize=18, pad=10)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        fontsize=20,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        handlelength=2.5,
        columnspacing=2.8,
    )
    save(fig, "cross_instability_modes")


def load_adv_runs(path: Path) -> List[Dict[str, Any]]:
    runs = list(load_json(path).get("runs") or [])
    for row in runs:
        row["family"] = model_family(str(row.get("model_mix") or ""))
    return runs


def target_steering_rate(rows: Sequence[Dict[str, Any]]) -> float:
    any_success_rows = [r for r in rows if bool(r.get("any_success"))]
    return rate(r.get("final_x") == -2 for r in any_success_rows)


def agent_outcome_mean(rows: Sequence[Dict[str, Any]], agent: str) -> Optional[float]:
    vals = []
    for row in rows:
        scores = row.get("scores_outcome") or row.get("scores") or {}
        if isinstance(scores, dict) and scores.get(agent) is not None:
            vals.append(float(scores[agent]))
    return mean(vals) if vals else None


def make_trust_risks(poly_runs: Sequence[Dict[str, Any]], cyber_records: Sequence[Dict[str, Any]], cyber_stats: Dict[str, Dict[str, float]]) -> None:
    baseline = [r for r in poly_runs if r["family"] in ("GPT-5", "Claude")]
    obstructive = [r for r in load_adv_runs(ADV_OBSTRUCTIVE_PATH) if r["family"] in ("GPT-5", "Claude")]
    targeted = [r for r in load_adv_runs(ADV_TARGETED_PATH) if r["family"] in ("GPT-5", "Claude")]

    baseline_any = rate(r.get("any_success") for r in baseline)
    obstructive_any = rate(r.get("any_success") for r in obstructive)
    obstructive_drop = baseline_any - obstructive_any
    # Match the thesis targeted-utility plot: compare Claude cooperative agents before/after targeted steering.
    coop_agents = ["Analyst A", "Builder B", "Critic C"]
    claude_baseline = [r for r in poly_runs if r["family"] == "Claude"]
    claude_targeted = [r for r in targeted if r["family"] == "Claude"]
    baseline_agent_means = [agent_outcome_mean(claude_baseline, agent) for agent in coop_agents]
    targeted_agent_means = [agent_outcome_mean(claude_targeted, agent) for agent in coop_agents]
    baseline_coop_utility = mean(v for v in baseline_agent_means if v is not None)
    targeted_coop_utility = mean(v for v in targeted_agent_means if v is not None)
    targeted_utility_loss = (baseline_coop_utility - targeted_coop_utility) / baseline_coop_utility
    poly_vals = [obstructive_drop, targeted_utility_loss]

    committee_conds = ["C3", "C4", "C5", "C6", "C7"]
    wrong = mean(cyber_stats[c]["wrong"] for c in committee_conds)
    under = mean(cyber_stats[c]["under"] for c in committee_conds)
    cyber_vals = [wrong, under]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.8, 9.2),
        dpi=300,
        sharey=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    ax = axes[0]
    labels = ["Obstructive\nsuccess drop", "Targeted\nutility loss"]
    bars = ax.bar(np.arange(2), poly_vals, color=COLORS["Neutral"], edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylabel("Risk rate / effect size", fontsize=16, labelpad=10)
    ax.set_title("A. Polynomial trust risks", fontsize=18, pad=10)
    setup_rate_axis(ax)
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.02, pct_label(y), ha="center", va="bottom", fontsize=13)

    ax = axes[1]
    labels = ["Wrong\nconsensus", "Under-severity"]
    bars = ax.bar(np.arange(2), cyber_vals, color=COLORS["Neutral"], edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels, fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylabel("Risk rate / effect size", fontsize=16, labelpad=10)
    ax.set_title("B. Cybersecurity trust risks", fontsize=18, pad=10)
    setup_rate_axis(ax)
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.02, pct_label(y), ha="center", va="bottom", fontsize=13)
    save(fig, "cross_trust_risks")


def make_benefit_vs_correctness(poly_dyn: Sequence[Dict[str, Any]], cyber_stats: Dict[str, Dict[str, float]]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.7),
        dpi=300,
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )

    traces = [r.get("collective_trace") or [] for r in poly_dyn if r.get("collective_trace")]
    max_len = max(len(t) for t in traces)
    xs = np.arange(max_len)
    means: List[float] = []
    q25: List[float] = []
    q75: List[float] = []
    for idx in range(max_len):
        vals = [float(t[idx]) for t in traces if idx < len(t)]
        means.append(float(np.mean(vals)))
        q25.append(float(np.quantile(vals, 0.25)))
        q75.append(float(np.quantile(vals, 0.75)))
    ax = axes[0]
    ax.fill_between(xs, q25, q75, color="#D9DEE7", alpha=0.9, linewidth=0)
    ax.plot(xs, means, color=COLORS["Neutral"], linewidth=2.2)
    ax.set_xlabel("Round", fontsize=13, labelpad=6)
    ax.set_ylabel("Collective score", fontsize=13, labelpad=8)
    ax.set_title("A. Polynomial process improvement", fontsize=15, pad=10)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)

    ax = axes[1]
    groups = [("GPT-5", "C1", "C3"), ("Claude", "C2", "C4")]
    xs = np.arange(len(groups))
    width = 0.34
    single = [cyber_stats[c1]["exact"] for _, c1, _ in groups]
    committee = [cyber_stats[c3]["exact"] for _, _, c3 in groups]
    ax.bar(xs - width / 2, single, width=width, color=[COLORS[g[0]] for g in groups], edgecolor="#333333", linewidth=0.6, label="Single")
    ax.bar(xs + width / 2, committee, width=width, color=[COLORS[g[0]] for g in groups], edgecolor="#333333", linewidth=0.6, hatch="//", label="3-agent no-prior")
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=12)
    ax.set_ylabel("Exact correctness", fontsize=13, labelpad=8)
    ax.set_title("B. Cybersecurity:\nno exact-correctness gain from committees", fontsize=15, pad=8)
    setup_rate_axis(ax)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    save(fig, "cross_negotiation_benefit_vs_task_correctness")


def make_hypothesis_map() -> None:
    rows = [
        ("Agreement vs. correctness", "n/a", "Partially supported", "Cyber-only test"),
        ("Severity less stable than type", "n/a", "Supported", "Cyber-only severity result"),
        ("Over-severity tendency", "n/a", "Contradicted", "Cyber-only calibration result"),
        ("Prior-knowledge effect", "Supported", "Observed, but not beneficial", "Asymmetric transfer across frameworks"),
    ]
    cols = ["Hypothesis", "Polynomial", "Cybersecurity", "Cross-framework interpretation"]
    color_by_text = {
        "n/a": "#E7E6E6",
        "Supported": "#D9D9D9",
        "Partially supported": "#F2F2F2",
        "Contradicted": "#E7E6E6",
        "Observed, but not beneficial": "#F2F2F2",
        "Cyber-only test": "#F2F2F2",
        "Cyber-only severity result": "#F2F2F2",
        "Cyber-only calibration result": "#F2F2F2",
        "Asymmetric transfer across frameworks": "#F2F2F2",
    }
    fig, ax = plt.subplots(figsize=(9.6, 2.9), dpi=300)
    ax.axis("off")
    table = ax.table(
        cellText=[list(r) for r in rows],
        colLabels=cols,
        colWidths=[0.28, 0.17, 0.21, 0.34],
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        if row == 0:
            cell.set_facecolor("#E7E6E6")
            cell.set_text_props(weight="bold")
        elif col > 0:
            text = cell.get_text().get_text()
            cell.set_facecolor(color_by_text.get(text, "white"))
        elif col == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight="bold")
    save(fig, "cross_framework_hypothesis_map")


def main() -> None:
    poly_runs = load_poly_runs()
    poly_dyn = load_poly_dynamics()
    cyber_records = load_cyber_latest()
    cyber_stats = cyber_condition_stats(cyber_records)

    make_transfer_matrix()
    make_prior_effects(poly_runs, cyber_stats)
    make_mixed_model_transfer(poly_runs, cyber_stats)
    make_instability_modes(poly_dyn, cyber_stats)
    make_trust_risks(poly_runs, cyber_records, cyber_stats)
    make_benefit_vs_correctness(poly_dyn, cyber_stats)
    make_hypothesis_map()

    print("\nCyber condition values used:")
    for cond in sorted(cyber_stats, key=condition_sort_key):
        vals = cyber_stats[cond]
        print(
            f"{cond}: exact={vals['exact']:.3f}, type={vals['type']:.3f}, "
            f"severity={vals['severity']:.3f}, wrong={vals['wrong']:.3f}, "
            f"under={vals['under']:.3f}, bias={vals['bias']:+.3f}"
        )


if __name__ == "__main__":
    main()
