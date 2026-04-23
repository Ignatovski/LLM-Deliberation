from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from thesis_plot_style import apply_thesis_style


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "viewer" / "plots" / "thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = ROOT / "viewer" / "metrics_summary.json"
DYNAMICS_PATH = ROOT / "viewer" / "dynamics_summary.json"
LEAKAGE_EVAL_PATH = ROOT / "summarys" / "leakage" / "eval_all.json"
ADV_OBSTRUCTIVE_PATH = ROOT / "summarys" / "metrics_summary.adversarial_obstructive.json"
ADV_TARGETED_PATH = ROOT / "summarys" / "metrics_summary.adversarial_outcome_targeted.json"

apply_thesis_style(font_size=11, y_grid=True)

COLORS = {
    "GPT-5": "#0072B2",
    "Claude": "#D55E00",
    "Llama": "#009E73",
    "Mixed": "#CC79A7",
    "Neutral": "#4D4D4D",
    "LightBand": "#CBD5E1",
}

MODEL_MIX_ORDER = [
    ("gpt-5x4", "All GPT-5", "GPT-5"),
    ("claude-sonnet-4-5x4", "All Claude", "Claude"),
    ("Llama-3.3-70B-Instructx4", "All Llama", "Llama"),
    ("claude-sonnet-4-5x2+gpt-5x2", "Mixed\n2 GPT-5 + 2 Claude", "Mixed"),
    (
        "Llama-3.3-70B-Instruct+claude-sonnet-4-5+gpt-5x2",
        "Mixed\n2 GPT-5 + 1 Claude + 1 Llama",
        "Mixed",
    ),
]

PRIOR_ORDER = [("Uniform", "Uniform", "-"), ("All Human", "Human prior", "--"), ("All AI", "LLM prior", ":")]
FAMILY_ORDER = ["GPT-5", "Claude", "Llama", "Mixed"]
FAMILY_MARKERS = {"GPT-5": "s", "Claude": "s", "Llama": "s", "Mixed": "D"}
VARIANT_ORDER = [("poly_x-7", "-7"), ("poly_x0", "0"), ("poly_x7", "+7")]
AGENTS = ["Analyst A", "Builder B", "Critic C", "Delegate D"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(path: Path) -> list[dict[str, Any]]:
    return list(load_json(path).get("runs") or [])


def norm_path(value: str) -> str:
    return str(value).replace("\\", "/")


def rate(rows: Sequence[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if bool(row.get(key))) / len(rows) if rows else 0.0


def mean_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return mean(vals) if vals else 0.0


def model_family(model_mix: str) -> str:
    if model_mix == "gpt-5x4":
        return "GPT-5"
    if model_mix == "claude-sonnet-4-5x4":
        return "Claude"
    if model_mix == "Llama-3.3-70B-Instructx4":
        return "Llama"
    return "Mixed"


def model_display_name(family: str) -> str:
    return {
        "GPT-5": "GPT-5",
        "Claude": "Claude 4.5",
        "Llama": "Llama 3.3 70B",
        "Mixed": "Mixed",
    }[family]


def category_from_path(path: str) -> str:
    path = norm_path(path)
    if "/polynomial_game_all_AI/" in path:
        return "All AI"
    if "/polynomial_game_human/" in path:
        return "All Human"
    return "Uniform"


def config_from_leakage_path(path: str) -> str | None:
    path = norm_path(path)
    if "/output_mix_split/" in path:
        return "mix_2gpt_2claude"
    if "/output_mix_all_diff/" in path:
        return "mix_2gpt_1claude_1llama"
    if "/output_llama/" in path:
        return "all_llama"
    if "/output_claude/" in path:
        return "all_claude"
    if "/output/" in path and "/output_mix" not in path:
        return "all_gpt"
    return None


def config_display_name(config_key: str) -> tuple[str, str]:
    mapping = {
        "all_claude": ("All Claude", "Claude"),
        "all_gpt": ("All GPT-5", "GPT-5"),
        "all_llama": ("All Llama", "Llama"),
        "mix_2gpt_1claude_1llama": ("Mixed\n2 GPT-5 + 1 Claude + 1 Llama", "Mixed"),
        "mix_2gpt_2claude": ("Mixed\n2 GPT-5 + 2 Claude", "Mixed"),
    }
    return mapping[config_key]


def eval_model_from_path_and_agent(path: str, agent: str) -> str | None:
    config = config_from_leakage_path(path)
    if config == "all_gpt":
        return "GPT-5"
    if config == "all_claude":
        return "Claude"
    if config == "all_llama":
        return "Llama"
    if config == "mix_2gpt_2claude":
        return {"Analyst A": "GPT-5", "Builder B": "GPT-5", "Critic C": "Claude", "Delegate D": "Claude"}.get(agent)
    if config == "mix_2gpt_1claude_1llama":
        return {"Analyst A": "GPT-5", "Builder B": "Claude", "Critic C": "Llama", "Delegate D": "GPT-5"}.get(agent)
    return None


def save_plot(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{Path(name).stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_percent_labels(ax: plt.Axes, bars: Sequence[Any], values: Sequence[float], *, dy: float = 1.5) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def add_count_labels(ax: plt.Axes, bars: Sequence[Any], values: Sequence[int], *, dy: float = 10.0) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def render_table_png(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    output_name: str,
    *,
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    save_plot(fig, output_name)


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def make_success_by_config(runs: Sequence[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[str(row.get("model_mix"))].append(row)

    labels = [label for _, label, _ in MODEL_MIX_ORDER]
    any_rates = [rate(grouped[mix], "any_success") * 100 for mix, _, _ in MODEL_MIX_ORDER]
    final_rates = [rate(grouped[mix], "final_success") * 100 for mix, _, _ in MODEL_MIX_ORDER]

    fig, ax = plt.subplots(figsize=(10.2, 5.4), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.34
    any_bars = []
    final_bars = []
    for idx, (_, _, family) in enumerate(MODEL_MIX_ORDER):
        color = COLORS[family]
        any_bar = ax.bar(
            x[idx] - width / 2,
            any_rates[idx],
            width=width,
            color=color,
            alpha=0.35,
            edgecolor=color,
            linewidth=1.0,
        )
        final_bar = ax.bar(
            x[idx] + width / 2,
            final_rates[idx],
            width=width,
            color="white",
            edgecolor=color,
            linewidth=1.4,
            hatch="///",
        )
        any_bars.extend(any_bar)
        final_bars.extend(final_bar)

    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 92)
    ax.legend(
        handles=[
            Patch(facecolor="#D9D9D9", edgecolor="#666666", label="Any success"),
            Patch(facecolor="white", edgecolor="#666666", hatch="///", label="Final success"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    add_percent_labels(ax, any_bars, any_rates)
    add_percent_labels(ax, final_bars, final_rates)
    save_plot(fig, "success_by_config.png")


def make_success_by_init(runs: Sequence[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[str(row.get("variant"))].append(row)

    labels = [label for _, label in VARIANT_ORDER]
    any_rates = [rate(grouped[variant], "any_success") * 100 for variant, _ in VARIANT_ORDER]
    final_rates = [rate(grouped[variant], "final_success") * 100 for variant, _ in VARIANT_ORDER]

    fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.34
    any_bars = ax.bar(
        x - width / 2,
        any_rates,
        width=width,
        color="#D9D9D9",
        edgecolor=COLORS["Neutral"],
        linewidth=1.0,
        label="Any success",
    )
    final_bars = ax.bar(
        x + width / 2,
        final_rates,
        width=width,
        color="white",
        edgecolor=COLORS["Neutral"],
        linewidth=1.2,
        hatch="///",
        label="Final success",
    )
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    add_percent_labels(ax, any_bars, any_rates)
    add_percent_labels(ax, final_bars, final_rates)
    save_plot(fig, "success_by_init.png")


def make_prior_effect_plot(runs: Sequence[dict[str, Any]], metric_key: str, ylabel: str, output_name: str) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(model_family(str(row.get("model_mix"))), str(row.get("category")))].append(row)

    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    x = np.arange(len(FAMILY_ORDER))

    for prior, display, linestyle in PRIOR_ORDER:
        y = [rate(grouped[(family, prior)], metric_key) for family in FAMILY_ORDER]
        ax.plot(x, y, color="#6B7280", linestyle=linestyle, linewidth=1.5, zorder=1)
        for xi, family, yi in zip(x, FAMILY_ORDER, y):
            ax.scatter(
                xi,
                yi,
                s=55,
                marker=FAMILY_MARKERS[family],
                color=COLORS[family],
                edgecolor="#222222",
                linewidth=0.6,
                zorder=2,
            )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x, FAMILY_ORDER)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 5))

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            color="none",
            markerfacecolor=COLORS[family],
            markeredgecolor="#222222",
            markersize=7,
            label=family,
        )
        for family in FAMILY_ORDER
    ]
    prior_handles = [
        Line2D([0], [0], color="#6B7280", linestyle=linestyle, linewidth=1.5, label=display)
        for _, display, linestyle in PRIOR_ORDER
    ]
    legend_models = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    ax.add_artist(legend_models)
    ax.legend(
        handles=prior_handles,
        title="Prior",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.62),
        frameon=False,
    )
    save_plot(fig, output_name)


def make_final_x_by_anchor(runs: Sequence[dict[str, Any]]) -> None:
    values = [[float(row.get("final_x")) for row in runs if row.get("variant") == variant] for variant, _ in VARIANT_ORDER]
    labels = [variant for variant, _ in VARIANT_ORDER]
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    bp = ax.boxplot(values, patch_artist=True, tick_labels=labels)
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor(COLORS["Neutral"])
        patch.set_linewidth(1.1)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set_color(COLORS["Neutral"])
            artist.set_linewidth(1.1)
    for flier in bp["fliers"]:
        flier.set_markeredgecolor(COLORS["Neutral"])
        flier.set_markerfacecolor("white")
    ax.set_ylabel("Final x")
    save_plot(fig, "polynomial_final_x_by_anchor_all.png")


def make_collective_band_plot(
    runs: Sequence[dict[str, Any]],
    output_name: str,
) -> None:
    max_len = max(len(row.get("collective_trace") or []) for row in runs)
    arr = np.array(
        [
            list(row.get("collective_trace") or []) + [np.nan] * (max_len - len(row.get("collective_trace") or []))
            for row in runs
        ],
        dtype=float,
    )
    lo = np.nanpercentile(arr, 25, axis=0)
    hi = np.nanpercentile(arr, 75, axis=0)
    rounds = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.fill_between(rounds, lo, hi, color=COLORS["Neutral"], alpha=0.22, linewidth=0)
    ax.set_xlabel("Round")
    ax.set_ylabel("Collective score")
    ax.set_xlim(rounds.min(), rounds.max())
    ax.set_ylim(10, 56)
    save_plot(fig, output_name)


def make_knowing_doing_gap_plot(dynamics_runs: Sequence[dict[str, Any]]) -> None:
    recent_window = 6
    counts: Counter[int] = Counter()
    for row in dynamics_runs:
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
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    bars = ax.bar(xs, ys, color=COLORS["Neutral"], edgecolor="#333333", linewidth=0.8)
    ax.set_xlabel("Occurrences of final x in recent public answers")
    ax.set_ylabel("Failed runs (count)")
    ax.set_xticks(xs)
    add_count_labels(ax, bars, ys, dy=6.0)
    save_plot(fig, "knowing_doing_gap_recent_repeat_distribution.png")


def make_unclosed_bracket_plots(leakage_rows: Sequence[dict[str, Any]]) -> None:
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for row in leakage_rows:
        if not row.get("leakage_due_to_unclosed_answer"):
            continue
        path = norm_path(row.get("path", ""))
        config = config_from_leakage_path(path)
        if config not in {"all_gpt", "all_claude", "all_llama"}:
            continue
        family = eval_model_from_path_and_agent(path, str(row.get("agent", "")))
        if not family:
            continue
        by_category[category_from_path(path)][family] += 1

    family_labels = ["GPT-5", "Claude", "Llama"]
    family_colors = [COLORS["GPT-5"], COLORS["Claude"], COLORS["Llama"]]

    all_counts = [sum(by_category[cat][family] for cat in ("Uniform", "All AI", "All Human")) for family in family_labels]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    bars = ax.bar(family_labels, all_counts, color=family_colors, edgecolor="#333333", linewidth=0.8)
    ax.set_ylabel("Unclosed answers (count)")
    ax.text(0.98, 1.04, f"Unclosed {sum(all_counts)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
    add_count_labels(ax, bars, all_counts, dy=25.0)
    save_plot(fig, "unclosed_brackets_all_categories.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharey=True, constrained_layout=True)
    for ax, category in zip(axes, ["Uniform", "All AI", "All Human"]):
        values = [by_category[category][family] for family in family_labels]
        bars = ax.bar(family_labels, values, color=family_colors, edgecolor="#333333", linewidth=0.8)
        ax.set_title(category)
        ax.text(0.98, 1.04, f"Unclosed {sum(values)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Unclosed answers (count)")
        add_count_labels(ax, bars, values, dy=10.0)
    save_plot(fig, "unclosed_brackets_by_category.png")


def make_llama_performance_tables(runs: Sequence[dict[str, Any]]) -> None:
    homogeneous = [row for row in runs if str(row.get("model_mix")) in {"gpt-5x4", "claude-sonnet-4-5x4", "Llama-3.3-70B-Instructx4"}]

    overall_headers = ["Model", "Any success (n)", "Any success (%)", "Final success (n)", "Final success (%)", "Runs (n)"]
    overall_rows_csv: list[list[str]] = []
    overall_rows_png: list[list[str]] = []
    for mix, family in [("gpt-5x4", "GPT-5"), ("claude-sonnet-4-5x4", "Claude"), ("Llama-3.3-70B-Instructx4", "Llama")]:
        sel = [row for row in homogeneous if row.get("model_mix") == mix]
        any_n = sum(1 for row in sel if row.get("any_success"))
        final_n = sum(1 for row in sel if row.get("final_success"))
        total = len(sel)
        csv_row = [
            model_display_name(family),
            str(any_n),
            f"{(100 * any_n / total):.1f}%",
            str(final_n),
            f"{(100 * final_n / total):.1f}%",
            str(total),
        ]
        overall_rows_csv.append(csv_row)
        overall_rows_png.append(csv_row)

    write_csv(OUT_DIR / "llama_performance_overall_table.csv", overall_headers, overall_rows_csv)
    render_table_png(overall_headers, overall_rows_png, "llama_performance_overall_table.png", figsize=(8.5, 2.3))

    by_cat_headers = [
        "Category",
        "Model",
        "Any success (%)",
        "Final success (%)",
        "Mean collective outcome",
        "Runs (n)",
    ]
    by_cat_headers_png = [
        "Category",
        "Model",
        "Any success\n(%)",
        "Final success\n(%)",
        "Mean collective\noutcome",
        "Runs\n(n)",
    ]
    by_cat_rows_csv: list[list[str]] = []
    by_cat_rows_png: list[list[str]] = []
    category_order = ["Uniform", "All AI", "All Human"]
    for category in category_order:
        for mix, family in [("gpt-5x4", "GPT-5"), ("claude-sonnet-4-5x4", "Claude"), ("Llama-3.3-70B-Instructx4", "Llama")]:
            sel = [row for row in homogeneous if row.get("model_mix") == mix and row.get("category") == category]
            any_rate = 100 * rate(sel, "any_success")
            final_rate = 100 * rate(sel, "final_success")
            mean_collective_outcome = mean_or_zero(float(row.get("collective_outcome") or 0.0) for row in sel)
            csv_row = [
                category,
                model_display_name(family),
                f"{any_rate:.1f}%",
                f"{final_rate:.1f}%",
                f"{mean_collective_outcome:.2f}",
                str(len(sel)),
            ]
            by_cat_rows_csv.append(csv_row)
            by_cat_rows_png.append(csv_row)

    write_csv(OUT_DIR / "llama_performance_by_category_table.csv", by_cat_headers, by_cat_rows_csv)
    render_table_png(by_cat_headers_png, by_cat_rows_png, "llama_performance_by_category_table.png", figsize=(10.8, 3.4))


def agent_outcome_mean(rows: Sequence[dict[str, Any]], agent: str) -> float:
    return mean_or_zero(float((row.get("scores_outcome") or {}).get(agent, 0.0)) for row in rows)


def make_adversarial_success_figures(
    baseline_runs: Sequence[dict[str, Any]],
    obstructive_runs: Sequence[dict[str, Any]],
    targeted_runs: Sequence[dict[str, Any]],
) -> None:
    conditions = [
        ("Baseline", baseline_runs),
        ("Obstructive", obstructive_runs),
        ("Outcome targeted", targeted_runs),
    ]

    table_headers = ["Condition", "Model", "Any success (n)", "Total runs (n)", "Any success (%)"]
    table_rows: list[list[str]] = []
    line_values: dict[str, list[float]] = {"GPT-5": [], "Claude": []}

    for condition_name, rows in conditions:
        for family in ("GPT-5", "Claude"):
            sel = [row for row in rows if model_family(str(row.get("model_mix"))) == family]
            any_n = sum(1 for row in sel if row.get("any_success"))
            total = len(sel)
            pct = 100 * any_n / total if total else 0.0
            table_rows.append([condition_name, family, str(any_n), str(total), f"{pct:.1f}%"])
            line_values[family].append(pct / 100.0)

    write_csv(OUT_DIR / "adversarial_any_success_rates_table.csv", table_headers, table_rows)
    render_table_png(
        table_headers,
        table_rows,
        "adversarial_any_success_rates_table.png",
        figsize=(8.2, 2.6),
    )

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for family in ("GPT-5", "Claude"):
        ax.plot(
            x,
            line_values[family],
            marker="s",
            markersize=8,
            linewidth=1.8,
            color=COLORS[family],
            label=family,
        )
    ax.set_xticks(x, [name for name, _ in conditions])
    ax.set_ylabel("Any success rate")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save_plot(fig, "adversarial_any_success_rates.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for family in ("GPT-5", "Claude"):
        ax.plot(
            x[:2],
            line_values[family][:2],
            marker="s",
            markersize=8,
            linewidth=1.8,
            color=COLORS[family],
            label=family,
        )
    ax.set_xticks(x[:2], [name for name, _ in conditions[:2]])
    ax.set_ylabel("Any success rate")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save_plot(fig, "adversarial_any_success_rates_baseline_obstructive.png")


def make_adversarial_utility_figures(
    baseline_runs: Sequence[dict[str, Any]],
    obstructive_runs: Sequence[dict[str, Any]],
    targeted_runs: Sequence[dict[str, Any]],
) -> None:
    condition_specs = [
        ("Baseline", baseline_runs),
        ("Obstructive", obstructive_runs),
        ("Outcome targeted", targeted_runs),
    ]
    families = ["GPT-5", "Claude"]

    values: dict[tuple[str, str], list[float]] = {}
    y_min = 0.0
    y_max = 0.0
    for family in families:
        for condition_name, rows in condition_specs:
            sel = [row for row in rows if model_family(str(row.get("model_mix"))) == family]
            vals = [agent_outcome_mean(sel, agent) for agent in AGENTS]
            values[(family, condition_name)] = vals
            y_min = min(y_min, min(vals))
            y_max = max(y_max, max(vals))

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharey=True, constrained_layout=True)
    for row_idx, family in enumerate(families):
        for col_idx, (condition_name, _) in enumerate(condition_specs):
            ax = axes[row_idx, col_idx]
            vals = values[(family, condition_name)]
            ax.bar(AGENTS, vals, color=COLORS[family], alpha=0.82, edgecolor="#333333", linewidth=0.8)
            ax.axhline(0, color="#666666", linewidth=0.8)
            if row_idx == 0:
                ax.set_title(condition_name)
            if col_idx == 0:
                ax.set_ylabel("Outcome utility")
            ax.tick_params(axis="x", rotation=0)
            ax.set_ylim(y_min - 0.4, y_max + 0.5)
    fig.text(0.015, 0.74, "GPT-5", color=COLORS["GPT-5"], fontsize=13, fontweight="bold", va="center")
    fig.text(0.015, 0.24, "Claude", color=COLORS["Claude"], fontsize=13, fontweight="bold", va="center")
    save_plot(fig, "adversarial_outcome_utility_by_agent.png")

    claude_baseline = values[("Claude", "Baseline")]
    claude_targeted = values[("Claude", "Outcome targeted")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, constrained_layout=True)
    for ax, title, vals in zip(axes, ["Baseline", "Outcome targeted"], [claude_baseline, claude_targeted]):
        ax.bar(AGENTS, vals, color=COLORS["Claude"], alpha=0.82, edgecolor="#333333", linewidth=0.8)
        ax.axhline(0, color="#666666", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=0)
    axes[0].set_ylabel("Outcome utility")
    axes[0].set_ylim(y_min - 0.4, y_max + 0.5)
    save_plot(fig, "adversarial_outcome_utility_claude_baseline_vs_targeted.png")


def make_leakage_by_config_plot(leakage_rows: Sequence[dict[str, Any]]) -> None:
    run_flags: dict[str, dict[str, Any]] = {}
    for row in leakage_rows:
        path = norm_path(row.get("path", ""))
        config = config_from_leakage_path(path)
        if not config:
            continue
        rec = run_flags.setdefault(path, {"config": config, "llm_leak": False})
        rec["llm_leak"] = rec["llm_leak"] or bool(row.get("llm_leak"))

    grouped: dict[str, list[bool]] = defaultdict(list)
    for record in run_flags.values():
        grouped[str(record["config"])].append(bool(record["llm_leak"]))

    config_order = ["all_gpt", "all_claude", "all_llama", "mix_2gpt_2claude", "mix_2gpt_1claude_1llama"]
    labels = [config_display_name(config)[0] for config in config_order]
    families = [config_display_name(config)[1] for config in config_order]
    rates = [100 * (sum(grouped[config]) / len(grouped[config])) if grouped[config] else 0.0 for config in config_order]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    bars = ax.bar(
        labels,
        rates,
        color=[COLORS[family] for family in families],
        alpha=0.82,
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.set_ylabel("Leakage rate (%)")
    ax.set_ylim(0, 105)
    add_percent_labels(ax, bars, rates, dy=2.0)
    save_plot(fig, "leakage_by_config.png")


def main() -> None:
    baseline_runs = load_runs(METRICS_PATH)
    dynamics_runs = load_json(DYNAMICS_PATH).get("runs") or []
    leakage_rows = load_json(LEAKAGE_EVAL_PATH)
    obstructive_runs = load_runs(ADV_OBSTRUCTIVE_PATH)
    targeted_runs = load_runs(ADV_TARGETED_PATH)

    make_success_by_config(baseline_runs)
    make_success_by_init(baseline_runs)
    make_prior_effect_plot(baseline_runs, "final_success", "Final success rate", "prior_effect_final_success.png")
    make_prior_effect_plot(baseline_runs, "any_success", "Any success rate", "prior_effect_any_success.png")
    make_final_x_by_anchor(baseline_runs)
    make_collective_band_plot(dynamics_runs, "mean_collective_score_over_time_dynamics_band_only.png")
    make_collective_band_plot(
        [row for row in dynamics_runs if model_family(str(row.get("model_mix"))) in {"GPT-5", "Claude"}],
        "mean_collective_score_over_time_dynamics_band_only_gpt_claude.png",
    )
    make_knowing_doing_gap_plot(dynamics_runs)
    make_unclosed_bracket_plots(leakage_rows)
    make_llama_performance_tables(baseline_runs)
    make_adversarial_success_figures(baseline_runs, obstructive_runs, targeted_runs)
    make_adversarial_utility_figures(baseline_runs, obstructive_runs, targeted_runs)
    make_leakage_by_config_plot(leakage_rows)
    print(f"Regenerated thesis root plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
