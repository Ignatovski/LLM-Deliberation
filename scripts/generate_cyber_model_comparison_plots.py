from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis_plot_style import apply_thesis_style


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT / "cyber_negotiation_v1" / "games_descriptions" / "cyber_game" / "output"
DEFAULT_GROUND_TRUTH_DIR = ROOT / "cyber_negotiation_v1" / "games_descriptions" / "cyber_game" / "ground_truth"
DEFAULT_PLOTS_DIR = ROOT / "viewer" / "plots" / "thesis" / "cybersecurity" / "model_comparison"

MODEL_COLORS = {
    "GPT-5": "#0072B2",
    "Claude": "#D55E00",
    "Kimi": "#009E73",
}

EDGE_COLOR = "#333333"
GRID_COLOR = "#D0D0D0"

CONDITION_META: Dict[str, Dict[str, str]] = {
    "C1": {"model": "GPT-5", "setup": "single", "label": "Single GPT-5"},
    "C2": {"model": "Claude", "setup": "single", "label": "Single Claude"},
    "C3": {"model": "GPT-5", "setup": "committee", "label": "3x GPT-5"},
    "C4": {"model": "Claude", "setup": "committee", "label": "3x Claude"},
    "C1.1": {"model": "Kimi", "setup": "single", "label": "Single Kimi"},
    "C3.1": {"model": "Kimi", "setup": "committee", "label": "3x Kimi"},
}

MODEL_ORDER = ["GPT-5", "Claude", "Kimi"]
SETUP_ORDER = ["single", "committee"]
SETUP_TITLES = {"single": "Single-Agent", "committee": "Three-Agent Committee"}

SCENARIO_ALIASES = {
    "ping_form_exec_output_001": "command_injection_ping_001",
    "command_injection_ping_001": "command_injection_ping_001",
    "cookie_security_attribute_observation_001": "cookie_security_attribute_observation_001",
    "hard_cookie_md5_002": "hard_cookie_md5_002",
    "medium_cookie_timestamps_001": "medium_cookie_timestamps_001",
    "reflected_input_password_change_guard_001": "reflected_input_password_change_guard_001",
    "info_apache": "info_apache",
    "error_message_path_disclosure_001": "error_message_path_disclosure_001",
}

SCENARIO_ORDER = [
    "command_injection_ping_001",
    "cookie_security_attribute_observation_001",
    "hard_cookie_md5_002",
    "medium_cookie_timestamps_001",
    "reflected_input_password_change_guard_001",
    "info_apache",
    "error_message_path_disclosure_001",
]

SCENARIO_TITLES = {
    "command_injection_ping_001": "Ping Form Command Injection",
    "cookie_security_attribute_observation_001": "Cookie Flags Observation",
    "hard_cookie_md5_002": "MD5-Like Cookie Pattern",
    "medium_cookie_timestamps_001": "Timestamp Cookie Pattern",
    "reflected_input_password_change_guard_001": "Reflected Input Password Change Guard",
    "info_apache": "Apache Header Disclosure",
    "error_message_path_disclosure_001": "Error Message Path Disclosure",
}

SEVERITY_ORDER = {
    "Info": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


@dataclass
class RunEntry:
    condition_id: str
    model: str
    setup: str
    scenario_key: str
    run_dir: Path
    exact: float
    finding_type: float
    severity_correct: Optional[float]
    under_severity: Optional[float]
    over_severity: Optional[float]
    wrong: float
    severity_bias: Optional[float]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_scenario_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return SCENARIO_ALIASES.get(str(value), str(value))


def load_ground_truth_map(ground_truth_dir: Path) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for path in sorted(ground_truth_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        scenario_key = canonical_scenario_id(payload.get("scenario_id")) or canonical_scenario_id(path.stem)
        if not scenario_key:
            continue
        mapping[scenario_key] = payload
        mapping[canonical_scenario_id(path.stem) or path.stem] = payload
    return mapping


def latest_completed_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = sorted(run_dir.glob("metrics_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        run_report = payload.get("run_report") or {}
        appendix = run_report.get("appendix_debug") or {}
        is_complete = bool(appendix.get("RunCompleted")) or str(appendix.get("RunStatus") or "").lower() == "completed"
        if is_complete:
            payload["_metrics_path"] = str(path)
            return payload
    return None


def committee_severity_value(run_report: Dict[str, Any]) -> str:
    committee_final = run_report.get("committee_final") or {}
    exact = committee_final.get("committee_exact")
    if isinstance(exact, dict):
        severity = str(exact.get("severity", "")).strip()
        if severity:
            return severity
    severity = committee_final.get("committee_exact_severity")
    return str(severity).strip() if severity else ""


def scan_completed_runs(output_root: Path, ground_truth_dir: Path) -> List[RunEntry]:
    ground_truth = load_ground_truth_map(ground_truth_dir)
    run_dirs = {path.parent for path in output_root.rglob("metrics_*.json")}
    runs: List[RunEntry] = []

    for run_dir in sorted(run_dirs):
        payload = latest_completed_metrics(run_dir)
        if payload is None:
            continue

        run_report = payload.get("run_report") or {}
        condition_id = str(run_report.get("condition_id") or "").strip()
        if condition_id not in CONDITION_META:
            continue

        scenario_key = canonical_scenario_id(run_report.get("scenario_id")) or canonical_scenario_id(run_dir.name)
        if not scenario_key or scenario_key not in SCENARIO_TITLES:
            continue

        meta = CONDITION_META[condition_id]
        headline = run_report.get("headline_metrics") or {}
        derived = (payload.get("condition_aggregate") or {}).get("derived_metrics") or {}
        gt = ground_truth.get(scenario_key, {})

        final_severity = committee_severity_value(run_report)
        gt_severity = str(gt.get("final_severity") or "").strip()
        final_num = SEVERITY_ORDER.get(final_severity)
        gt_num = SEVERITY_ORDER.get(gt_severity)
        severity_correct = None
        under = None
        over = None
        if final_num is not None and gt_num is not None:
            severity_correct = 1.0 if final_num == gt_num else 0.0
            under = 1.0 if final_num < gt_num else 0.0
            over = 1.0 if final_num > gt_num else 0.0

        runs.append(
            RunEntry(
                condition_id=condition_id,
                model=meta["model"],
                setup=meta["setup"],
                scenario_key=scenario_key,
                run_dir=run_dir,
                exact=float(headline.get("FinalCorrectExact", 0.0)),
                finding_type=float(headline.get("FinalCorrectType", 0.0)),
                severity_correct=severity_correct,
                under_severity=under if under is not None else safe_float(derived.get("UnderSeverityRate")),
                over_severity=over if over is not None else safe_float(derived.get("OverSeverityRate")),
                wrong=float((run_report.get("derived_metrics") or {}).get("WrongConsensusExact", 0.0)),
                severity_bias=safe_float(headline.get("SeverityBias")),
            )
        )

    return runs


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_optional(values: Iterable[Optional[float]]) -> float:
    numeric = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not numeric:
        return float("nan")
    return float(sum(numeric) / len(numeric))


def fmt_pct(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_float(value: float, digits: int = 2) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def setup_ax(ax, ylim: tuple[float, float] = (0.0, 1.0)) -> None:
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, out_dir: Path, name: str, generated: List[Path], *, use_tight_layout: bool = True) -> None:
    path = out_dir / f"{name}.png"
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)


def aggregate_model_setup(runs: Sequence[RunEntry]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for setup in SETUP_ORDER:
        for model in MODEL_ORDER:
            items = [run for run in runs if run.setup == setup and run.model == model]
            grouped[setup][model] = {
                "exact": mean_optional(run.exact for run in items),
                "type": mean_optional(run.finding_type for run in items),
                "severity": mean_optional(run.severity_correct for run in items),
                "under": mean_optional(run.under_severity for run in items),
                "over": mean_optional(run.over_severity for run in items),
                "wrong": mean_optional(run.wrong for run in items),
                "severity_bias": mean_optional(run.severity_bias for run in items),
            }
    return grouped


def plot_accuracy_by_setup(stats: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path, generated: List[Path]) -> None:
    metrics = ["exact", "type", "severity"]
    labels = ["Exact", "Type", "Severity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=300, sharey=True)

    for ax, setup in zip(axes, SETUP_ORDER):
        xs = np.arange(len(metrics))
        width = 0.22
        offsets = np.linspace(-width, width, len(MODEL_ORDER))
        for offset, model in zip(offsets, MODEL_ORDER):
            values = [stats[setup][model][metric] for metric in metrics]
            ax.bar(
                xs + offset,
                values,
                width=width,
                color=MODEL_COLORS[model],
                edgecolor=EDGE_COLOR,
                linewidth=0.6,
                label=model,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(SETUP_TITLES[setup], fontsize=10, pad=4)
        setup_ax(ax, (0.0, 1.0))
    axes[0].set_ylabel("Rate")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "cyber_model_accuracy_by_setup", generated)


def plot_over_under_by_setup(stats: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path, generated: List[Path]) -> None:
    metrics = ["under", "over"]
    labels = ["Under-severity", "Over-severity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=300, sharey=True)

    for ax, setup in zip(axes, SETUP_ORDER):
        xs = np.arange(len(metrics))
        width = 0.22
        offsets = np.linspace(-width, width, len(MODEL_ORDER))
        for offset, model in zip(offsets, MODEL_ORDER):
            values = [stats[setup][model][metric] for metric in metrics]
            ax.bar(
                xs + offset,
                values,
                width=width,
                color=MODEL_COLORS[model],
                edgecolor=EDGE_COLOR,
                linewidth=0.6,
                label=model,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(SETUP_TITLES[setup], fontsize=10, pad=4)
        setup_ax(ax, (0.0, 1.0))
    axes[0].set_ylabel("Rate")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "cyber_model_over_under_by_setup", generated)


def plot_committee_gain(stats: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path, generated: List[Path]) -> None:
    perf_metrics = ["exact", "type", "severity"]
    perf_labels = ["Exact", "Type", "Severity"]
    calib_metrics = ["under", "over"]
    calib_labels = ["Under", "Over"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), dpi=300)

    xs = np.arange(len(MODEL_ORDER))
    width = 0.22
    perf_offsets = np.linspace(-width, width, len(perf_metrics))
    for offset, metric, label in zip(perf_offsets, perf_metrics, perf_labels):
        values = [stats["committee"][model][metric] - stats["single"][model][metric] for model in MODEL_ORDER]
        axes[0].bar(xs + offset, values, width=width, edgecolor=EDGE_COLOR, linewidth=0.6, label=label)
    axes[0].axhline(0, color="#222222", linewidth=1.2)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(MODEL_ORDER)
    axes[0].set_title("Committee Minus Single\nAccuracy", fontsize=10, pad=4)
    axes[0].set_ylabel("Delta")
    axes[0].grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axes[0].set_axisbelow(True)

    calib_offsets = np.linspace(-width / 2, width / 2, len(calib_metrics))
    for offset, metric, label in zip(calib_offsets, calib_metrics, calib_labels):
        values = [stats["committee"][model][metric] - stats["single"][model][metric] for model in MODEL_ORDER]
        axes[1].bar(xs + offset, values, width=width, edgecolor=EDGE_COLOR, linewidth=0.6, label=label)
    axes[1].axhline(0, color="#222222", linewidth=1.2)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(MODEL_ORDER)
    axes[1].set_title("Committee Minus Single\nCalibration", fontsize=10, pad=4)
    axes[1].grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axes[1].set_axisbelow(True)

    color_cycle = ["#777777", "#BBBBBB", "#444444"]
    for index, bar in enumerate(axes[0].patches):
        bar.set_facecolor(color_cycle[index % len(perf_metrics)])
    for index, bar in enumerate(axes[1].patches):
        bar.set_facecolor(color_cycle[index % len(calib_metrics)])

    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "cyber_model_committee_gain", generated)


def build_heatmap_matrix(runs: Sequence[RunEntry], setup: str) -> np.ndarray:
    matrix = np.full((len(SCENARIO_ORDER), len(MODEL_ORDER)), np.nan, dtype=float)
    for row_index, scenario_key in enumerate(SCENARIO_ORDER):
        for col_index, model in enumerate(MODEL_ORDER):
            matches = [run.exact for run in runs if run.setup == setup and run.model == model and run.scenario_key == scenario_key]
            matrix[row_index, col_index] = mean_optional(matches)
    return matrix


def plot_scenario_exact_heatmap(runs: Sequence[RunEntry], out_dir: Path, generated: List[Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.8), dpi=300, sharey=True)
    matrices = [build_heatmap_matrix(runs, setup) for setup in SETUP_ORDER]

    for ax, setup, matrix in zip(axes, SETUP_ORDER, matrices):
        image = ax.imshow(matrix, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_yticks(np.arange(len(SCENARIO_ORDER)))
        ax.set_yticklabels([SCENARIO_TITLES[key] for key in SCENARIO_ORDER], fontsize=8)
        ax.set_title(f"{SETUP_TITLES[setup]}\nExact correctness", fontsize=10, pad=4)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if math.isnan(value):
                    text = "n/a"
                    color = "black"
                else:
                    text = fmt_pct(value)
                    color = "white" if value < 0.5 else "black"
                ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8, color=color)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label("Rate", fontsize=8)
    fig.subplots_adjust(wspace=0.28, right=0.88)
    save(fig, out_dir, "cyber_model_scenario_exact_heatmap", generated, use_tight_layout=False)


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]], out_dir: Path, name: str, generated: List[Path]) -> None:
    fig_w = 8.8
    fig_h = (len(rows) + 1) * 0.36 + 0.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    table = ax.table(cellText=[list(headers), *[list(row) for row in rows]], cellLoc="center", loc="center")

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#F2F5F8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    save(fig, out_dir, name, generated)


def write_summary_table(stats: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path, generated: List[Path]) -> None:
    headers = ["Setup", "Model", "Exact", "Type", "Severity", "Under", "Over", "Wrong", "Severity Bias"]
    rows: List[List[str]] = []
    for setup in SETUP_ORDER:
        for model in MODEL_ORDER:
            row = stats[setup][model]
            rows.append(
                [
                    SETUP_TITLES[setup],
                    model,
                    fmt_pct(row["exact"]),
                    fmt_pct(row["type"]),
                    fmt_pct(row["severity"]),
                    fmt_pct(row["under"]),
                    fmt_pct(row["over"]),
                    fmt_pct(row["wrong"]),
                    fmt_float(row["severity_bias"]),
                ]
            )
    render_table(headers, rows, out_dir, "cyber_model_summary_table", generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GPT-5 vs Claude vs Kimi cybersecurity comparison plots.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Cyber output directory.")
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR, help="Ground-truth directory.")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR, help="Destination for comparison PNGs.")
    args = parser.parse_args()

    apply_thesis_style(font_size=11, y_grid=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    runs = scan_completed_runs(args.output_root, args.ground_truth_dir)
    if not runs:
        raise SystemExit(f"No completed comparison runs found under {args.output_root}")

    stats = aggregate_model_setup(runs)
    generated: List[Path] = []

    plot_accuracy_by_setup(stats, args.plots_dir, generated)
    plot_over_under_by_setup(stats, args.plots_dir, generated)
    plot_committee_gain(stats, args.plots_dir, generated)
    plot_scenario_exact_heatmap(runs, args.plots_dir, generated)
    write_summary_table(stats, args.plots_dir, generated)

    print(f"Generated {len(generated)} comparison plot files in {args.plots_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
