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
from matplotlib.patches import Patch

from thesis_plot_style import apply_thesis_style

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT / "cyber_negotiation_v1" / "games_descriptions" / "cyber_game" / "output"
DEFAULT_GROUND_TRUTH_DIR = ROOT / "cyber_negotiation_v1" / "games_descriptions" / "cyber_game" / "ground_truth"
DEFAULT_PLOTS_DIR = ROOT / "viewer" / "plots" / "thesis" / "cybersecurity" / "kimi"

COLOR_KIMI = "#009E73"
COLOR_CLAUDE = "#D55E00"
COLOR_MIXED = "#CC79A7"
COLOR_NEUTRAL = "#4D4D4D"
COLOR_WARM = "#D55E00"
COLOR_COOL = "#0072B2"
EDGE_COLOR = "#333333"
GRID_COLOR = "#D0D0D0"

COLORS = {
    "Kimi": COLOR_KIMI,
    "Claude": COLOR_CLAUDE,
    "Mixed": COLOR_MIXED,
    "Neutral": COLOR_NEUTRAL,
    "Warm": COLOR_WARM,
    "Cool": COLOR_COOL,
}

CONDITION_META: Dict[str, Dict[str, str]] = {
    "C1.1": {"label": "Single Kimi", "family": "Kimi", "setup": "single", "prior": "none"},
    "C3.1": {"label": "3x Kimi", "family": "Kimi", "setup": "committee", "prior": "none"},
    "C4.1": {"label": "Kimi + Claude", "family": "Mixed", "setup": "mixed", "prior": "none"},
    "C5.1": {"label": "3x Kimi + LLM Prior", "family": "Kimi", "setup": "committee", "prior": "llm"},
    "C6.1": {"label": "3x Kimi + Human Prior", "family": "Kimi", "setup": "committee", "prior": "human"},
}

CATEGORY_ORDER = [
    "command_injection",
    "cookies",
    "csrf",
    "info_findings",
    "path_disclosure",
]

CATEGORY_TITLES = {
    "command_injection": "Command Injection",
    "cookies": "Cookies",
    "csrf": "CSRF",
    "info_findings": "Info Findings",
    "path_disclosure": "Path Disclosure",
}

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

SCENARIO_TITLES = {
    "command_injection_ping_001": "Ping Form Command Injection",
    "cookie_security_attribute_observation_001": "Cookie Flags Observation",
    "hard_cookie_md5_002": "MD5-Like Cookie Pattern",
    "info_apache": "Apache Header Disclosure",
    "medium_cookie_timestamps_001": "Timestamp Cookie Pattern",
    "reflected_input_password_change_guard_001": "Reflected Input Password Change Guard",
    "error_message_path_disclosure_001": "Error Message Path Disclosure",
}

SCENARIO_CATEGORY = {
    "command_injection_ping_001": "command_injection",
    "cookie_security_attribute_observation_001": "cookies",
    "hard_cookie_md5_002": "cookies",
    "medium_cookie_timestamps_001": "cookies",
    "reflected_input_password_change_guard_001": "csrf",
    "info_apache": "info_findings",
    "error_message_path_disclosure_001": "path_disclosure",
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
    scenario_key: str
    scenario_title: str
    category_key: str
    category_title: str
    run_dir: Path
    metrics_path: Path
    history_path: Optional[Path]
    exact: float
    finding_type: float
    severity: float
    wrong: float
    final_agreement_exact: float
    any_agreement_exact: float
    final_agreement_type: float
    any_agreement_type: float
    severity_bias: float
    public_turns: float
    type_transitions: float
    severity_transitions: float
    under_severity: Optional[float]
    over_severity: Optional[float]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_scenario_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return SCENARIO_ALIASES.get(str(value), str(value))


def scenario_title(scenario_key: str) -> str:
    if scenario_key in SCENARIO_TITLES:
        return SCENARIO_TITLES[scenario_key]
    parts = [part for part in scenario_key.split("_") if part and not part.isdigit()]
    return " ".join(part.capitalize() for part in parts) or scenario_key


def condition_sort_key(condition_id: str) -> tuple[int, int]:
    match = re.match(r"^C(\d+)(?:\.(\d+))?$", condition_id)
    if not match:
        return (999, 999)
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor)


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


def save(fig: plt.Figure, out_dir: Path, name: str, generated: List[Path]) -> None:
    out_path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(out_path)


def committee_type_value(snapshot: Dict[str, Any]) -> str:
    value = snapshot.get("committee_type_label") or snapshot.get("committee_type")
    return str(value) if value is not None else "None"


def committee_severity_value(snapshot: Dict[str, Any]) -> str:
    exact = snapshot.get("committee_exact")
    if isinstance(exact, dict):
        severity = str(exact.get("severity", "")).strip()
        if severity:
            return severity
    value = snapshot.get("committee_exact_severity")
    return str(value).strip() if value else "None"


def count_transitions(values: Sequence[str]) -> int:
    if not values:
        return 0
    transitions = 0
    for left, right in zip(values, values[1:]):
        if left != right:
            transitions += 1
    return transitions


def load_ground_truth_map(ground_truth_dir: Path) -> Dict[str, Dict[str, Any]]:
    ground_truth: Dict[str, Dict[str, Any]] = {}
    for path in sorted(ground_truth_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        scenario_key = canonical_scenario_id(payload.get("scenario_id")) or canonical_scenario_id(path.stem)
        if not scenario_key:
            continue
        ground_truth[scenario_key] = payload
        ground_truth[canonical_scenario_id(path.stem) or path.stem] = payload
    return ground_truth


def latest_completed_metrics(run_dir: Path) -> Optional[tuple[Path, Dict[str, Any], Optional[Path], Dict[str, Any]]]:
    candidates = sorted(run_dir.glob("metrics_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for metrics_path in candidates:
        try:
            payload = load_json(metrics_path)
        except json.JSONDecodeError:
            continue
        run_report = payload.get("run_report") or {}
        appendix = run_report.get("appendix_debug") or {}
        run_completed = bool(appendix.get("RunCompleted")) or str(appendix.get("RunStatus") or "").lower() == "completed"
        if not run_completed:
            continue
        run_id = str(run_report.get("run_id") or "").strip()
        history_path = run_dir / f"{run_id}.json" if run_id else None
        history_payload: Dict[str, Any] = {}
        if history_path and history_path.exists():
            try:
                history_payload = load_json(history_path)
            except json.JSONDecodeError:
                history_payload = {}
        return metrics_path, payload, history_path, history_payload
    return None


def extract_run_entry(
    run_dir: Path,
    ground_truth_map: Dict[str, Dict[str, Any]],
) -> Optional[RunEntry]:
    latest = latest_completed_metrics(run_dir)
    if not latest:
        return None
    metrics_path, metrics_payload, history_path, history_payload = latest
    run_report = metrics_payload.get("run_report") or {}
    condition_id = str(run_report.get("condition_id") or "").strip()
    if not condition_id.startswith("C") or condition_id not in CONDITION_META:
        return None
    scenario_key = canonical_scenario_id(run_report.get("scenario_id")) or canonical_scenario_id(run_dir.name)
    if not scenario_key or scenario_key not in SCENARIO_CATEGORY:
        return None

    category_key = SCENARIO_CATEGORY[scenario_key]
    category_title = CATEGORY_TITLES[category_key]
    scenario_label = scenario_title(scenario_key)

    headline = run_report.get("headline_metrics") or {}
    derived = run_report.get("derived_metrics") or {}
    trajectory = list(run_report.get("decision_trajectory") or history_payload.get("decision_trajectory") or [])

    type_states = [committee_type_value(snapshot) for snapshot in trajectory]
    severity_states = [committee_severity_value(snapshot) for snapshot in trajectory]
    public_turns = float(sum(1 for snapshot in trajectory if snapshot.get("phase") == "public"))
    type_transitions = float(count_transitions(type_states))
    severity_transitions = float(count_transitions(severity_states))

    final_severity = committee_severity_value(run_report.get("committee_final") or {})
    ground_truth = ground_truth_map.get(scenario_key, {})
    gt_severity = str(ground_truth.get("final_severity") or "").strip()
    final_num = SEVERITY_ORDER.get(final_severity)
    gt_num = SEVERITY_ORDER.get(gt_severity)
    under_severity = None
    over_severity = None
    if final_num is not None and gt_num is not None:
        under_severity = 1.0 if final_num < gt_num else 0.0
        over_severity = 1.0 if final_num > gt_num else 0.0

    return RunEntry(
        condition_id=condition_id,
        scenario_key=scenario_key,
        scenario_title=scenario_label,
        category_key=category_key,
        category_title=category_title,
        run_dir=run_dir,
        metrics_path=metrics_path,
        history_path=history_path,
        exact=float(headline.get("FinalCorrectExact", 0.0)),
        finding_type=float(headline.get("FinalCorrectType", 0.0)),
        severity=float(headline.get("FinalCorrectSeverity", 0.0)),
        wrong=float(derived.get("WrongConsensusExact", 0.0)),
        final_agreement_exact=float(headline.get("FinalAgreementExact", 0.0)),
        any_agreement_exact=float(headline.get("AnyAgreementExact", 0.0)),
        final_agreement_type=float(derived.get("FinalAgreementType", 0.0)),
        any_agreement_type=float(derived.get("AnyAgreementType", 0.0)),
        severity_bias=float(headline.get("SeverityBias", 0.0)),
        public_turns=public_turns,
        type_transitions=type_transitions,
        severity_transitions=severity_transitions,
        under_severity=under_severity,
        over_severity=over_severity,
    )


def collect_runs(output_root: Path, ground_truth_dir: Path) -> List[RunEntry]:
    ground_truth_map = load_ground_truth_map(ground_truth_dir)
    runs: List[RunEntry] = []
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("kimi_"):
            continue
        entry = extract_run_entry(run_dir, ground_truth_map)
        if entry is not None:
            runs.append(entry)
    return runs


def aggregate_conditions(runs: Sequence[RunEntry]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[RunEntry]] = defaultdict(list)
    for run in runs:
        grouped[run.condition_id].append(run)

    rows: List[Dict[str, Any]] = []
    for condition_id, items in sorted(grouped.items(), key=lambda item: condition_sort_key(item[0])):
        meta = CONDITION_META[condition_id]
        rows.append(
            {
                "condition_id": condition_id,
                "label": meta["label"],
                "family": meta["family"],
                "setup": meta["setup"],
                "prior": meta["prior"],
                "runs": len(items),
                "exact": mean_optional(run.exact for run in items),
                "type": mean_optional(run.finding_type for run in items),
                "severity": mean_optional(run.severity for run in items),
                "wrong": mean_optional(run.wrong for run in items),
                "severity_bias": mean_optional(run.severity_bias for run in items),
                "public_turns": mean_optional(run.public_turns for run in items),
                "type_transitions": mean_optional(run.type_transitions for run in items),
                "severity_transitions": mean_optional(run.severity_transitions for run in items),
                "under": mean_optional(run.under_severity for run in items),
                "over": mean_optional(run.over_severity for run in items),
                "agreement_exact": mean_optional(run.final_agreement_exact for run in items),
            }
        )
    return rows


def aggregate_scenarios(runs: Sequence[RunEntry]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[RunEntry]] = defaultdict(list)
    for run in runs:
        grouped[run.scenario_key].append(run)

    rows: List[Dict[str, Any]] = []
    for scenario_key, items in grouped.items():
        first = items[0]
        rows.append(
            {
                "scenario_key": scenario_key,
                "scenario": first.scenario_title,
                "category_key": first.category_key,
                "category": first.category_title,
                "runs": len(items),
                "exact": mean_optional(run.exact for run in items),
                "type": mean_optional(run.finding_type for run in items),
                "severity": mean_optional(run.severity for run in items),
                "wrong": mean_optional(run.wrong for run in items),
                "under": mean_optional(run.under_severity for run in items),
            }
        )
    rows.sort(
        key=lambda row: (
            CATEGORY_ORDER.index(row["category_key"]) if row["category_key"] in CATEGORY_ORDER else 999,
            row["scenario"],
        )
    )
    return rows


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]], out_dir: Path, name: str, generated: List[Path]) -> None:
    nrows = len(rows) + 1
    ncols = len(headers)
    col_widths = [1.0, 0.7, 0.9, 0.9, 1.0, 1.0, 0.9, 0.9, 0.9, 1.0]
    fig_w = sum(col_widths) * 0.58
    fig_h = nrows * 0.34 + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    table = ax.table(cellText=[list(headers), *[list(row) for row in rows]], cellLoc="center", loc="center")

    for (r_idx, c_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.6)
        if r_idx == 0:
            cell.set_facecolor("#F2F5F8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    total = sum(col_widths)
    for c_idx in range(ncols):
        width = col_widths[c_idx] / total
        for r_idx in range(nrows):
            table[(r_idx, c_idx)].set_width(width)

    save(fig, out_dir, name, generated)


def plot_exact_by_condition(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=300)
    xs = np.arange(len(cond_rows))
    values = [row["exact"] for row in cond_rows]
    bar_colors = [COLORS.get(row["family"], COLOR_NEUTRAL) for row in cond_rows]
    ax.bar(xs, values, color=bar_colors, edgecolor=EDGE_COLOR, linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([row["condition_id"] for row in cond_rows])
    ax.set_ylabel("Final exact correctness")
    setup_ax(ax, (0.0, 1.0))
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, fmt_pct(value), ha="center", va="bottom", fontsize=8)
    save(fig, out_dir, "kimi_exact_by_condition", generated)


def plot_type_vs_severity_by_condition(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=300)
    xs = np.arange(len(cond_rows))
    width = 0.35
    type_values = [row["type"] for row in cond_rows]
    severity_values = [row["severity"] for row in cond_rows]
    colors = [COLORS.get(row["family"], COLOR_NEUTRAL) for row in cond_rows]

    ax.bar(xs - width / 2, type_values, width=width, color=colors, edgecolor=EDGE_COLOR, linewidth=0.6)
    ax.bar(xs + width / 2, severity_values, width=width, color=colors, edgecolor=EDGE_COLOR, linewidth=0.6, hatch="//")
    ax.set_xticks(xs)
    ax.set_xticklabels([row["condition_id"] for row in cond_rows])
    ax.set_ylabel("Correctness")
    setup_ax(ax, (0.0, 1.0))

    families = []
    for row in cond_rows:
        if row["family"] not in families:
            families.append(row["family"])
    legend_models = [Patch(facecolor=COLORS[family], edgecolor=EDGE_COLOR, label=family) for family in families if family in COLORS]
    legend_metrics = [
        Patch(facecolor="white", edgecolor=EDGE_COLOR, label="Type"),
        Patch(facecolor="white", edgecolor=EDGE_COLOR, hatch="//", label="Severity"),
    ]
    if legend_models:
        legend_a = ax.legend(handles=legend_models, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.add_artist(legend_a)
    ax.legend(handles=legend_metrics, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 0.65))
    save(fig, out_dir, "kimi_type_vs_severity_by_condition", generated)


def plot_single_vs_committee(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    by_condition = {row["condition_id"]: row for row in cond_rows}
    if "C1.1" not in by_condition or "C3.1" not in by_condition:
        return

    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    single = by_condition["C1.1"]
    committee = by_condition["C3.1"]

    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=300)
    xs = np.arange(len(metrics))
    width = 0.35
    ax.bar(xs - width / 2, [single[key] for key in metrics], width=width, color=COLOR_KIMI, edgecolor=EDGE_COLOR, linewidth=0.6, label="Single")
    ax.bar(xs + width / 2, [committee[key] for key in metrics], width=width, color=COLOR_KIMI, edgecolor=EDGE_COLOR, linewidth=0.6, hatch="//", label="3-agent no-prior")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    setup_ax(ax, (0.0, 1.0))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "kimi_single_vs_committee", generated)


def plot_severity_bias_by_condition(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=300)
    xs = np.arange(len(cond_rows))
    values = [row["severity_bias"] for row in cond_rows]
    colors = [COLOR_WARM if value > 0 else COLOR_COOL for value in values]
    ax.bar(xs, values, color=colors, edgecolor=EDGE_COLOR, linewidth=0.6)
    ax.axhline(0, color="#222222", linewidth=1.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([row["condition_id"] for row in cond_rows])
    ax.set_ylabel("Severity bias")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, out_dir, "kimi_severity_bias_by_condition", generated)


def plot_type_vs_severity_transitions(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=300)
    xs = np.arange(len(cond_rows))
    width = 0.35
    type_values = [row["type_transitions"] for row in cond_rows]
    severity_values = [row["severity_transitions"] for row in cond_rows]
    colors = [COLORS.get(row["family"], COLOR_NEUTRAL) for row in cond_rows]

    ax.bar(xs - width / 2, type_values, width=width, color=colors, edgecolor=EDGE_COLOR, linewidth=0.6)
    ax.bar(xs + width / 2, severity_values, width=width, color=colors, edgecolor=EDGE_COLOR, linewidth=0.6, hatch="//")
    ax.set_xticks(xs)
    ax.set_xticklabels([row["condition_id"] for row in cond_rows])
    ax.set_ylabel("Mean transitions")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)

    for index, value in enumerate(type_values):
        ax.text(index - width / 2, value + 0.05, fmt_float(value), ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(severity_values):
        ax.text(index + width / 2, value + 0.05, fmt_float(value), ha="center", va="bottom", fontsize=8)

    legend_metrics = [
        Patch(facecolor="white", edgecolor=EDGE_COLOR, label="Type"),
        Patch(facecolor="white", edgecolor=EDGE_COLOR, hatch="//", label="Severity"),
    ]
    ax.legend(handles=legend_metrics, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "kimi_type_vs_severity_transitions", generated)


def plot_scenario_heatmap(scenario_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    scenarios = [f"{row['category']} | {row['scenario']}" for row in scenario_rows]
    values = np.array([[row[key] for key in metrics] for row in scenario_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.45 * len(scenarios))), dpi=300)
    image = ax.imshow(values, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                fmt_pct(value),
                ha="center",
                va="center",
                color="white" if value < 0.5 else "black",
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Rate", fontsize=8)
    save(fig, out_dir, "kimi_scenario_heatmap", generated)


def plot_agreement_vs_wrong_consensus(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    baseline = [row for row in cond_rows if row["setup"] == "single"]
    negotiation = [row for row in cond_rows if row["setup"] != "single"]
    if not baseline or not negotiation:
        return

    baseline_agree = mean_optional(row["agreement_exact"] for row in baseline)
    baseline_wrong = mean_optional(row["wrong"] for row in baseline)
    negotiation_agree = mean_optional(row["agreement_exact"] for row in negotiation)
    negotiation_wrong = mean_optional(row["wrong"] for row in negotiation)

    fig, ax = plt.subplots(figsize=(5.8, 3.3), dpi=300)
    xs = np.arange(2)
    ax.scatter(xs - 0.08, [baseline_agree, baseline_wrong], color=COLOR_NEUTRAL, marker="o", s=48, edgecolor="#222222", linewidth=0.4, label="Baseline")
    ax.scatter(xs + 0.08, [negotiation_agree, negotiation_wrong], color=COLOR_KIMI, marker="s", s=48, edgecolor="#222222", linewidth=0.4, label="Negotiation")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Agreement", "Wrong consensus"])
    ax.set_ylabel("Rate")
    setup_ax(ax, (0.0, 1.0))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "kimi_agreement_vs_wrong_consensus", generated)


def plot_cookie_breakdown(scenario_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    cookie_rows = [row for row in scenario_rows if row["category_key"] == "cookies"]
    if not cookie_rows:
        return
    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    values = np.array([[row[key] for key in metrics] for row in cookie_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 2.6), dpi=300)
    image = ax.imshow(values, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(cookie_rows)))
    ax.set_yticklabels([row["scenario"] for row in cookie_rows])
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                fmt_pct(value),
                ha="center",
                va="center",
                color="white" if value < 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).set_label("Rate")
    save(fig, out_dir, "kimi_cookie_breakdown", generated)


def plot_ceiling_vs_ambiguity(scenario_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    ceiling = {"csrf", "command_injection", "info_findings"}
    ambiguity = {"cookies", "path_disclosure"}

    def group_mean(category_keys: set[str]) -> Dict[str, float]:
        rows = [row for row in scenario_rows if row["category_key"] in category_keys]
        return {
            "exact": mean_optional(row["exact"] for row in rows),
            "type": mean_optional(row["type"] for row in rows),
            "severity": mean_optional(row["severity"] for row in rows),
            "wrong": mean_optional(row["wrong"] for row in rows),
        }

    ceiling_values = group_mean(ceiling)
    ambiguity_values = group_mean(ambiguity)

    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=300)
    xs = np.arange(len(metrics))
    width = 0.35
    ax.bar(xs - width / 2, [ceiling_values[key] for key in metrics], width=width, color=COLOR_NEUTRAL, edgecolor=EDGE_COLOR, linewidth=0.6, label="Ceiling")
    ax.bar(xs + width / 2, [ambiguity_values[key] for key in metrics], width=width, color="#999999", edgecolor=EDGE_COLOR, linewidth=0.6, hatch="//", label="Ambiguity")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    setup_ax(ax, (0.0, 1.0))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "kimi_ceiling_vs_ambiguity", generated)


def plot_mixed_vs_homogeneous(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    by_condition = {row["condition_id"]: row for row in cond_rows}
    if "C3.1" not in by_condition or "C4.1" not in by_condition:
        return

    homogeneous = by_condition["C3.1"]
    mixed = by_condition["C4.1"]
    metrics = ["exact", "severity", "wrong", "under"]
    labels = ["Exact", "Severity", "Wrong", "Under"]

    fig, ax = plt.subplots(figsize=(6.8, 3.3), dpi=300)
    xs = np.arange(len(metrics))
    width = 0.35
    bars_h = ax.bar(xs - width / 2, [homogeneous[key] for key in metrics], width=width, color=COLOR_NEUTRAL, edgecolor=EDGE_COLOR, linewidth=0.6, label="Homogeneous (C3.1)")
    bars_m = ax.bar(xs + width / 2, [mixed[key] for key in metrics], width=width, color=COLOR_MIXED, edgecolor=EDGE_COLOR, linewidth=0.6, label="Mixed (C4.1)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    setup_ax(ax, (0.0, 1.0))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    for bar in list(bars_h) + list(bars_m):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, fmt_pct(height), ha="center", va="bottom", fontsize=8)
    save(fig, out_dir, "kimi_mixed_vs_homogeneous", generated)


def plot_mixed_vs_homogeneous_bias(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    by_condition = {row["condition_id"]: row for row in cond_rows}
    if "C3.1" not in by_condition or "C4.1" not in by_condition:
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.3), dpi=300)
    xs = np.arange(2)
    values = [by_condition["C3.1"]["severity_bias"], by_condition["C4.1"]["severity_bias"]]
    ax.bar(xs[0], values[0], color=COLOR_NEUTRAL, edgecolor=EDGE_COLOR, linewidth=0.6, label="Homogeneous (C3.1)")
    ax.bar(xs[1], values[1], color=COLOR_MIXED, edgecolor=EDGE_COLOR, linewidth=0.6, label="Mixed (C4.1)")
    ax.axhline(0, color="#222222", linewidth=1.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Homogeneous", "Mixed"])
    ax.set_ylabel("Severity bias")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, out_dir, "kimi_mixed_vs_homogeneous_bias", generated)


def plot_mixed_vs_priors(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    by_condition = {row["condition_id"]: row for row in cond_rows}
    compare = []
    if "C4.1" in by_condition:
        compare.append(("Mixed (C4.1)", by_condition["C4.1"], COLOR_MIXED, ""))
    if "C5.1" in by_condition:
        compare.append(("LLM prior (C5.1)", by_condition["C5.1"], COLOR_KIMI, "//"))
    if "C6.1" in by_condition:
        compare.append(("Human prior (C6.1)", by_condition["C6.1"], COLOR_CLAUDE, "\\\\"))
    if len(compare) < 2:
        return

    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=300)
    xs = np.arange(len(metrics))
    width = 0.24
    offsets = np.linspace(-width, width, len(compare))
    bars: List[Any] = []
    for offset, (label, row, color, hatch) in zip(offsets, compare):
        current = ax.bar(
            xs + offset,
            [row[key] for key in metrics],
            width=width,
            color=color,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
            hatch=hatch,
            label=label,
        )
        bars.extend(current)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    setup_ax(ax, (0.0, 1.0))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, fmt_pct(height), ha="center", va="bottom", fontsize=7)
    save(fig, out_dir, "kimi_mixed_vs_priors", generated)


def plot_prior_effects(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    by_condition = {row["condition_id"]: row for row in cond_rows}
    panels = []
    if "C3.1" in by_condition and "C5.1" in by_condition:
        panels.append(("LLM prior", by_condition["C3.1"], by_condition["C5.1"]))
    if "C3.1" in by_condition and "C6.1" in by_condition:
        panels.append(("Human prior", by_condition["C3.1"], by_condition["C6.1"]))
    if not panels:
        return

    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), dpi=300, sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, base_row, prior_row) in zip(axes, panels):
        xs = np.arange(len(metrics))
        width = 0.35
        ax.bar(xs - width / 2, [base_row[key] for key in metrics], width=width, color=COLOR_KIMI, edgecolor=EDGE_COLOR, linewidth=0.6, label="No prior")
        ax.bar(xs + width / 2, [prior_row[key] for key in metrics], width=width, color=COLOR_KIMI, edgecolor=EDGE_COLOR, linewidth=0.6, hatch="//", label=title)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=10, pad=4)
        setup_ax(ax, (0.0, 1.0))
    axes[0].set_ylabel("Rate")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "kimi_prior_effects", generated)


def plot_prior_effects_bias(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    by_condition = {row["condition_id"]: row for row in cond_rows}
    panels = []
    if "C3.1" in by_condition and "C5.1" in by_condition:
        panels.append(("LLM prior", by_condition["C3.1"], by_condition["C5.1"]))
    if "C3.1" in by_condition and "C6.1" in by_condition:
        panels.append(("Human prior", by_condition["C3.1"], by_condition["C6.1"]))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.3), dpi=300, sharey=True)
    if len(panels) == 1:
        axes = [axes]

    all_values: List[float] = []
    for ax, (title, base_row, prior_row) in zip(axes, panels):
        xs = np.arange(2)
        values = [base_row["severity_bias"], prior_row["severity_bias"]]
        bars = ax.bar(xs, values, color=COLOR_KIMI, edgecolor=EDGE_COLOR, linewidth=0.6)
        bars[1].set_hatch("//")
        ax.axhline(0, color="#222222", linewidth=1.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(["No prior", "With prior"])
        ax.set_title(title, fontsize=10, pad=4)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for bar in bars:
            height = bar.get_height()
            offset = 0.05 if height >= 0 else -0.08
            ax.text(bar.get_x() + bar.get_width() / 2, height + offset, fmt_float(height), ha="center", va="bottom" if height >= 0 else "top", fontsize=8)
        all_values.extend(values)

    axes[0].set_ylabel("Severity bias")
    low = min(all_values) - 0.2
    high = max(all_values) + 0.2
    for ax in axes:
        ax.set_ylim(low, high)
    save(fig, out_dir, "kimi_prior_effects_bias", generated)


def write_condition_table(cond_rows: Sequence[Dict[str, Any]], out_dir: Path, generated: List[Path]) -> None:
    headers = [
        "Condition",
        "Runs",
        "Exact Correct",
        "Type Correct",
        "Severity Correct",
        "Wrong Consensus",
        "Severity Bias",
        "Public Turns",
        "Type Transitions",
        "Severity Transitions",
    ]
    rows = []
    for row in cond_rows:
        rows.append(
            [
                row["condition_id"],
                str(row["runs"]),
                fmt_pct(row["exact"]),
                fmt_pct(row["type"]),
                fmt_pct(row["severity"]),
                fmt_pct(row["wrong"]),
                fmt_float(row["severity_bias"]),
                fmt_float(row["public_turns"]),
                fmt_float(row["type_transitions"]),
                fmt_float(row["severity_transitions"]),
            ]
        )
    render_table(headers, rows, out_dir, "kimi_condition_statistics_table", generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kimi-only cybersecurity thesis plots.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Cyber output directory with kimi_* run folders.")
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR, help="Ground-truth directory used for severity comparisons.")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR, help="Destination directory for generated Kimi PNGs.")
    args = parser.parse_args()

    apply_thesis_style(font_size=11, y_grid=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args.output_root, args.ground_truth_dir)
    if not runs:
        raise SystemExit(f"No completed Kimi runs found in {args.output_root}")

    cond_rows = aggregate_conditions(runs)
    scenario_rows = aggregate_scenarios(runs)
    generated: List[Path] = []

    plot_exact_by_condition(cond_rows, args.plots_dir, generated)
    plot_type_vs_severity_by_condition(cond_rows, args.plots_dir, generated)
    plot_single_vs_committee(cond_rows, args.plots_dir, generated)
    plot_severity_bias_by_condition(cond_rows, args.plots_dir, generated)
    plot_type_vs_severity_transitions(cond_rows, args.plots_dir, generated)
    plot_scenario_heatmap(scenario_rows, args.plots_dir, generated)
    plot_agreement_vs_wrong_consensus(cond_rows, args.plots_dir, generated)
    plot_cookie_breakdown(scenario_rows, args.plots_dir, generated)
    plot_ceiling_vs_ambiguity(scenario_rows, args.plots_dir, generated)
    plot_mixed_vs_homogeneous(cond_rows, args.plots_dir, generated)
    plot_mixed_vs_homogeneous_bias(cond_rows, args.plots_dir, generated)
    plot_mixed_vs_priors(cond_rows, args.plots_dir, generated)
    plot_prior_effects(cond_rows, args.plots_dir, generated)
    plot_prior_effects_bias(cond_rows, args.plots_dir, generated)
    write_condition_table(cond_rows, args.plots_dir, generated)

    print(f"Generated {len(generated)} Kimi plot files in {args.plots_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
