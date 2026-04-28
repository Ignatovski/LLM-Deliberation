from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from thesis_plot_style import apply_thesis_style


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT / "cyber_negotiation_v1" / "games_descriptions" / "cyber_game" / "output"
DEFAULT_GROUND_TRUTH_DIR = ROOT / "cyber_negotiation_v1" / "games_descriptions" / "cyber_game" / "ground_truth"
DEFAULT_PLOTS_ROOT = ROOT / "viewer" / "plots" / "thesis" / "cybersecurity"

EDGE_COLOR = "#333333"
GRID_COLOR = "#D0D0D0"
COLOR_NEUTRAL = "#4D4D4D"
COLOR_WARM = "#D55E00"
COLOR_COOL = "#0072B2"

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

SCENARIO_CATEGORY = {
    "command_injection_ping_001": "command_injection",
    "cookie_security_attribute_observation_001": "cookies",
    "hard_cookie_md5_002": "cookies",
    "medium_cookie_timestamps_001": "cookies",
    "reflected_input_password_change_guard_001": "csrf",
    "info_apache": "info_findings",
    "error_message_path_disclosure_001": "path_disclosure",
}

SETUP_ORDER = ["single", "committee"]
SETUP_TITLES = {"single": "Single-Agent", "committee": "Three-Agent Committee"}

SEVERITY_ORDER = {
    "Info": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    label: str
    short_label: str
    setup: str


@dataclass(frozen=True)
class FamilySpec:
    key: str
    display_name: str
    axis_label: str
    color: str
    prefix: str
    conditions: tuple[ConditionSpec, ...]


FAMILY_ORDER = ["sonnet", "opus46", "opus47"]

FAMILY_SPECS: Dict[str, FamilySpec] = {
    "sonnet": FamilySpec(
        key="sonnet",
        display_name="Claude Sonnet",
        axis_label="Sonnet",
        color="#D55E00",
        prefix="sonnet",
        conditions=(
            ConditionSpec("C2", "Single Claude Sonnet", "Single", "single"),
            ConditionSpec("C4", "3x Claude Sonnet", "3-agent", "committee"),
        ),
    ),
    "opus46": FamilySpec(
        key="opus46",
        display_name="Claude Opus 4.6",
        axis_label="Opus 4.6",
        color="#009E73",
        prefix="opus46",
        conditions=(
            ConditionSpec("C1.1_OPUS46", "Single Claude Opus 4.6", "Single", "single"),
            ConditionSpec("C3.1_OPUS46", "3x Claude Opus 4.6", "3-agent", "committee"),
        ),
    ),
    "opus47": FamilySpec(
        key="opus47",
        display_name="Claude Opus 4.7",
        axis_label="Opus 4.7",
        color="#0072B2",
        prefix="opus47",
        conditions=(
            ConditionSpec("C1.1_OPUS47", "Single Claude Opus 4.7", "Single", "single"),
            ConditionSpec("C3.1_OPUS47", "3x Claude Opus 4.7", "3-agent", "committee"),
        ),
    ),
}

CONDITION_INDEX: Dict[str, tuple[str, ConditionSpec]] = {}
for family_key, family_spec in FAMILY_SPECS.items():
    for condition_spec in family_spec.conditions:
        CONDITION_INDEX[condition_spec.condition_id] = (family_key, condition_spec)


@dataclass
class RunEntry:
    family_key: str
    model_name: str
    model_axis_label: str
    color: str
    condition_id: str
    condition_label: str
    condition_short_label: str
    setup: str
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


def fmt_delta(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value * 100:+.0f} pp"


def run_metric_value(run: RunEntry, metric: str) -> float:
    if metric == "type":
        return run.finding_type
    return float(getattr(run, metric))


def plot_ready(values: Sequence[float]) -> List[float]:
    return [0.0 if math.isnan(value) else value for value in values]


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


def iter_run_dirs(output_root: Path) -> List[Path]:
    return sorted({path.parent for path in output_root.rglob("metrics_*.json")})


def extract_run_entry(run_dir: Path, ground_truth_map: Dict[str, Dict[str, Any]]) -> Optional[RunEntry]:
    latest = latest_completed_metrics(run_dir)
    if not latest:
        return None
    metrics_path, metrics_payload, history_path, history_payload = latest
    run_report = metrics_payload.get("run_report") or {}
    condition_id = str(run_report.get("condition_id") or "").strip()
    meta = CONDITION_INDEX.get(condition_id)
    if meta is None:
        return None
    family_key, condition_spec = meta
    family_spec = FAMILY_SPECS[family_key]

    scenario_key = canonical_scenario_id(run_report.get("scenario_id")) or canonical_scenario_id(run_dir.name)
    if not scenario_key or scenario_key not in SCENARIO_CATEGORY:
        return None

    category_key = SCENARIO_CATEGORY[scenario_key]
    category_title = CATEGORY_TITLES[category_key]
    scenario_title = SCENARIO_TITLES[scenario_key]

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
    severity = safe_float(headline.get("FinalCorrectSeverity"))
    under_severity = None
    over_severity = None
    if final_num is not None and gt_num is not None:
        severity = 1.0 if final_num == gt_num else 0.0
        under_severity = 1.0 if final_num < gt_num else 0.0
        over_severity = 1.0 if final_num > gt_num else 0.0

    return RunEntry(
        family_key=family_key,
        model_name=family_spec.display_name,
        model_axis_label=family_spec.axis_label,
        color=family_spec.color,
        condition_id=condition_id,
        condition_label=condition_spec.label,
        condition_short_label=condition_spec.short_label,
        setup=condition_spec.setup,
        scenario_key=scenario_key,
        scenario_title=scenario_title,
        category_key=category_key,
        category_title=category_title,
        run_dir=run_dir,
        metrics_path=metrics_path,
        history_path=history_path,
        exact=float(headline.get("FinalCorrectExact", 0.0)),
        finding_type=float(headline.get("FinalCorrectType", 0.0)),
        severity=severity if severity is not None else float("nan"),
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


def collect_runs(
    output_root: Path,
    ground_truth_dir: Path,
    *,
    family_keys: Optional[Sequence[str]] = None,
) -> List[RunEntry]:
    allowed = set(family_keys or FAMILY_SPECS.keys())
    ground_truth_map = load_ground_truth_map(ground_truth_dir)
    runs: List[RunEntry] = []
    for run_dir in iter_run_dirs(output_root):
        entry = extract_run_entry(run_dir, ground_truth_map)
        if entry is not None and entry.family_key in allowed:
            runs.append(entry)
    return runs


def aggregate_conditions(runs: Sequence[RunEntry], family_spec: FamilySpec) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[RunEntry]] = defaultdict(list)
    for run in runs:
        grouped[run.condition_id].append(run)

    rows: List[Dict[str, Any]] = []
    for condition_spec in family_spec.conditions:
        items = grouped.get(condition_spec.condition_id, [])
        rows.append(
            {
                "condition_id": condition_spec.condition_id,
                "label": condition_spec.label,
                "short_label": condition_spec.short_label,
                "setup": condition_spec.setup,
                "family": family_spec.display_name,
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
    for scenario_key in SCENARIO_ORDER:
        items = grouped.get(scenario_key, [])
        category_key = SCENARIO_CATEGORY[scenario_key]
        rows.append(
            {
                "scenario_key": scenario_key,
                "scenario": SCENARIO_TITLES[scenario_key],
                "category_key": category_key,
                "category": CATEGORY_TITLES[category_key],
                "runs": len(items),
                "exact": mean_optional(run.exact for run in items),
                "type": mean_optional(run.finding_type for run in items),
                "severity": mean_optional(run.severity for run in items),
                "wrong": mean_optional(run.wrong for run in items),
                "under": mean_optional(run.under_severity for run in items),
            }
        )
    return rows


def masked_image(values: np.ndarray, cmap_name: str = "cividis") -> tuple[np.ma.MaskedArray, Any]:
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#F2F5F8")
    return masked, cmap


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]], out_dir: Path, name: str, generated: List[Path]) -> None:
    nrows = len(rows) + 1
    fig_h = nrows * 0.34 + 0.4
    fig_w = max(7.0, len(headers) * 0.78)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    table = ax.table(cellText=[list(headers), *[list(row) for row in rows]], cellLoc="center", loc="center")

    for (r_idx, _c_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.6)
        if r_idx == 0:
            cell.set_facecolor("#F2F5F8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    save(fig, out_dir, name, generated)


def plot_exact_by_condition(cond_rows: Sequence[Dict[str, Any]], family_spec: FamilySpec, out_dir: Path, generated: List[Path]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=300)
    xs = np.arange(len(cond_rows))
    values = [row["exact"] for row in cond_rows]
    bars = ax.bar(xs, plot_ready(values), color=family_spec.color, edgecolor=EDGE_COLOR, linewidth=0.6)
    for bar, row in zip(bars, cond_rows):
        if row["setup"] == "committee":
            bar.set_hatch("//")
    ax.set_xticks(xs)
    ax.set_xticklabels([row["short_label"] for row in cond_rows])
    ax.set_ylabel("Final exact correctness")
    setup_ax(ax, (0.0, 1.0))
    for index, value in enumerate(values):
        ax.text(index, (0.02 if math.isnan(value) else value + 0.02), fmt_pct(value), ha="center", va="bottom", fontsize=8)
    save(fig, out_dir, f"{family_spec.prefix}_exact_by_condition", generated)


def plot_type_vs_severity_by_condition(
    cond_rows: Sequence[Dict[str, Any]],
    family_spec: FamilySpec,
    out_dir: Path,
    generated: List[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=300)
    xs = np.arange(len(cond_rows))
    width = 0.35
    type_values = [row["type"] for row in cond_rows]
    severity_values = [row["severity"] for row in cond_rows]

    ax.bar(xs - width / 2, plot_ready(type_values), width=width, color=family_spec.color, edgecolor=EDGE_COLOR, linewidth=0.6)
    ax.bar(
        xs + width / 2,
        plot_ready(severity_values),
        width=width,
        color=family_spec.color,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        hatch="//",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([row["short_label"] for row in cond_rows])
    ax.set_ylabel("Correctness")
    setup_ax(ax, (0.0, 1.0))
    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor=EDGE_COLOR, label="Type"),
            Patch(facecolor="white", edgecolor=EDGE_COLOR, hatch="//", label="Severity"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    save(fig, out_dir, f"{family_spec.prefix}_type_vs_severity_by_condition", generated)


def plot_single_vs_committee(cond_rows: Sequence[Dict[str, Any]], family_spec: FamilySpec, out_dir: Path, generated: List[Path]) -> None:
    by_setup = {row["setup"]: row for row in cond_rows}
    if "single" not in by_setup or "committee" not in by_setup:
        return

    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    single = by_setup["single"]
    committee = by_setup["committee"]

    fig, ax = plt.subplots(figsize=(5.8, 3.3), dpi=300)
    xs = np.arange(len(metrics))
    width = 0.35
    ax.bar(
        xs - width / 2,
        plot_ready([single[key] for key in metrics]),
        width=width,
        color=family_spec.color,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        label="Single",
    )
    ax.bar(
        xs + width / 2,
        plot_ready([committee[key] for key in metrics]),
        width=width,
        color=family_spec.color,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        hatch="//",
        label="3-agent",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    setup_ax(ax, (0.0, 1.0))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, f"{family_spec.prefix}_single_vs_committee", generated)


def plot_severity_bias_by_condition(
    cond_rows: Sequence[Dict[str, Any]],
    family_spec: FamilySpec,
    out_dir: Path,
    generated: List[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=300)
    xs = np.arange(len(cond_rows))
    values = [row["severity_bias"] for row in cond_rows]
    colors = [COLOR_WARM if value > 0 else COLOR_COOL for value in values]
    ax.bar(xs, plot_ready(values), color=colors, edgecolor=EDGE_COLOR, linewidth=0.6)
    ax.axhline(0, color="#222222", linewidth=1.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([row["short_label"] for row in cond_rows])
    ax.set_ylabel("Severity bias")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, out_dir, f"{family_spec.prefix}_severity_bias_by_condition", generated)


def plot_type_vs_severity_transitions(
    cond_rows: Sequence[Dict[str, Any]],
    family_spec: FamilySpec,
    out_dir: Path,
    generated: List[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=300)
    xs = np.arange(len(cond_rows))
    width = 0.35
    type_values = [row["type_transitions"] for row in cond_rows]
    severity_values = [row["severity_transitions"] for row in cond_rows]

    ax.bar(xs - width / 2, plot_ready(type_values), width=width, color=family_spec.color, edgecolor=EDGE_COLOR, linewidth=0.6)
    ax.bar(
        xs + width / 2,
        plot_ready(severity_values),
        width=width,
        color=family_spec.color,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        hatch="//",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([row["short_label"] for row in cond_rows])
    ax.set_ylabel("Mean transitions")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)

    for index, value in enumerate(type_values):
        ax.text(index - width / 2, (0.05 if math.isnan(value) else value + 0.05), fmt_float(value), ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(severity_values):
        ax.text(index + width / 2, (0.05 if math.isnan(value) else value + 0.05), fmt_float(value), ha="center", va="bottom", fontsize=8)

    ax.legend(
        handles=[
            Patch(facecolor="white", edgecolor=EDGE_COLOR, label="Type"),
            Patch(facecolor="white", edgecolor=EDGE_COLOR, hatch="//", label="Severity"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    save(fig, out_dir, f"{family_spec.prefix}_type_vs_severity_transitions", generated)


def plot_scenario_heatmap(scenario_rows: Sequence[Dict[str, Any]], family_spec: FamilySpec, out_dir: Path, generated: List[Path]) -> None:
    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    scenarios = [f"{row['category']} | {row['scenario']}" for row in scenario_rows]
    values = np.array([[row[key] for key in metrics] for row in scenario_rows], dtype=float)
    masked, cmap = masked_image(values)

    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.45 * len(scenarios))), dpi=300)
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            text = fmt_pct(value)
            color = "black" if math.isnan(value) or value >= 0.5 else "white"
            ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Rate", fontsize=8)
    save(fig, out_dir, f"{family_spec.prefix}_scenario_heatmap", generated)


def plot_agreement_vs_wrong_consensus(
    cond_rows: Sequence[Dict[str, Any]],
    family_spec: FamilySpec,
    out_dir: Path,
    generated: List[Path],
) -> None:
    baseline = [row for row in cond_rows if row["setup"] == "single"]
    negotiation = [row for row in cond_rows if row["setup"] == "committee"]
    if not baseline or not negotiation:
        return

    baseline_agree = mean_optional(row["agreement_exact"] for row in baseline)
    baseline_wrong = mean_optional(row["wrong"] for row in baseline)
    negotiation_agree = mean_optional(row["agreement_exact"] for row in negotiation)
    negotiation_wrong = mean_optional(row["wrong"] for row in negotiation)

    fig, ax = plt.subplots(figsize=(5.4, 3.3), dpi=300)
    xs = np.arange(2)
    ax.scatter(
        xs - 0.08,
        [baseline_agree, baseline_wrong],
        color=COLOR_NEUTRAL,
        marker="o",
        s=48,
        edgecolor="#222222",
        linewidth=0.4,
        label="Single",
    )
    ax.scatter(
        xs + 0.08,
        [negotiation_agree, negotiation_wrong],
        color=family_spec.color,
        marker="s",
        s=48,
        edgecolor="#222222",
        linewidth=0.4,
        label="Committee",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(["Agreement", "Wrong consensus"])
    ax.set_ylabel("Rate")
    setup_ax(ax, (0.0, 1.0))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, f"{family_spec.prefix}_agreement_vs_wrong_consensus", generated)


def plot_cookie_breakdown(scenario_rows: Sequence[Dict[str, Any]], family_spec: FamilySpec, out_dir: Path, generated: List[Path]) -> None:
    cookie_rows = [row for row in scenario_rows if row["category_key"] == "cookies"]
    if not cookie_rows:
        return
    metrics = ["exact", "type", "severity", "wrong"]
    labels = ["Exact", "Type", "Severity", "Wrong"]
    values = np.array([[row[key] for key in metrics] for row in cookie_rows], dtype=float)
    masked, cmap = masked_image(values)

    fig, ax = plt.subplots(figsize=(6.6, 2.6), dpi=300)
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(cookie_rows)))
    ax.set_yticklabels([row["scenario"] for row in cookie_rows])
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            text = fmt_pct(value)
            color = "black" if math.isnan(value) or value >= 0.5 else "white"
            ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).set_label("Rate")
    save(fig, out_dir, f"{family_spec.prefix}_cookie_breakdown", generated)


def plot_ceiling_vs_ambiguity(scenario_rows: Sequence[Dict[str, Any]], family_spec: FamilySpec, out_dir: Path, generated: List[Path]) -> None:
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
    fig, ax = plt.subplots(figsize=(5.8, 3.3), dpi=300)
    xs = np.arange(len(metrics))
    width = 0.35
    ax.bar(
        xs - width / 2,
        plot_ready([ceiling_values[key] for key in metrics]),
        width=width,
        color=COLOR_NEUTRAL,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        label="Ceiling",
    )
    ax.bar(
        xs + width / 2,
        plot_ready([ambiguity_values[key] for key in metrics]),
        width=width,
        color="#999999",
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        hatch="//",
        label="Ambiguity",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    setup_ax(ax, (0.0, 1.0))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, f"{family_spec.prefix}_ceiling_vs_ambiguity", generated)


def write_condition_table(cond_rows: Sequence[Dict[str, Any]], family_spec: FamilySpec, out_dir: Path, generated: List[Path]) -> None:
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
                f"{row['label']} ({row['condition_id']})",
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
    render_table(headers, rows, out_dir, f"{family_spec.prefix}_condition_statistics_table", generated)


def generate_family_plots(family_spec: FamilySpec, runs: Sequence[RunEntry], plots_root: Path) -> List[Path]:
    out_dir = plots_root / family_spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_rows = aggregate_conditions(runs, family_spec)
    scenario_rows = aggregate_scenarios(runs)
    generated: List[Path] = []

    plot_exact_by_condition(cond_rows, family_spec, out_dir, generated)
    plot_type_vs_severity_by_condition(cond_rows, family_spec, out_dir, generated)
    plot_single_vs_committee(cond_rows, family_spec, out_dir, generated)
    plot_severity_bias_by_condition(cond_rows, family_spec, out_dir, generated)
    plot_type_vs_severity_transitions(cond_rows, family_spec, out_dir, generated)
    plot_scenario_heatmap(scenario_rows, family_spec, out_dir, generated)
    plot_agreement_vs_wrong_consensus(cond_rows, family_spec, out_dir, generated)
    plot_cookie_breakdown(scenario_rows, family_spec, out_dir, generated)
    plot_ceiling_vs_ambiguity(scenario_rows, family_spec, out_dir, generated)
    write_condition_table(cond_rows, family_spec, out_dir, generated)
    return generated


def aggregate_model_setup(runs: Sequence[RunEntry]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for setup in SETUP_ORDER:
        for family_key in FAMILY_ORDER:
            model_key = family_key
            items = [run for run in runs if run.setup == setup and run.family_key == family_key]
            grouped[setup][model_key] = {
                "exact": mean_optional(run.exact for run in items),
                "type": mean_optional(run.finding_type for run in items),
                "severity": mean_optional(run.severity for run in items),
                "under": mean_optional(run.under_severity for run in items),
                "over": mean_optional(run.over_severity for run in items),
                "wrong": mean_optional(run.wrong for run in items),
                "severity_bias": mean_optional(run.severity_bias for run in items),
            }
    return grouped


def plot_comparison_accuracy_by_setup(stats: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path, generated: List[Path]) -> None:
    metrics = ["exact", "type", "severity"]
    labels = ["Exact", "Type", "Severity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), dpi=300, sharey=True)

    for ax, setup in zip(axes, SETUP_ORDER):
        xs = np.arange(len(metrics))
        width = 0.22
        offsets = np.linspace(-width, width, len(FAMILY_ORDER))
        for offset, family_key in zip(offsets, FAMILY_ORDER):
            spec = FAMILY_SPECS[family_key]
            values = [stats[setup][family_key][metric] for metric in metrics]
            ax.bar(
                xs + offset,
                plot_ready(values),
                width=width,
                color=spec.color,
                edgecolor=EDGE_COLOR,
                linewidth=0.6,
                label=spec.axis_label,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(SETUP_TITLES[setup], fontsize=10, pad=4)
        setup_ax(ax, (0.0, 1.0))
    axes[0].set_ylabel("Rate")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "anthropic_model_accuracy_by_setup", generated)


def plot_comparison_over_under_by_setup(stats: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path, generated: List[Path]) -> None:
    metrics = ["under", "over"]
    labels = ["Under-severity", "Over-severity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=300, sharey=True)

    for ax, setup in zip(axes, SETUP_ORDER):
        xs = np.arange(len(metrics))
        width = 0.22
        offsets = np.linspace(-width, width, len(FAMILY_ORDER))
        for offset, family_key in zip(offsets, FAMILY_ORDER):
            spec = FAMILY_SPECS[family_key]
            values = [stats[setup][family_key][metric] for metric in metrics]
            ax.bar(
                xs + offset,
                plot_ready(values),
                width=width,
                color=spec.color,
                edgecolor=EDGE_COLOR,
                linewidth=0.6,
                label=spec.axis_label,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(SETUP_TITLES[setup], fontsize=10, pad=4)
        setup_ax(ax, (0.0, 1.0))
    axes[0].set_ylabel("Rate")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "anthropic_model_over_under_by_setup", generated)


def plot_comparison_committee_gain(stats: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path, generated: List[Path]) -> None:
    perf_metrics = ["exact", "type", "severity"]
    perf_labels = ["Exact", "Type", "Severity"]
    calib_metrics = ["under", "over"]
    calib_labels = ["Under", "Over"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=300)

    xs = np.arange(len(FAMILY_ORDER))
    width = 0.22
    perf_offsets = np.linspace(-width, width, len(perf_metrics))
    calib_offsets = np.linspace(-width / 2, width / 2, len(calib_metrics))

    for offset, metric, label in zip(perf_offsets, perf_metrics, perf_labels):
        values = [stats["committee"][family_key][metric] - stats["single"][family_key][metric] for family_key in FAMILY_ORDER]
        axes[0].bar(xs + offset, plot_ready(values), width=width, edgecolor=EDGE_COLOR, linewidth=0.6, label=label)
    axes[0].axhline(0, color="#222222", linewidth=1.2)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([FAMILY_SPECS[key].axis_label for key in FAMILY_ORDER])
    axes[0].set_title("Committee Minus Single\nAccuracy", fontsize=10, pad=4)
    axes[0].set_ylabel("Delta")
    axes[0].grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axes[0].set_axisbelow(True)

    for offset, metric, label in zip(calib_offsets, calib_metrics, calib_labels):
        values = [stats["committee"][family_key][metric] - stats["single"][family_key][metric] for family_key in FAMILY_ORDER]
        axes[1].bar(xs + offset, plot_ready(values), width=width, edgecolor=EDGE_COLOR, linewidth=0.6, label=label)
    axes[1].axhline(0, color="#222222", linewidth=1.2)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([FAMILY_SPECS[key].axis_label for key in FAMILY_ORDER])
    axes[1].set_title("Committee Minus Single\nCalibration", fontsize=10, pad=4)
    axes[1].grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axes[1].set_axisbelow(True)

    color_cycle = ["#777777", "#BBBBBB", "#444444"]
    for index, bar in enumerate(axes[0].patches):
        bar.set_facecolor(color_cycle[index % len(perf_metrics)])
    for index, bar in enumerate(axes[1].patches):
        bar.set_facecolor(color_cycle[index % len(calib_metrics)])

    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, out_dir, "anthropic_model_committee_gain", generated)


def build_comparison_matrix(runs: Sequence[RunEntry], setup: str, metric: str) -> np.ndarray:
    matrix = np.full((len(SCENARIO_ORDER), len(FAMILY_ORDER)), np.nan, dtype=float)
    for row_index, scenario_key in enumerate(SCENARIO_ORDER):
        for col_index, family_key in enumerate(FAMILY_ORDER):
            values = [
                run_metric_value(run, metric)
                for run in runs
                if run.setup == setup and run.family_key == family_key and run.scenario_key == scenario_key
            ]
            matrix[row_index, col_index] = mean_optional(values)
    return matrix


def plot_comparison_scenario_exact_heatmap(runs: Sequence[RunEntry], out_dir: Path, generated: List[Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), dpi=300, sharey=True)
    matrices = [build_comparison_matrix(runs, setup, "exact") for setup in SETUP_ORDER]

    for ax, setup, matrix in zip(axes, SETUP_ORDER, matrices):
        masked, cmap = masked_image(matrix)
        image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(FAMILY_ORDER)))
        ax.set_xticklabels([FAMILY_SPECS[key].axis_label for key in FAMILY_ORDER])
        ax.set_yticks(np.arange(len(SCENARIO_ORDER)))
        ax.set_yticklabels([SCENARIO_TITLES[key] for key in SCENARIO_ORDER], fontsize=8)
        ax.set_title(f"{SETUP_TITLES[setup]}\nExact correctness", fontsize=10, pad=4)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                text = fmt_pct(value)
                color = "black" if math.isnan(value) or value >= 0.5 else "white"
                ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8, color=color)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label("Rate", fontsize=8)
    fig.subplots_adjust(wspace=0.28, right=0.88)
    save(fig, out_dir, "anthropic_model_scenario_exact_heatmap", generated, use_tight_layout=False)


def write_comparison_table(stats: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path, generated: List[Path]) -> None:
    headers = ["Setup", "Model", "Exact", "Type", "Severity", "Under", "Over", "Wrong", "Severity Bias"]
    rows: List[List[str]] = []
    for setup in SETUP_ORDER:
        for family_key in FAMILY_ORDER:
            spec = FAMILY_SPECS[family_key]
            row = stats[setup][family_key]
            rows.append(
                [
                    SETUP_TITLES[setup],
                    spec.display_name,
                    fmt_pct(row["exact"]),
                    fmt_pct(row["type"]),
                    fmt_pct(row["severity"]),
                    fmt_pct(row["under"]),
                    fmt_pct(row["over"]),
                    fmt_pct(row["wrong"]),
                    fmt_float(row["severity_bias"]),
                ]
            )
    render_table(headers, rows, out_dir, "anthropic_model_summary_table", generated)


def build_opus_delta_matrix(runs: Sequence[RunEntry], setup: str) -> np.ndarray:
    metrics = ["exact", "type", "severity", "wrong"]
    matrix = np.full((len(SCENARIO_ORDER), len(metrics)), np.nan, dtype=float)
    for row_index, scenario_key in enumerate(SCENARIO_ORDER):
        opus46 = [
            run
            for run in runs
            if run.setup == setup and run.family_key == "opus46" and run.scenario_key == scenario_key
        ]
        opus47 = [
            run
            for run in runs
            if run.setup == setup and run.family_key == "opus47" and run.scenario_key == scenario_key
        ]
        for col_index, metric in enumerate(metrics):
            left = mean_optional(run_metric_value(run, metric) for run in opus46)
            right = mean_optional(run_metric_value(run, metric) for run in opus47)
            if math.isnan(left) or math.isnan(right):
                matrix[row_index, col_index] = float("nan")
            else:
                matrix[row_index, col_index] = right - left
    return matrix


def plot_opus_delta_heatmap(runs: Sequence[RunEntry], out_dir: Path, generated: List[Path]) -> None:
    metrics = ["Exact", "Type", "Severity", "Wrong"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), dpi=300, sharey=True)

    for ax, setup in zip(axes, SETUP_ORDER):
        matrix = build_opus_delta_matrix(runs, setup)
        masked, cmap = masked_image(matrix, "RdBu_r")
        image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0))
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_yticks(np.arange(len(SCENARIO_ORDER)))
        ax.set_yticklabels([SCENARIO_TITLES[key] for key in SCENARIO_ORDER], fontsize=8)
        ax.set_title(f"{SETUP_TITLES[setup]}\nOpus 4.7 minus Opus 4.6", fontsize=10, pad=4)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                text = fmt_delta(value)
                color = "black" if math.isnan(value) or abs(value) < 0.35 else "white"
                ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8, color=color)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label("Delta", fontsize=8)
    fig.subplots_adjust(wspace=0.28, right=0.88)
    save(fig, out_dir, "anthropic_opus46_vs_opus47_delta", generated, use_tight_layout=False)


def generate_comparison_plots(runs: Sequence[RunEntry], plots_root: Path) -> List[Path]:
    out_dir = plots_root / "anthropic_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    stats = aggregate_model_setup(runs)

    plot_comparison_accuracy_by_setup(stats, out_dir, generated)
    plot_comparison_over_under_by_setup(stats, out_dir, generated)
    plot_comparison_committee_gain(stats, out_dir, generated)
    plot_comparison_scenario_exact_heatmap(runs, out_dir, generated)
    write_comparison_table(stats, out_dir, generated)
    plot_opus_delta_heatmap(runs, out_dir, generated)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Sonnet, Opus 4.6, and Opus 4.7 cybersecurity thesis plots plus comparison figures."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Cyber output directory.")
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR, help="Ground-truth directory.")
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOTS_ROOT, help="Destination root for generated PNGs.")
    args = parser.parse_args()

    apply_thesis_style(font_size=11, y_grid=True)
    args.plots_root.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args.output_root, args.ground_truth_dir, family_keys=FAMILY_ORDER)
    if not runs:
        raise SystemExit(f"No completed Sonnet/Opus runs found in {args.output_root}")

    generated: List[Path] = []
    for family_key in FAMILY_ORDER:
        family_runs = [run for run in runs if run.family_key == family_key]
        if not family_runs:
            continue
        generated.extend(generate_family_plots(FAMILY_SPECS[family_key], family_runs, args.plots_root))
    generated.extend(generate_comparison_plots(runs, args.plots_root))

    print(f"Generated {len(generated)} anthropic cybersecurity plot files in {args.plots_root}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
