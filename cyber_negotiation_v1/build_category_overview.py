from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cyber_utils import aggregate_condition_results


CONDITION_META: Dict[str, Dict[str, str]] = {
    "C1": {"label": "Single GPT-5", "family": "GPT-5", "mode": "baseline"},
    "C2": {"label": "Single Claude", "family": "Claude", "mode": "baseline"},
    "C3": {"label": "3x GPT-5", "family": "GPT-5", "mode": "negotiation"},
    "C4": {"label": "3x Claude", "family": "Claude", "mode": "negotiation"},
    "C5": {"label": "Mixed Committee", "family": "Mixed", "mode": "negotiation"},
    "C6": {"label": "3x GPT-5 + LLM Prior", "family": "GPT-5", "mode": "negotiation"},
    "C7": {"label": "3x Claude + Human Prior", "family": "Claude", "mode": "negotiation"},
}

CATEGORY_TITLES = {
    "command_injection": "Command Injection",
    "cookies": "Cookies",
    "csrf": "CSRF",
    "path_disclosure": "Path Disclosure",
}

SCENARIO_TITLES = {
    "ping_form_exec_output_001": "Ping Form Command Injection",
    "cookie_security_attribute_observation_001": "Cookie Flags Observation",
    "hard_cookie_md5_002": "MD5-Like Cookie Pattern",
    "info_apache": "Apache Header Disclosure",
    "medium_cookie_timestamps_001": "Timestamp Cookie Pattern",
    "reflected_input_password_change_guard_001": "Reflected Input Password Change Guard",
    "error_message_path_disclosure_001": "Error Message Path Disclosure",
}


@dataclass
class RunEntry:
    category: str
    scenario_id: str
    condition_id: str
    run_id: str
    run_dir: Path
    metrics_path: Path
    history_path: Path
    report_path: Path
    run_report: Dict[str, Any]
    history: Dict[str, Any]
    llm_eval: Optional[Dict[str, Any]]
    public_turns: int
    type_transitions: int
    exact_transitions: int
    type_states: int
    exact_states: int


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_sort_key(condition_id: str) -> int:
    return int(condition_id[1:]) if len(condition_id) > 1 and condition_id[1:].isdigit() else 999


def scenario_title(scenario_id: str) -> str:
    if scenario_id in SCENARIO_TITLES:
        return SCENARIO_TITLES[scenario_id]
    parts = [part for part in scenario_id.split("_") if part and not part.isdigit()]
    return " ".join(part.capitalize() for part in parts) or scenario_id


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return mean(numeric)


def committee_type_value(snapshot: Dict[str, Any]) -> str:
    value = snapshot.get("committee_type_label") or snapshot.get("committee_type")
    return str(value) if value is not None else "None"


def committee_exact_value(snapshot: Dict[str, Any]) -> str:
    value = snapshot.get("committee_exact")
    if isinstance(value, dict):
        label = str(value.get("label", "")).strip()
        severity = str(value.get("severity", "")).strip()
        if label or severity:
            return f"{label}/{severity}"
    exact_label = snapshot.get("committee_exact_label")
    exact_severity = snapshot.get("committee_exact_severity")
    if exact_label or exact_severity:
        return f"{exact_label}/{exact_severity}"
    return "None"


def count_transitions(values: Sequence[str]) -> int:
    if not values:
        return 0
    transitions = 0
    for left, right in zip(values, values[1:]):
        if left != right:
            transitions += 1
    return transitions


def load_llm_eval_map(output_root: Path) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], Optional[Path]]:
    eval_dir = output_root / "llm_evaluator"
    if not eval_dir.exists():
        return {}, None
    candidates = sorted(eval_dir.glob("llm_eval_per_run_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return {}, None
    payload = load_json(candidates[0])
    records: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for item in list(payload.get("runs") or []):
        key = (str(item.get("scenario_id", "")), str(item.get("condition_id", "")), str(item.get("run_id", "")))
        if all(key):
            records[key] = item
    return records, candidates[0]


def scan_runs(output_root: Path) -> Tuple[List[RunEntry], Optional[Path]]:
    llm_eval_map, llm_eval_source = load_llm_eval_map(output_root)
    latest_by_key: Dict[Tuple[str, str, str], RunEntry] = {}

    for metrics_path in sorted(output_root.rglob("metrics_*.json")):
        relative_parts = metrics_path.relative_to(output_root).parts
        if not relative_parts or relative_parts[0] == "llm_evaluator":
            continue

        payload = load_json(metrics_path)
        run_report = dict(payload.get("run_report") or {})
        if not run_report:
            continue

        category = relative_parts[0]
        scenario_id = str(run_report.get("scenario_id") or "")
        condition_id = str(run_report.get("condition_id") or "")
        run_id = str(run_report.get("run_id") or metrics_path.stem.replace("metrics_", "", 1))
        if not category or not scenario_id or not condition_id or not run_id:
            continue

        run_dir = metrics_path.parent
        history_path = run_dir / f"{run_id}.json"
        report_path = run_dir / f"report_{run_id}.html"
        if not history_path.exists():
            continue

        history = dict(load_json(history_path) or {})
        if history.get("run_status") != "completed":
            continue

        trajectory = list(run_report.get("decision_trajectory") or history.get("decision_trajectory") or [])
        type_path = [committee_type_value(snapshot) for snapshot in trajectory]
        exact_path = [committee_exact_value(snapshot) for snapshot in trajectory]
        key = (category, scenario_id, condition_id)

        entry = RunEntry(
            category=category,
            scenario_id=scenario_id,
            condition_id=condition_id,
            run_id=run_id,
            run_dir=run_dir,
            metrics_path=metrics_path,
            history_path=history_path,
            report_path=report_path,
            run_report=run_report,
            history=history,
            llm_eval=llm_eval_map.get((scenario_id, condition_id, run_id)),
            public_turns=len(list(history.get("rounds") or [])),
            type_transitions=count_transitions(type_path),
            exact_transitions=count_transitions(exact_path),
            type_states=len(set(type_path)) if type_path else 0,
            exact_states=len(set(exact_path)) if exact_path else 0,
        )

        current = latest_by_key.get(key)
        if current is None or metrics_path.stat().st_mtime > current.metrics_path.stat().st_mtime:
            latest_by_key[key] = entry

    runs = sorted(
        latest_by_key.values(),
        key=lambda item: (item.category, item.scenario_id, condition_sort_key(item.condition_id)),
    )
    return runs, llm_eval_source


def aggregate_entries(entries: Sequence[RunEntry], label: str) -> Dict[str, Any]:
    reports = [entry.run_report for entry in entries]
    aggregate = aggregate_condition_results(reports, condition_id=label)
    aggregate["extras"] = {
        "public_turns_mean": mean_optional([float(entry.public_turns) for entry in entries]),
        "type_transitions_mean": mean_optional([float(entry.type_transitions) for entry in entries]),
        "exact_transitions_mean": mean_optional([float(entry.exact_transitions) for entry in entries]),
        "type_states_mean": mean_optional([float(entry.type_states) for entry in entries]),
        "exact_states_mean": mean_optional([float(entry.exact_states) for entry in entries]),
        "scenario_count": len({entry.scenario_id for entry in entries}),
    }
    return aggregate


def get_metric(aggregate: Dict[str, Any], section: str, key: str) -> Optional[float]:
    return as_float((aggregate.get(section) or {}).get(key))


def metric_bundle(aggregate: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "exact": get_metric(aggregate, "headline_metrics", "FinalCorrectExact"),
        "type": get_metric(aggregate, "headline_metrics", "FinalCorrectType"),
        "agreement": get_metric(aggregate, "headline_metrics", "FinalAgreementExact"),
        "any_agreement": get_metric(aggregate, "headline_metrics", "AnyAgreementExact"),
        "bias": get_metric(aggregate, "headline_metrics", "SeverityBias"),
        "trust": get_metric(aggregate, "headline_metrics", "TrustHygieneRate"),
        "wrong": get_metric(aggregate, "derived_metrics", "WrongConsensusExactRate"),
        "false_agreement": get_metric(aggregate, "derived_metrics", "FalseAgreementWithoutSignoffExactRate"),
        "late_drift": get_metric(aggregate, "derived_metrics", "LateDriftAgreementExactRate"),
        "any_correct_type": get_metric(aggregate, "derived_metrics", "AnyCorrectConsensusTypeRate"),
        "final_type_agreement": get_metric(aggregate, "derived_metrics", "FinalAgreementTypeRate"),
        "any_type_agreement": get_metric(aggregate, "derived_metrics", "AnyAgreementTypeRate"),
        "over": get_metric(aggregate, "derived_metrics", "OverSeverityRate"),
        "under": get_metric(aggregate, "derived_metrics", "UnderSeverityRate"),
        "latency": get_metric(aggregate, "derived_metrics", "ConsensusLatencyExactMean"),
        "type_transitions": get_metric(aggregate, "extras", "type_transitions_mean"),
        "exact_transitions": get_metric(aggregate, "extras", "exact_transitions_mean"),
        "public_turns": get_metric(aggregate, "extras", "public_turns_mean"),
    }


def pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def signed_num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def svg_rate_chart(
    items: Sequence[Tuple[str, Optional[float]]],
    title: str,
    *,
    color: str,
    width: int = 520,
    height: int = 240,
) -> str:
    data = [(label, value) for label, value in items if value is not None]
    if not data:
        return f"<p class='muted'>No chart data for {html.escape(title)}.</p>"

    pad = 36
    chart_w = width - 2 * pad
    chart_h = height - 2 * pad
    gap = 12
    bar_w = max(20, (chart_w - gap * (len(data) - 1)) / len(data))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{html.escape(title)}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="transparent"/>',
        f'<text x="{pad}" y="20" font-size="14" font-weight="700" fill="#1f2937">{html.escape(title)}</text>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + chart_h}" stroke="#64748b" stroke-width="1"/>',
        f'<line x1="{pad}" y1="{pad + chart_h}" x2="{pad + chart_w}" y2="{pad + chart_h}" stroke="#64748b" stroke-width="1"/>',
    ]
    for tick in range(0, 5):
        ratio = tick / 4
        y = pad + chart_h - ratio * chart_h
        parts.append(f'<line x1="{pad}" y1="{y:.2f}" x2="{pad + chart_w}" y2="{y:.2f}" stroke="rgba(100,116,139,0.18)" stroke-width="1"/>')
        parts.append(f'<text x="{pad - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="10" fill="#64748b">{ratio:.2f}</text>')

    for idx, (label, value) in enumerate(data):
        x = pad + idx * (bar_w + gap)
        bar_h = max(0.0, min(1.0, float(value))) * chart_h
        y = pad + chart_h - bar_h
        short = label if len(label) <= 10 else label[:9] + "…"
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" rx="7" fill="{color}" opacity="0.88"/>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-size="10" fill="#334155">{value:.2f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{pad + chart_h + 16:.2f}" text-anchor="middle" font-size="10" fill="#475569">{html.escape(short)}</text>')

    parts.append("</svg>")
    return "".join(parts)


def svg_bias_chart(
    items: Sequence[Tuple[str, Optional[float]]],
    title: str,
    width: int = 520,
    height: int = 240,
) -> str:
    data = [(label, value) for label, value in items if value is not None]
    if not data:
        return f"<p class='muted'>No chart data for {html.escape(title)}.</p>"

    max_abs = max(abs(float(value)) for _, value in data) or 1.0
    pad = 36
    chart_w = width - 2 * pad
    chart_h = height - 2 * pad
    half_h = chart_h / 2
    baseline_y = pad + half_h
    gap = 12
    bar_w = max(20, (chart_w - gap * (len(data) - 1)) / len(data))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{html.escape(title)}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="transparent"/>',
        f'<text x="{pad}" y="20" font-size="14" font-weight="700" fill="#1f2937">{html.escape(title)}</text>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + chart_h}" stroke="#64748b" stroke-width="1"/>',
        f'<line x1="{pad}" y1="{baseline_y:.2f}" x2="{pad + chart_w}" y2="{baseline_y:.2f}" stroke="#64748b" stroke-width="1.2"/>',
    ]
    for tick in (-max_abs, 0.0, max_abs):
        offset = (tick / max_abs) * half_h if max_abs else 0.0
        y = baseline_y - offset
        parts.append(f'<text x="{pad - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="10" fill="#64748b">{tick:+.1f}</text>')

    for idx, (label, value) in enumerate(data):
        number = float(value)
        x = pad + idx * (bar_w + gap)
        bar_h = abs(number) / max_abs * half_h
        y = baseline_y - bar_h if number >= 0 else baseline_y
        color = "#dc2626" if number > 0 else ("#2563eb" if number < 0 else "#64748b")
        short = label if len(label) <= 10 else label[:9] + "…"
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" rx="7" fill="{color}" opacity="0.84"/>')
        label_y = y - 6 if number >= 0 else y + bar_h + 14
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{label_y:.2f}" text-anchor="middle" font-size="10" fill="#334155">{number:+.2f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{pad + chart_h + 16:.2f}" text-anchor="middle" font-size="10" fill="#475569">{html.escape(short)}</text>')

    parts.append("</svg>")
    return "".join(parts)


def comparison_score(reference: Dict[str, Optional[float]], candidate: Dict[str, Optional[float]], trust_focus: bool = False) -> int:
    score = 0

    def add_if(delta: Optional[float], threshold: float, weight: int) -> None:
        nonlocal score
        if delta is None:
            return
        if delta > threshold:
            score += weight
        elif delta < -threshold:
            score -= weight

    exact_delta = None if reference["exact"] is None or candidate["exact"] is None else candidate["exact"] - reference["exact"]
    type_delta = None if reference["type"] is None or candidate["type"] is None else candidate["type"] - reference["type"]
    wrong_delta = None if reference["wrong"] is None or candidate["wrong"] is None else reference["wrong"] - candidate["wrong"]
    drift_delta = None if reference["late_drift"] is None or candidate["late_drift"] is None else reference["late_drift"] - candidate["late_drift"]
    trust_delta = None if reference["trust"] is None or candidate["trust"] is None else reference["trust"] - candidate["trust"]
    bias_delta = None
    if reference["bias"] is not None and candidate["bias"] is not None:
        bias_delta = abs(reference["bias"]) - abs(candidate["bias"])

    add_if(exact_delta, 0.05, 3 if not trust_focus else 2)
    add_if(type_delta, 0.05, 2 if not trust_focus else 1)
    add_if(wrong_delta, 0.05, 2 if not trust_focus else 3)
    add_if(drift_delta, 0.05, 1 if not trust_focus else 2)
    add_if(trust_delta, 0.05, 1 if not trust_focus else 2)
    add_if(bias_delta, 0.25, 1)
    return score


def verdict_from_score(score: int) -> str:
    if score >= 4:
        return "Better"
    if score <= -4:
        return "Worse"
    if -1 <= score <= 1:
        return "No Clear Change"
    return "Mixed"


def summarize_comparison(
    title: str,
    reference_aggregate: Optional[Dict[str, Any]],
    candidate_aggregate: Optional[Dict[str, Any]],
    *,
    trust_focus: bool = False,
) -> Dict[str, Any]:
    if reference_aggregate is None or candidate_aggregate is None:
        return {
            "title": title,
            "status": "Insufficient Data",
            "summary": "At least one side of the comparison is missing completed runs.",
            "evidence": [],
        }

    reference = metric_bundle(reference_aggregate)
    candidate = metric_bundle(candidate_aggregate)
    score = comparison_score(reference, candidate, trust_focus=trust_focus)
    status = verdict_from_score(score)

    if trust_focus:
        summary = (
            "Trust-related metrics do not show a clear advantage."
            if status == "No Clear Change"
            else (
                "Trust-related metrics improve on the candidate side."
                if status == "Better"
                else (
                    "Trust-related metrics degrade on the candidate side."
                    if status == "Worse"
                    else "Correctness and trust metrics move in different directions."
                )
            )
        )
    else:
        summary = (
            "Outcome quality is materially better on the candidate side."
            if status == "Better"
            else (
                "Outcome quality is materially worse on the candidate side."
                if status == "Worse"
                else (
                    "Outcome quality is broadly unchanged."
                    if status == "No Clear Change"
                    else "The comparison shows tradeoffs rather than a clean improvement."
                )
            )
        )

    evidence = [
        f"Final exact correctness: {pct(reference['exact'])} -> {pct(candidate['exact'])}",
        f"Final type correctness: {pct(reference['type'])} -> {pct(candidate['type'])}",
        f"Wrong consensus rate: {pct(reference['wrong'])} -> {pct(candidate['wrong'])}",
        f"Trust hygiene rate: {pct(reference['trust'])} -> {pct(candidate['trust'])}",
        f"Severity bias: {signed_num(reference['bias'])} -> {signed_num(candidate['bias'])}",
    ]
    return {"title": title, "status": status, "summary": summary, "evidence": evidence}


def summarize_instruction_priors(
    gpt_base: Optional[Dict[str, Any]],
    gpt_prior: Optional[Dict[str, Any]],
    claude_base: Optional[Dict[str, Any]],
    claude_prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if gpt_base is None or gpt_prior is None or claude_base is None or claude_prior is None:
        return {
            "title": "Instruction-Only Priors",
            "status": "Insufficient Data",
            "summary": "The prior/non-prior comparison is incomplete.",
            "evidence": [],
        }

    gpt_reference = metric_bundle(gpt_base)
    gpt_candidate = metric_bundle(gpt_prior)
    claude_reference = metric_bundle(claude_base)
    claude_candidate = metric_bundle(claude_prior)

    gpt_score = comparison_score(gpt_reference, gpt_candidate, trust_focus=False)
    claude_score = comparison_score(claude_reference, claude_candidate, trust_focus=False)

    if abs(gpt_score) <= 1 and abs(claude_score) <= 1:
        status = "Little or No Effect"
        summary = "Changing only the prior framing does not materially alter quality or stability in this category."
    elif gpt_score >= 4 and claude_score >= 4:
        status = "Consistent Improvement"
        summary = "Both prior framings improve quality or stability relative to the matching no-prior committees."
    elif gpt_score <= -4 and claude_score <= -4:
        status = "Consistent Degradation"
        summary = "Both prior framings degrade quality or stability relative to the matching no-prior committees."
    else:
        status = "Observable but Inconsistent"
        summary = "The prior framing changes committee behavior, but the effect is category-dependent rather than uniformly positive."

    evidence = [
        f"GPT prior (C3 -> C6): exact {pct(gpt_reference['exact'])} -> {pct(gpt_candidate['exact'])}, type {pct(gpt_reference['type'])} -> {pct(gpt_candidate['type'])}, wrong consensus {pct(gpt_reference['wrong'])} -> {pct(gpt_candidate['wrong'])}",
        f"Claude prior (C4 -> C7): exact {pct(claude_reference['exact'])} -> {pct(claude_candidate['exact'])}, type {pct(claude_reference['type'])} -> {pct(claude_candidate['type'])}, wrong consensus {pct(claude_reference['wrong'])} -> {pct(claude_candidate['wrong'])}",
        f"Stability proxy (exact transitions): GPT {num(gpt_reference['exact_transitions'])} -> {num(gpt_candidate['exact_transitions'])}, Claude {num(claude_reference['exact_transitions'])} -> {num(claude_candidate['exact_transitions'])}",
    ]
    return {"title": "Instruction-Only Priors", "status": status, "summary": summary, "evidence": evidence}


def summarize_c7_expectation(
    c2: Optional[Dict[str, Any]],
    c4: Optional[Dict[str, Any]],
    c7: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if c2 is None or c4 is None or c7 is None:
        return {
            "title": "C7 Expectation",
            "status": "Insufficient Data",
            "summary": "Cannot compare Claude conditions cleanly.",
            "evidence": [],
        }

    c2_metrics = metric_bundle(c2)
    c4_metrics = metric_bundle(c4)
    c7_metrics = metric_bundle(c7)
    best_reference_exact = max(value for value in [c2_metrics["exact"], c4_metrics["exact"]] if value is not None)

    if c7_metrics["exact"] is not None and c7_metrics["exact"] > best_reference_exact + 0.05:
        status = "Supported"
        summary = "C7 clearly improves exact correctness over the other Claude settings in this category."
    elif c7_metrics["exact"] is not None and abs(c7_metrics["exact"] - best_reference_exact) <= 0.05:
        status = "Not Clearly Supported"
        summary = "C7 matches the best Claude setup, but it does not produce a clear improvement."
    else:
        status = "Not Supported"
        summary = "C7 underperforms the best Claude alternative on the available outcome metrics."

    evidence = [
        f"C2 exact/type: {pct(c2_metrics['exact'])} / {pct(c2_metrics['type'])}",
        f"C4 exact/type: {pct(c4_metrics['exact'])} / {pct(c4_metrics['type'])}",
        f"C7 exact/type: {pct(c7_metrics['exact'])} / {pct(c7_metrics['type'])}",
        f"C7 wrong consensus / bias: {pct(c7_metrics['wrong'])} / {signed_num(c7_metrics['bias'])}",
    ]
    return {"title": "C7 Expectation", "status": status, "summary": summary, "evidence": evidence}


def evaluate_hypotheses(condition_aggregates: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_entries = [condition_aggregates[key] for key in sorted(condition_aggregates)]
    baseline = [condition_aggregates[key] for key in ("C1", "C2") if key in condition_aggregates]
    negotiation = [condition_aggregates[key] for key in ("C3", "C4", "C5", "C6", "C7") if key in condition_aggregates]

    def combine(items: Sequence[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
        if not items:
            return None
        return {
            "headline_metrics": {
                "FinalCorrectExact": mean_optional([get_metric(item, "headline_metrics", "FinalCorrectExact") for item in items]),
                "FinalCorrectType": mean_optional([get_metric(item, "headline_metrics", "FinalCorrectType") for item in items]),
                "FinalAgreementExact": mean_optional([get_metric(item, "headline_metrics", "FinalAgreementExact") for item in items]),
                "AnyAgreementExact": mean_optional([get_metric(item, "headline_metrics", "AnyAgreementExact") for item in items]),
                "SeverityBias": mean_optional([get_metric(item, "headline_metrics", "SeverityBias") for item in items]),
                "TrustHygieneRate": mean_optional([get_metric(item, "headline_metrics", "TrustHygieneRate") for item in items]),
            },
            "derived_metrics": {
                "WrongConsensusExactRate": mean_optional([get_metric(item, "derived_metrics", "WrongConsensusExactRate") for item in items]),
                "FalseAgreementWithoutSignoffExactRate": mean_optional(
                    [get_metric(item, "derived_metrics", "FalseAgreementWithoutSignoffExactRate") for item in items]
                ),
                "LateDriftAgreementExactRate": mean_optional(
                    [get_metric(item, "derived_metrics", "LateDriftAgreementExactRate") for item in items]
                ),
                "AnyCorrectConsensusTypeRate": mean_optional(
                    [get_metric(item, "derived_metrics", "AnyCorrectConsensusTypeRate") for item in items]
                ),
                "ConsensusLatencyExactMean": mean_optional(
                    [get_metric(item, "derived_metrics", "ConsensusLatencyExactMean") for item in items]
                ),
                "FinalAgreementTypeRate": mean_optional(
                    [get_metric(item, "derived_metrics", "FinalAgreementTypeRate") for item in items]
                ),
                "AnyAgreementTypeRate": mean_optional(
                    [get_metric(item, "derived_metrics", "AnyAgreementTypeRate") for item in items]
                ),
                "OverSeverityRate": mean_optional([get_metric(item, "derived_metrics", "OverSeverityRate") for item in items]),
                "UnderSeverityRate": mean_optional([get_metric(item, "derived_metrics", "UnderSeverityRate") for item in items]),
            },
            "extras": {
                "type_transitions_mean": mean_optional([get_metric(item, "extras", "type_transitions_mean") for item in items]),
                "exact_transitions_mean": mean_optional([get_metric(item, "extras", "exact_transitions_mean") for item in items]),
            },
            "label": label,
        }

    baseline_bundle = combine(baseline, "baseline")
    negotiation_bundle = combine(negotiation, "negotiation")
    overall_bundle = combine(all_entries, "overall")

    hypotheses: List[Dict[str, Any]] = []

    if baseline_bundle and negotiation_bundle:
        baseline_metrics = metric_bundle(baseline_bundle)
        negotiation_metrics = metric_bundle(negotiation_bundle)
        agreement_gain = None
        wrong_gain = None
        if baseline_metrics["agreement"] is not None and negotiation_metrics["agreement"] is not None:
            agreement_gain = negotiation_metrics["agreement"] - baseline_metrics["agreement"]
        if baseline_metrics["wrong"] is not None and negotiation_metrics["wrong"] is not None:
            wrong_gain = negotiation_metrics["wrong"] - baseline_metrics["wrong"]

        if agreement_gain is not None and agreement_gain > 0.05 and wrong_gain is not None and wrong_gain > 0.05:
            status = "Supported"
        elif agreement_gain is not None and agreement_gain > 0.05:
            status = "Partially Supported"
        else:
            status = "Not Supported"

        hypotheses.append(
            {
                "title": "H1 — Agreement vs Correctness Tradeoff",
                "status": status,
                "summary": (
                    f"Baseline final agreement {pct(baseline_metrics['agreement'])} vs negotiation {pct(negotiation_metrics['agreement'])}; "
                    f"baseline wrong-consensus {pct(baseline_metrics['wrong'])} vs negotiation {pct(negotiation_metrics['wrong'])}."
                ),
            }
        )

        type_gap = None
        exact_gap = None
        if negotiation_metrics["type_transitions"] is not None and negotiation_metrics["exact_transitions"] is not None:
            type_gap = negotiation_metrics["type_transitions"]
            exact_gap = negotiation_metrics["exact_transitions"]
        if exact_gap is not None and type_gap is not None and exact_gap > type_gap + 0.25:
            status = "Supported"
        elif exact_gap is not None and type_gap is not None and exact_gap > type_gap:
            status = "Partially Supported"
        else:
            status = "Not Supported"

        hypotheses.append(
            {
                "title": "H2 — Severity Is Less Stable Than Type",
                "status": status,
                "summary": (
                    f"Negotiation mean type transitions {num(type_gap)} vs exact transitions {num(exact_gap)}; "
                    f"final type agreement {pct(negotiation_metrics['final_type_agreement'])} vs final exact agreement {pct(negotiation_metrics['agreement'])}."
                ),
            }
        )
    else:
        hypotheses.append(
            {
                "title": "H1 — Agreement vs Correctness Tradeoff",
                "status": "Insufficient Data",
                "summary": "Baseline and negotiation groups are both required.",
            }
        )
        hypotheses.append(
            {
                "title": "H2 — Severity Is Less Stable Than Type",
                "status": "Insufficient Data",
                "summary": "Negotiation-group trajectory metrics are required.",
            }
        )

    if overall_bundle:
        overall_metrics = metric_bundle(overall_bundle)
        if (
            overall_metrics["over"] is not None
            and overall_metrics["under"] is not None
            and overall_metrics["bias"] is not None
            and overall_metrics["over"] > overall_metrics["under"] + 0.05
            and overall_metrics["bias"] > 0.25
        ):
            status = "Supported"
        elif overall_metrics["bias"] is not None and overall_metrics["bias"] > 0:
            status = "Partially Supported"
        else:
            status = "Not Supported"

        hypotheses.append(
            {
                "title": "H3 — Over-Severity Tendency",
                "status": status,
                "summary": (
                    f"Mean severity bias {signed_num(overall_metrics['bias'])}; "
                    f"over-severity rate {pct(overall_metrics['over'])}; under-severity rate {pct(overall_metrics['under'])}."
                ),
            }
        )
    else:
        hypotheses.append(
            {
                "title": "H3 — Over-Severity Tendency",
                "status": "Insufficient Data",
                "summary": "No aggregate category metrics available.",
            }
        )

    c3 = condition_aggregates.get("C3")
    c4 = condition_aggregates.get("C4")
    c6 = condition_aggregates.get("C6")
    c7 = condition_aggregates.get("C7")
    if c3 and c4 and c6 and c7:
        prior_summary = summarize_instruction_priors(c3, c6, c4, c7)
        status = "Supported" if prior_summary["status"] in {"Consistent Improvement", "Observable but Inconsistent"} else "Not Supported"
        hypotheses.append(
            {
                "title": "H4 — Prior-Knowledge Effect",
                "status": status,
                "summary": prior_summary["summary"],
            }
        )
    else:
        hypotheses.append(
            {
                "title": "H4 — Prior-Knowledge Effect",
                "status": "Insufficient Data",
                "summary": "Need both prior and non-prior committee runs.",
            }
        )

    return hypotheses


def category_synopsis(category: str, condition_aggregates: Dict[str, Dict[str, Any]]) -> str:
    exact_values = [
        get_metric(aggregate, "headline_metrics", "FinalCorrectExact")
        for aggregate in condition_aggregates.values()
        if get_metric(aggregate, "headline_metrics", "FinalCorrectExact") is not None
    ]
    bias_values = [
        get_metric(aggregate, "headline_metrics", "SeverityBias")
        for aggregate in condition_aggregates.values()
        if get_metric(aggregate, "headline_metrics", "SeverityBias") is not None
    ]
    wrong_values = [
        get_metric(aggregate, "derived_metrics", "WrongConsensusExactRate")
        for aggregate in condition_aggregates.values()
        if get_metric(aggregate, "derived_metrics", "WrongConsensusExactRate") is not None
    ]

    if exact_values and min(exact_values) == 1.0 and max(exact_values) == 1.0:
        outcome_line = "This is a ceiling case: every condition reaches the exact ground-truth outcome."
    elif exact_values and min(exact_values) == 0.0 and max(exact_values) == 0.0:
        outcome_line = "This is a uniformly hard category: every condition converges to the wrong exact outcome."
    else:
        outcome_line = (
            f"Outcome quality is discriminative here: exact correctness ranges from {pct(min(exact_values))} to {pct(max(exact_values))}."
            if exact_values
            else "Outcome quality is unavailable."
        )

    if bias_values:
        mean_bias = mean(abs(value) for value in bias_values)
        if mean(bias_values) > 0.25:
            bias_line = f"Severity tends to run hot in this category (mean bias {signed_num(mean(bias_values))}, mean |bias| {num(mean_bias)})."
        elif mean(bias_values) < -0.25:
            bias_line = f"Severity tends to run low in this category (mean bias {signed_num(mean(bias_values))}, mean |bias| {num(mean_bias)})."
        else:
            bias_line = f"Severity is roughly centered overall (mean bias {signed_num(mean(bias_values))}, mean |bias| {num(mean_bias)})."
    else:
        bias_line = "Severity bias is unavailable."

    if wrong_values and max(wrong_values) > 0:
        trust_line = f"Wrong-consensus risk is present: the highest condition-level wrong-consensus rate is {pct(max(wrong_values))}."
    else:
        trust_line = "Wrong-consensus risk does not appear in the current runs."

    title = CATEGORY_TITLES.get(category, category.replace("_", " ").title())
    return f"{title}: {outcome_line} {bias_line} {trust_line}"


def rate_style(value: Optional[float], *, higher_is_better: bool = True) -> str:
    if value is None:
        return ""
    bounded = max(0.0, min(1.0, value))
    if not higher_is_better:
        bounded = 1.0 - bounded
    low = (252, 226, 226)
    high = (220, 252, 231)
    red = int(low[0] + (high[0] - low[0]) * bounded)
    green = int(low[1] + (high[1] - low[1]) * bounded)
    blue = int(low[2] + (high[2] - low[2]) * bounded)
    return f"background: rgb({red}, {green}, {blue});"


def bias_style(value: Optional[float]) -> str:
    if value is None:
        return ""
    magnitude = min(abs(value) / 3.0, 1.0)
    if value > 0:
        return f"background: rgba(239, 68, 68, {0.08 + magnitude * 0.22:.2f});"
    if value < 0:
        return f"background: rgba(59, 130, 246, {0.08 + magnitude * 0.22:.2f});"
    return "background: rgba(15, 23, 42, 0.04);"


def link_href(target: Path, output_path: Path) -> str:
    return html.escape(os.path.relpath(target, output_path.parent))


def render_condition_table(
    category_runs: Sequence[RunEntry],
    condition_aggregates: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> str:
    rows: List[str] = []
    for condition_id in sorted(condition_aggregates, key=condition_sort_key):
        aggregate = condition_aggregates[condition_id]
        headline = aggregate.get("headline_metrics", {})
        derived = aggregate.get("derived_metrics", {})
        extras = aggregate.get("extras", {})
        meta = CONDITION_META.get(condition_id, {})
        sample_entries = sorted(
            [entry for entry in category_runs if entry.condition_id == condition_id],
            key=lambda entry: entry.report_path.name,
        )
        sample_entry = sample_entries[0] if sample_entries else None
        sample_report = (
            f"<a href='{link_href(sample_entry.report_path, output_path)}'>{html.escape(sample_entry.report_path.name)}</a>"
            if sample_entry
            else "n/a"
        )
        rows.append(
            "<tr>"
            f"<td><span class='cond-tag'>{html.escape(condition_id)}</span></td>"
            f"<td>{html.escape(meta.get('label', condition_id))}</td>"
            f"<td>{html.escape(meta.get('family', 'n/a'))}</td>"
            f"<td>{html.escape(meta.get('mode', 'n/a'))}</td>"
            f"<td style='{rate_style(as_float(headline.get('FinalCorrectExact')), higher_is_better=True)}'>{pct(as_float(headline.get('FinalCorrectExact')))}</td>"
            f"<td style='{rate_style(as_float(headline.get('FinalCorrectType')), higher_is_better=True)}'>{pct(as_float(headline.get('FinalCorrectType')))}</td>"
            f"<td style='{rate_style(as_float(headline.get('FinalAgreementExact')), higher_is_better=True)}'>{pct(as_float(headline.get('FinalAgreementExact')))}</td>"
            f"<td style='{rate_style(as_float(derived.get('WrongConsensusExactRate')), higher_is_better=False)}'>{pct(as_float(derived.get('WrongConsensusExactRate')))}</td>"
            f"<td style='{rate_style(as_float(derived.get('LateDriftAgreementExactRate')), higher_is_better=False)}'>{pct(as_float(derived.get('LateDriftAgreementExactRate')))}</td>"
            f"<td style='{bias_style(as_float(headline.get('SeverityBias')))}'>{signed_num(as_float(headline.get('SeverityBias')))}</td>"
            f"<td style='{rate_style(as_float(derived.get('OverSeverityRate')), higher_is_better=False)}'>{pct(as_float(derived.get('OverSeverityRate')))}</td>"
            f"<td>{pct(as_float(headline.get('TrustHygieneRate')))}</td>"
            f"<td>{num(as_float(extras.get('type_transitions_mean')))}</td>"
            f"<td>{num(as_float(extras.get('exact_transitions_mean')))}</td>"
            f"<td>{num(as_float(derived.get('ConsensusLatencyExactMean')))}</td>"
            f"<td>{sample_report}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'>"
        "<table>"
        "<thead><tr>"
        "<th>Condition</th><th>Setup</th><th>Family</th><th>Mode</th>"
        "<th>Exact Correct</th><th>Type Correct</th><th>Final Agreement</th>"
        "<th>Wrong Consensus</th><th>Late Drift</th><th>Severity Bias</th>"
        "<th>Over-Severity</th><th>Trust Hygiene</th><th>Type Transitions</th>"
        "<th>Exact Transitions</th><th>Latency</th><th>Sample Report</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def render_question_card(item: Dict[str, Any]) -> str:
    evidence_items = "".join(f"<li>{html.escape(line)}</li>" for line in list(item.get("evidence") or []))
    return (
        "<article class='question-card'>"
        f"<div class='card-head'><h3>{html.escape(str(item.get('title', '')))}</h3><span class='status-badge'>{html.escape(str(item.get('status', '')))}</span></div>"
        f"<p>{html.escape(str(item.get('summary', '')))}</p>"
        f"<ul>{evidence_items}</ul>"
        "</article>"
    )


def render_hypothesis_card(item: Dict[str, Any]) -> str:
    return (
        "<article class='hypothesis-card'>"
        f"<div class='card-head'><h3>{html.escape(str(item.get('title', '')))}</h3><span class='status-badge'>{html.escape(str(item.get('status', '')))}</span></div>"
        f"<p>{html.escape(str(item.get('summary', '')))}</p>"
        "</article>"
    )


def render_metric_charts(condition_aggregates: Dict[str, Dict[str, Any]]) -> str:
    ordered_ids = sorted(condition_aggregates, key=condition_sort_key)
    exact_items = [(condition_id, get_metric(condition_aggregates[condition_id], "headline_metrics", "FinalCorrectExact")) for condition_id in ordered_ids]
    wrong_items = [(condition_id, get_metric(condition_aggregates[condition_id], "derived_metrics", "WrongConsensusExactRate")) for condition_id in ordered_ids]
    bias_items = [(condition_id, get_metric(condition_aggregates[condition_id], "headline_metrics", "SeverityBias")) for condition_id in ordered_ids]

    return (
        "<div class='chart-grid'>"
        f"<div class='chart-card'>{svg_rate_chart(exact_items, 'Final Exact Correctness by Condition', color='#0f766e')}</div>"
        f"<div class='chart-card'>{svg_rate_chart(wrong_items, 'Wrong Consensus by Condition', color='#b45309')}</div>"
        f"<div class='chart-card'>{svg_bias_chart(bias_items, 'Severity Bias by Condition')}</div>"
        "</div>"
    )


def scenario_synopsis(scenario_id: str, condition_aggregates: Dict[str, Dict[str, Any]]) -> str:
    exact_values = [
        get_metric(aggregate, "headline_metrics", "FinalCorrectExact")
        for aggregate in condition_aggregates.values()
        if get_metric(aggregate, "headline_metrics", "FinalCorrectExact") is not None
    ]
    bias_values = [
        get_metric(aggregate, "headline_metrics", "SeverityBias")
        for aggregate in condition_aggregates.values()
        if get_metric(aggregate, "headline_metrics", "SeverityBias") is not None
    ]
    wrong_values = [
        get_metric(aggregate, "derived_metrics", "WrongConsensusExactRate")
        for aggregate in condition_aggregates.values()
        if get_metric(aggregate, "derived_metrics", "WrongConsensusExactRate") is not None
    ]
    title = scenario_title(scenario_id)
    exact_line = (
        f"Exact correctness spans {pct(min(exact_values))} to {pct(max(exact_values))}."
        if exact_values
        else "Exact correctness unavailable."
    )
    bias_line = (
        f"Mean severity bias {signed_num(mean(bias_values))}."
        if bias_values
        else "Severity bias unavailable."
    )
    wrong_line = (
        f"Wrong-consensus peaks at {pct(max(wrong_values))}."
        if wrong_values
        else "Wrong-consensus unavailable."
    )
    return f"{title}: {exact_line} {bias_line} {wrong_line}"


def render_scenario_block(
    scenario_id: str,
    scenario_runs: Sequence[RunEntry],
    output_path: Path,
) -> str:
    condition_aggregates = {
        condition_id: aggregate_entries([entry for entry in scenario_runs if entry.condition_id == condition_id], condition_id)
        for condition_id in sorted({entry.condition_id for entry in scenario_runs}, key=condition_sort_key)
    }
    return (
        "<article class='scenario-block'>"
        f"<div class='scenario-head'><h4>{html.escape(scenario_title(scenario_id))}</h4><p>{html.escape(scenario_synopsis(scenario_id, condition_aggregates))}</p></div>"
        f"{render_metric_charts(condition_aggregates)}"
        f"{render_condition_table(scenario_runs, condition_aggregates, output_path)}"
        "</article>"
    )


def render_run_table(category_runs: Sequence[RunEntry], output_path: Path) -> str:
    rows: List[str] = []
    for entry in sorted(category_runs, key=lambda item: (item.scenario_id, int(item.condition_id[1:]))):
        headline = entry.run_report.get("headline_metrics", {})
        derived = entry.run_report.get("derived_metrics", {})
        committee_final = dict(entry.run_report.get("committee_final") or {})
        final_exact = committee_final.get("committee_exact")
        if isinstance(final_exact, dict):
            final_label = str(final_exact.get("label", ""))
            final_severity = str(final_exact.get("severity", ""))
        else:
            final_label = str(committee_final.get("committee_type") or "")
            final_severity = ""
        ground_truth = dict(entry.history.get("ground_truth") or {})
        llm_eval = entry.llm_eval or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(entry.scenario_id)}</td>"
            f"<td>{html.escape(entry.condition_id)}</td>"
            f"<td>{html.escape(CONDITION_META.get(entry.condition_id, {}).get('label', entry.condition_id))}</td>"
            f"<td>{html.escape(final_label)}</td>"
            f"<td>{html.escape(final_severity)}</td>"
            f"<td>{html.escape(str(ground_truth.get('final_label', '')))}</td>"
            f"<td>{html.escape(str(ground_truth.get('final_severity', '')))}</td>"
            f"<td style='{rate_style(as_float(headline.get('FinalCorrectExact')), higher_is_better=True)}'>{pct(as_float(headline.get('FinalCorrectExact')))}</td>"
            f"<td style='{rate_style(as_float(headline.get('FinalCorrectType')), higher_is_better=True)}'>{pct(as_float(headline.get('FinalCorrectType')))}</td>"
            f"<td style='{rate_style(as_float(derived.get('WrongConsensusExact')), higher_is_better=False)}'>{pct(as_float(derived.get('WrongConsensusExact')))}</td>"
            f"<td style='{bias_style(as_float(headline.get('SeverityBias')))}'>{signed_num(as_float(headline.get('SeverityBias')))}</td>"
            f"<td>{entry.public_turns}</td>"
            f"<td>{entry.type_transitions}</td>"
            f"<td>{entry.exact_transitions}</td>"
            f"<td>{html.escape(str(llm_eval.get('q4_report_defensibility', '')) or 'n/a')}</td>"
            f"<td><a href='{link_href(entry.report_path, output_path)}'>report</a></td>"
            f"<td><a href='{link_href(entry.history_path, output_path)}'>history</a></td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'>"
        "<table>"
        "<thead><tr>"
        "<th>Scenario</th><th>Condition</th><th>Setup</th><th>Final Label</th><th>Final Severity</th>"
        "<th>GT Label</th><th>GT Severity</th><th>Exact Correct</th><th>Type Correct</th>"
        "<th>Wrong Consensus</th><th>Severity Bias</th><th>Public Turns</th>"
        "<th>Type Transitions</th><th>Exact Transitions</th><th>LLM Q4</th><th>Report</th><th>History</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def render_category_section(
    category: str,
    category_runs: Sequence[RunEntry],
    output_path: Path,
) -> str:
    condition_aggregates = {
        condition_id: aggregate_entries([entry for entry in category_runs if entry.condition_id == condition_id], condition_id)
        for condition_id in sorted({entry.condition_id for entry in category_runs})
    }

    c1 = condition_aggregates.get("C1")
    c2 = condition_aggregates.get("C2")
    c3 = condition_aggregates.get("C3")
    c4 = condition_aggregates.get("C4")
    c5 = condition_aggregates.get("C5")
    c6 = condition_aggregates.get("C6")
    c7 = condition_aggregates.get("C7")

    def combine_from_ids(label: str, ids: Sequence[str]) -> Optional[Dict[str, Any]]:
        items = [entry for entry in category_runs if entry.condition_id in ids]
        if not items:
            return None
        return aggregate_entries(items, label)

    gpt_committee = combine_from_ids("GPT Committee", ("C3", "C6"))
    claude_committee = combine_from_ids("Claude Committee", ("C4", "C7"))
    homogeneous_committees = combine_from_ids("Homogeneous Committees", ("C3", "C4", "C6", "C7"))

    questions = [
        summarize_comparison("GPT-5 Committee vs Single GPT-5", c1, gpt_committee, trust_focus=False),
        summarize_comparison("Claude Committee vs Single Claude", c2, claude_committee, trust_focus=False),
        summarize_comparison("Mixed-Model Negotiation Trust", homogeneous_committees, c5, trust_focus=True),
        summarize_comparison("LLM Prior Effect (C3 -> C6)", c3, c6, trust_focus=False),
        summarize_comparison("Human Prior Effect (C4 -> C7)", c4, c7, trust_focus=False),
        summarize_c7_expectation(c2, c4, c7),
        summarize_instruction_priors(c3, c6, c4, c7),
    ]
    hypotheses = evaluate_hypotheses(condition_aggregates)

    question_cards = "".join(render_question_card(item) for item in questions)
    hypothesis_cards = "".join(render_hypothesis_card(item) for item in hypotheses)
    scenario_count = len({entry.scenario_id for entry in category_runs})
    llm_q4_coverage = sum(1 for entry in category_runs if entry.llm_eval and entry.llm_eval.get("q4_report_defensibility"))
    title = CATEGORY_TITLES.get(category, category.replace("_", " ").title())
    scenario_blocks = "".join(
        render_scenario_block(scenario_id, [entry for entry in category_runs if entry.scenario_id == scenario_id], output_path)
        for scenario_id in sorted({entry.scenario_id for entry in category_runs})
    )

    return (
        f"<section class='category-section' id='{html.escape(category)}'>"
        f"<div class='section-head'><h2>{html.escape(title)}</h2><p>{html.escape(category_synopsis(category, condition_aggregates))}</p></div>"
        "<div class='meta-line'>"
        f"<span>{len(category_runs)} completed runs</span>"
        f"<span>{scenario_count} scenarios</span>"
        f"<span>{llm_q4_coverage} runs with LLM evaluator Q4 coverage</span>"
        "</div>"
        "<div class='card-grid'>"
        f"{question_cards}"
        "</div>"
        "<div class='card-grid card-grid-small'>"
        f"{hypothesis_cards}"
        "</div>"
        "<div class='panel'>"
        "<h3>Metric Graphs</h3>"
        "<p class='muted'>These charts use the same aggregated condition metrics as the tables below, so you can spot strong cases and failure patterns faster.</p>"
        f"{render_metric_charts(condition_aggregates)}"
        "</div>"
        + (
            "<div class='panel'>"
            "<h3>Scenario Subcategories</h3>"
            "<p class='muted'>Categories with multiple scenarios are broken out below so the hard cookie cases do not get averaged together.</p>"
            f"<div class='scenario-stack'>{scenario_blocks}</div>"
            "</div>"
            if scenario_count > 1
            else ""
        )
        +
        "<div class='panel'>"
        "<h3>Condition Comparison</h3>"
        "<p class='muted'>Condition rows are aggregated over all completed runs in this category. Higher is better for correctness, lower is better for wrong-consensus and late-drift.</p>"
        f"{render_condition_table(category_runs, condition_aggregates, output_path)}"
        "</div>"
        "<div class='panel'>"
        "<h3>Run-Level Details</h3>"
        f"{render_run_table(category_runs, output_path)}"
        "</div>"
        "</section>"
    )


def render_dashboard(runs: Sequence[RunEntry], output_root: Path, output_path: Path, llm_eval_source: Optional[Path]) -> str:
    categories = sorted({entry.category for entry in runs})
    sections = []
    for category in categories:
        category_runs = [entry for entry in runs if entry.category == category]
        sections.append(render_category_section(category, category_runs, output_path))

    nav_links = "".join(
        f"<button type='button' class='nav-link' data-target='{html.escape(category)}'>{html.escape(CATEGORY_TITLES.get(category, category.replace('_', ' ').title()))}</button>"
        for category in categories
    )
    generated_note = f"LLM evaluator source: {llm_eval_source.name}" if llm_eval_source else "No LLM evaluator file detected."

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <base href="./" />
  <title>Cyber Category Research Overview</title>
  <style>
    :root {{
      --bg-a: #f3efe7;
      --bg-b: #d8e3f0;
      --panel: rgba(255, 255, 255, 0.88);
      --ink: #18212b;
      --muted: #53616f;
      --line: #cdd6df;
      --accent: #0f766e;
      --accent-2: #b45309;
      --shadow: 0 16px 40px rgba(24, 33, 43, 0.10);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.10), transparent 24%),
        linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 100%);
      scroll-behavior: smooth;
    }}
    a {{ color: #0f4c81; }}
    .wrap {{
      max-width: 1460px;
      margin: 0 auto;
      padding: 28px 24px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(244, 248, 251, 0.78));
      border: 1px solid rgba(205, 214, 223, 0.9);
      border-radius: 28px;
      padding: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 10px 0;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 0;
      max-width: 1000px;
      color: var(--muted);
      line-height: 1.55;
    }}
    .meta-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .pill {{
      border: 1px solid rgba(15, 118, 110, 0.18);
      background: rgba(255, 255, 255, 0.72);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    .condition-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .condition-tile {{
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(205, 214, 223, 0.92);
      border-radius: 16px;
      padding: 12px 14px;
    }}
    .condition-tile strong {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 6px;
    }}
    .condition-tile span {{
      color: var(--muted);
      font-size: 14px;
    }}
    .sticky-nav {{
      position: sticky;
      top: 0;
      z-index: 5;
      margin: 18px 0;
      background: rgba(248, 250, 252, 0.86);
      border: 1px solid rgba(205, 214, 223, 0.92);
      border-radius: 18px;
      padding: 10px;
      box-shadow: 0 10px 24px rgba(24, 33, 43, 0.08);
      backdrop-filter: blur(12px);
    }}
    .nav-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .nav-link {{
      appearance: none;
      text-decoration: none;
      color: var(--ink);
      background: rgba(255,255,255,0.86);
      border: 1px solid rgba(205, 214, 223, 0.94);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      font-family: inherit;
    }}
    .nav-link:hover {{
      border-color: rgba(15, 118, 110, 0.42);
      background: rgba(255,255,255,0.96);
    }}
    .category-section {{
      margin-bottom: 26px;
      scroll-margin-top: 82px;
    }}
    .section-head {{
      margin-bottom: 10px;
    }}
    .section-head h2 {{
      margin: 0 0 6px 0;
      font-size: 30px;
      letter-spacing: -0.03em;
    }}
    .section-head p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    .meta-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }}
    .meta-line span {{
      background: rgba(255,255,255,0.68);
      border: 1px solid rgba(205, 214, 223, 0.88);
      border-radius: 999px;
      padding: 6px 10px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    .card-grid-small {{
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    .question-card, .hypothesis-card, .panel {{
      background: var(--panel);
      border: 1px solid rgba(205, 214, 223, 0.94);
      border-radius: var(--radius);
      padding: 16px 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .card-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .card-head h3 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
    }}
    .status-badge {{
      flex: 0 0 auto;
      border: 1px solid rgba(15, 118, 110, 0.22);
      background: rgba(15, 118, 110, 0.08);
      color: #0b5f58;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .question-card p, .hypothesis-card p {{
      margin: 0;
      line-height: 1.55;
      color: var(--muted);
    }}
    .question-card ul {{
      margin: 10px 0 0 18px;
      padding: 0;
      color: var(--ink);
    }}
    .question-card li {{
      margin: 0 0 6px 0;
      line-height: 1.45;
    }}
    .panel h3 {{
      margin: 0 0 8px 0;
      font-size: 18px;
    }}
    .muted {{
      color: var(--muted);
      margin: 0 0 10px 0;
      line-height: 1.5;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .chart-card {{
      border: 1px solid rgba(205, 214, 223, 0.92);
      border-radius: 16px;
      padding: 10px;
      background: rgba(255, 255, 255, 0.84);
    }}
    .scenario-stack {{
      display: grid;
      gap: 14px;
    }}
    .scenario-block {{
      border: 1px solid rgba(205, 214, 223, 0.92);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.76);
    }}
    .scenario-head {{
      margin-bottom: 10px;
    }}
    .scenario-head h4 {{
      margin: 0 0 6px 0;
      font-size: 18px;
    }}
    .scenario-head p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid rgba(205, 214, 223, 0.92);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.82);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 1120px;
    }}
    th, td {{
      border-bottom: 1px solid rgba(205, 214, 223, 0.74);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #eef4f8;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #384657;
    }}
    tbody tr:hover {{
      background: rgba(15, 118, 110, 0.04);
    }}
    .cond-tag {{
      display: inline-block;
      min-width: 42px;
      text-align: center;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.10);
      color: #0b5f58;
      border: 1px solid rgba(15, 118, 110, 0.18);
      padding: 4px 8px;
      font-weight: 700;
    }}
    @media (max-width: 760px) {{
      .wrap {{
        padding: 18px 14px 34px;
      }}
      .hero {{
        padding: 18px;
      }}
      .hero h1 {{
        font-size: 28px;
      }}
      .section-head h2 {{
        font-size: 25px;
      }}
      .card-grid, .card-grid-small {{
        grid-template-columns: 1fr;
      }}
    }}
</style>
<script>
  document.addEventListener("DOMContentLoaded", function () {{
    document.querySelectorAll(".nav-link[data-target]").forEach(function (button) {{
      button.addEventListener("click", function () {{
        var targetId = button.getAttribute("data-target");
        if (!targetId) return;
        var target = document.getElementById(targetId);
        if (!target) return;
        target.scrollIntoView({{ behavior: "smooth", block: "start" }});
        if (window.history && window.history.replaceState) {{
          window.history.replaceState(null, "", "#" + targetId);
        }} else {{
          window.location.hash = targetId;
        }}
      }});
    }});
  }});
</script>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Cyber Triage Research Overview</h1>
      <p>
        This dashboard scans completed cyber triage runs in <code>{html.escape(str(output_root))}</code>,
        groups them by category, and answers the current research questions directly from the stored metrics.
        Comparisons use exact correctness, type correctness, wrong-consensus rate, trust-hygiene rate,
        severity bias, consensus latency, and trajectory volatility.
      </p>
      <div class="meta-strip">
        <span class="pill">{len(runs)} completed runs</span>
        <span class="pill">{len(categories)} categories</span>
        <span class="pill">{len({entry.scenario_id for entry in runs})} scenarios</span>
        <span class="pill">{html.escape(generated_note)}</span>
      </div>
      <div class="condition-strip">
        <div class="condition-tile"><strong>C1</strong><span>Single GPT-5 baseline</span></div>
        <div class="condition-tile"><strong>C2</strong><span>Single Claude baseline</span></div>
        <div class="condition-tile"><strong>C3</strong><span>3-agent GPT-5 negotiation</span></div>
        <div class="condition-tile"><strong>C4</strong><span>3-agent Claude negotiation</span></div>
        <div class="condition-tile"><strong>C5</strong><span>Mixed-model negotiation</span></div>
        <div class="condition-tile"><strong>C6</strong><span>GPT-5 negotiation with LLM prior</span></div>
        <div class="condition-tile"><strong>C7</strong><span>Claude negotiation with human prior</span></div>
      </div>
    </section>

    <nav class="sticky-nav">
      <div class="nav-grid">
        {nav_links}
      </div>
    </nav>

    {''.join(sections)}
  </div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a category-level HTML overview for cyber triage runs.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="games_descriptions/cyber_game/output",
        help="Cyber output root containing category folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional HTML output path. Defaults to <output_root>/research_overview_by_category.html",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    output_path = Path(args.output).resolve() if args.output else (output_root / "research_overview_by_category.html")
    runs, llm_eval_source = scan_runs(output_root)
    if not runs:
        raise SystemExit("No completed cyber runs were found under the output root.")

    html_doc = render_dashboard(runs, output_root, output_path, llm_eval_source)
    output_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote category overview: {output_path}")


if __name__ == "__main__":
    main()
