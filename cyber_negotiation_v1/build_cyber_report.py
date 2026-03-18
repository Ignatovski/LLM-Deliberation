from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_output_root(run_dir: Path) -> Optional[Path]:
    for candidate in (run_dir, *run_dir.parents):
        if (candidate / "llm_evaluator").exists():
            return candidate
    return None


def load_llm_eval_map(output_root: Optional[Path]) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], Optional[Path]]:
    if output_root is None:
        return {}, None
    eval_dir = output_root / "llm_evaluator"
    if not eval_dir.exists():
        return {}, None

    candidates = list(eval_dir.glob("llm_trust_hygiene_per_run_*.json")) + list(eval_dir.glob("llm_eval_per_run_*.json"))
    if not candidates:
        return {}, None

    scored_candidates: List[Tuple[int, int, int, float, Path, Dict[str, Any]]] = []
    for path in candidates:
        payload = load_json(path)
        items = list(payload.get("runs") or [])
        completed_count = sum(1 for item in items if str(item.get("status") or "") == "completed")
        trust_metric = str(payload.get("metric_name") or "") == "TrustHygieneRate_LLM_GPT5" or path.name.startswith(
            "llm_trust_hygiene_per_run_"
        )
        scored_candidates.append(
            (
                1 if trust_metric else 0,
                completed_count,
                len(items),
                path.stat().st_mtime,
                path,
                payload,
            )
        )

    _, _, _, _, source_path, payload = max(scored_candidates, key=lambda item: item[:4])
    records: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for item in list(payload.get("runs") or []):
        key = (str(item.get("scenario_id", "")), str(item.get("condition_id", "")), str(item.get("run_id", "")))
        if all(key):
            records[key] = item
    return records, source_path


def bool_text(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return "n/a"


def llm_trust_hygiene_value(llm_eval: Optional[Dict[str, Any]]) -> Optional[float]:
    if not llm_eval or str(llm_eval.get("status") or "") != "completed":
        return None
    evaluation = dict(llm_eval.get("llm_evaluation") or {})
    value = as_float(evaluation.get("trust_hygiene_rate"))
    if value is not None:
        return value
    violated = evaluation.get("violated_run")
    if isinstance(violated, bool):
        return 1.0 if violated else 0.0
    return as_float(llm_eval.get("trust_hygiene_rate"))


def render_llm_audit_card(
    llm_eval: Optional[Dict[str, Any]],
    *,
    llm_eval_source: Optional[Path],
    stored_trust_hygiene_rate: Any,
) -> str:
    source_note = html.escape(llm_eval_source.name) if llm_eval_source else "none"
    if not llm_eval:
        return (
            "<div class='card wide'>"
            "<h3>GPT-5 Trust Audit</h3>"
            f"<p class='muted'>Audit source: {source_note}</p>"
            "<p class='muted'>No GPT-5 trust-audit record was found for this run.</p>"
            "</div>"
        )

    status = str(llm_eval.get("status") or "unknown")
    if status != "completed":
        error_text = str(llm_eval.get("error") or "")
        return (
            "<div class='card wide'>"
            "<h3>GPT-5 Trust Audit</h3>"
            f"<p class='muted'>Audit source: {source_note}</p>"
            "<div class='table-wrap'><table class='kv-table'>"
            f"<tr><th>Audit Status</th><td>{html.escape(status)}</td></tr>"
            f"<tr><th>Error</th><td>{html.escape(error_text or 'n/a')}</td></tr>"
            "</table></div>"
            "</div>"
        )

    evaluation = dict(llm_eval.get("llm_evaluation") or {})
    audit_rate = llm_trust_hygiene_value(llm_eval)
    summary = str(evaluation.get("summary") or "").strip()
    evidence_refs = list(evaluation.get("evidence_refs") or [])
    rows = {
        "Audit Status": status,
        "Stored TrustHygieneRate": stored_trust_hygiene_rate,
        "GPT-5 TrustHygieneRate": audit_rate,
        "Violated Run": bool_text(evaluation.get("violated_run")),
        "Citation Violation": bool_text(evaluation.get("citation_violation")),
        "Invalid Public Message": bool_text(evaluation.get("invalid_public_message")),
        "Leakage": bool_text(evaluation.get("leakage")),
        "Schema Failure After Retries": bool_text(evaluation.get("schema_failure_after_retries")),
        "Matches Stored Metric": bool_text(llm_eval.get("matches_stored_metric")),
        "Response ID": evaluation.get("response_id") or "n/a",
    }

    evidence_html = ""
    if evidence_refs:
        evidence_rows: List[str] = []
        for item in evidence_refs:
            if not isinstance(item, dict):
                continue
            location_bits = []
            if item.get("agent"):
                location_bits.append(f"agent={item.get('agent')}")
            if item.get("turn_index") is not None:
                location_bits.append(f"turn={item.get('turn_index')}")
            if item.get("public_turn_index") is not None:
                location_bits.append(f"public_turn={item.get('public_turn_index')}")
            if item.get("line_ids"):
                location_bits.append("lines=" + ", ".join(str(x) for x in list(item.get("line_ids") or [])))
            location = " | ".join(location_bits) if location_bits else "n/a"
            evidence_rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('issue_type') or 'n/a'))}</td>"
                f"<td>{html.escape(str(item.get('source_type') or 'n/a'))}</td>"
                f"<td>{html.escape(location)}</td>"
                f"<td>{html.escape(str(item.get('excerpt') or ''))}</td>"
                f"<td>{html.escape(str(item.get('reason') or ''))}</td>"
                "</tr>"
            )
        if evidence_rows:
            evidence_html = (
                "<div style='margin-top:10px;'>"
                "<h3 style='margin:0 0 8px; font-size:15px;'>Evidence References</h3>"
                "<div class='table-wrap'><table class='data-table'>"
                "<thead><tr><th>Issue</th><th>Source</th><th>Location</th><th>Excerpt</th><th>Reason</th></tr></thead>"
                f"<tbody>{''.join(evidence_rows)}</tbody>"
                "</table></div>"
                "</div>"
            )

    return (
        "<div class='card wide'>"
        "<h3>GPT-5 Trust Audit</h3>"
        f"<p class='muted'>Audit source: {source_note}</p>"
        f"<div class='table-wrap'><table class='kv-table'>{to_rows(rows)}</table></div>"
        + evidence_html
        + (
            "<div style='margin-top:10px;'>"
            "<h3 style='margin:0 0 8px; font-size:15px;'>GPT-5 Decision Summary</h3>"
            f"<pre>{html.escape(summary)}</pre>"
            "</div>"
            if summary
            else ""
        )
        + "</div>"
    )


def detect_run_files(run_dir: Path, stem: str | None) -> Dict[str, Path]:
    if stem:
        history_path = run_dir / f"{stem}.json"
        if not history_path.exists():
            raise FileNotFoundError(f"History file not found: {history_path}")
    else:
        history_candidates = sorted(run_dir.glob("history*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not history_candidates:
            raise FileNotFoundError(f"No history*.json files found in {run_dir}")
        history_path = history_candidates[0]
        stem = history_path.stem

    metrics_path = run_dir / f"metrics_{stem}.json"
    condition_path = run_dir / f"condition_summary_{stem}.json"
    return {
        "history": history_path,
        "metrics": metrics_path,
        "condition_summary": condition_path,
        "stem": Path(stem),
    }


def to_rows(d: Dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(k))}</th>"
            f"<td>{html.escape(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def svg_line_chart_binary(series: Dict[str, List[int]], width: int = 960, height: int = 240) -> str:
    pad = 36
    w = width - 2 * pad
    h = height - 2 * pad
    colors = ["#0b84f3", "#f39c12", "#2ecc71", "#e74c3c"]
    max_len = max((len(v) for v in series.values()), default=0)
    if max_len <= 1:
        return "<p>Not enough points for trajectory chart.</p>"

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="Binary trajectory chart">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{pad}" y1="{pad+h}" x2="{pad+w}" y2="{pad+h}" stroke="#333" stroke-width="1"/>',
    ]
    # y labels for binary chart
    for y_val in (0, 1):
        y = pad + (1 - y_val) * h
        parts.append(f'<line x1="{pad}" y1="{y}" x2="{pad+w}" y2="{y}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{pad-12}" y="{y+4}" text-anchor="end" font-size="12">{y_val}</text>')

    for idx, (name, values) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        points = []
        for i, val in enumerate(values):
            x = pad + (i / (max_len - 1)) * w
            y = pad + (1 - max(0, min(1, val))) * h
            points.append((x, y))
        point_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{point_str}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{color}"/>')

    tick_step = max(1, max_len // 8)
    for i in range(0, max_len, tick_step):
        x = pad + (i / (max_len - 1)) * w
        parts.append(f'<line x1="{x:.2f}" y1="{pad+h}" x2="{x:.2f}" y2="{pad+h+4}" stroke="#333" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{pad+h+18}" text-anchor="middle" font-size="11">{i}</text>')
    if (max_len - 1) % tick_step != 0:
        x = pad + w
        parts.append(f'<line x1="{x:.2f}" y1="{pad+h}" x2="{x:.2f}" y2="{pad+h+4}" stroke="#333" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{pad+h+18}" text-anchor="middle" font-size="11">{max_len-1}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def svg_bar_chart(counts: Iterable[Tuple[str, int]], title: str, width: int = 960, height: int = 260) -> str:
    items = list(counts)
    if not items:
        return f"<p>No data for {html.escape(title)}.</p>"
    pad = 36
    w = width - 2 * pad
    h = height - 2 * pad
    max_v = max(v for _, v in items) or 1
    bar_gap = 10
    bar_w = max(12, (w - (len(items) - 1) * bar_gap) / len(items))

    out = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{html.escape(title)}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{pad}" y="20" font-size="14" font-weight="600">{html.escape(title)}</text>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{pad}" y1="{pad+h}" x2="{pad+w}" y2="{pad+h}" stroke="#333" stroke-width="1"/>',
    ]
    for i, (name, value) in enumerate(items):
        x = pad + i * (bar_w + bar_gap)
        bh = (value / max_v) * (h - 8)
        y = pad + h - bh
        out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" fill="#0b84f3" opacity="0.85"/>')
        out.append(f'<text x="{x + bar_w/2:.2f}" y="{y - 5:.2f}" text-anchor="middle" font-size="11">{value}</text>')
        short_name = name if len(name) <= 16 else (name[:15] + "…")
        out.append(
            f'<text x="{x + bar_w/2:.2f}" y="{pad+h+14:.2f}" text-anchor="middle" font-size="10">{html.escape(short_name)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def slot_summary(slot: Dict[str, Any]) -> str:
    phase = slot.get("phase", "")
    turn = slot.get("turn_index", "")
    agent = slot.get("agent", "")
    top1 = slot.get("top1_exact") or {}
    top1_label = top1.get("label", slot.get("top1_label", ""))
    top1_sev = top1.get("severity", slot.get("top1_severity", ""))
    accept = slot.get("accept")
    accept_str = "accept=true" if accept is True else ("accept=false" if accept is False else "accept=unknown")
    return f"{phase} | turn={turn} | {agent} | top1={top1_label}/{top1_sev} | {accept_str}"


def _to_pre_block(value: Any) -> str:
    if value is None:
        return "(empty)"
    if isinstance(value, (dict, list)):
        if not value:
            return "(empty)"
        return html.escape(json.dumps(value, indent=2, ensure_ascii=False))
    text = str(value).strip()
    return html.escape(text) if text else "(empty)"


def render_turn_card(slot: Dict[str, Any]) -> str:
    assessment = slot.get("assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}
    validation = slot.get("validation", {})
    if not isinstance(validation, dict):
        validation = {}

    meta = {
        "phase": slot.get("phase"),
        "turn_index": slot.get("turn_index"),
        "agent": slot.get("agent"),
        "top1_label": slot.get("top1_label"),
        "top1_severity": slot.get("top1_severity"),
        "accept": slot.get("accept"),
        "block_reason": slot.get("block_reason"),
        "attempts": slot.get("attempts"),
        "rank1_citations": slot.get("rank1_citations"),
        "user_assumption_verdict": slot.get("user_assumption_verdict"),
    }

    meta_rows = []
    for key, value in meta.items():
        if value is None or value == "":
            continue
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        meta_rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(rendered)}</td>"
            "</tr>"
        )
    meta_table = (
        "<div class='table-wrap'><table class='kv-table'>"
        + "".join(meta_rows)
        + "</table></div>"
        if meta_rows
        else "<p class='muted'>No metadata.</p>"
    )

    # Keep only non-prompt raw fields for optional debugging.
    raw_turn = dict(slot)
    raw_turn.pop("prompt", None)
    raw_turn.pop("full_prompt_sent", None)
    raw_turn.pop("full_answer", None)

    return (
        "<details class='turn-card'>"
        f"<summary>{html.escape(slot_summary(slot))}</summary>"
        "<div class='turn-meta'>"
        "<h4>Turn Overview</h4>"
        f"{meta_table}"
        "</div>"
        "<div class='turn-grid'>"
        "<section class='turn-pane'>"
        "<h4>Public Answer</h4>"
        f"<pre>{_to_pre_block(slot.get('public_answer'))}</pre>"
        "</section>"
        "<section class='turn-pane'>"
        "<h4>Assessment</h4>"
        f"<pre>{_to_pre_block(assessment)}</pre>"
        "</section>"
        "</div>"
        "<details class='turn-sub'><summary>Scratchpad (Private)</summary>"
        f"<pre>{_to_pre_block(slot.get('private_notes'))}</pre>"
        "</details>"
        "<details class='turn-sub'><summary>Plan (Private)</summary>"
        f"<pre>{_to_pre_block(slot.get('private_plan'))}</pre>"
        "</details>"
        "<details class='turn-sub'><summary>Validation</summary>"
        f"<pre>{_to_pre_block(validation)}</pre>"
        "</details>"
        "<details class='turn-sub'><summary>Raw Turn JSON (No Prompts)</summary>"
        f"<pre>{_to_pre_block(raw_turn)}</pre>"
        "</details>"
        "</details>"
    )


def render_failed_attempt_card(item: Dict[str, Any]) -> str:
    meta = {
        "agent": item.get("agent"),
        "phase": item.get("phase"),
        "attempt": item.get("attempt"),
        "error": item.get("error"),
    }
    meta_rows = []
    for key, value in meta.items():
        if value is None or value == "":
            continue
        meta_rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
        )
    meta_table = (
        "<div class='table-wrap'><table class='kv-table'>"
        + "".join(meta_rows)
        + "</table></div>"
        if meta_rows
        else "<p class='muted'>No metadata.</p>"
    )

    return (
        "<details class='turn-card'>"
        f"<summary>failed attempt | {html.escape(str(item.get('agent') or 'unknown'))} | attempt={html.escape(str(item.get('attempt') or 'n/a'))}</summary>"
        "<div class='turn-meta'>"
        "<h4>Attempt Overview</h4>"
        f"{meta_table}"
        "</div>"
        "<div class='turn-grid'>"
        "<section class='turn-pane'>"
        "<h4>Response Text</h4>"
        f"<pre>{_to_pre_block(item.get('response_text'))}</pre>"
        "</section>"
        "<section class='turn-pane'>"
        "<h4>Failure Context</h4>"
        f"<pre>{_to_pre_block({'error': item.get('error'), 'phase': item.get('phase'), 'agent': item.get('agent'), 'attempt': item.get('attempt')})}</pre>"
        "</section>"
        "</div>"
        "</details>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HTML report for a cyber negotiation run.")
    parser.add_argument("--run_dir", type=str, required=True, help="Run output directory containing history*.json")
    parser.add_argument("--run_stem", type=str, default="", help="History stem without .json (example: history18_55_16)")
    parser.add_argument("--output", type=str, default="", help="Optional output html path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    files = detect_run_files(run_dir, args.run_stem.strip() or None)
    stem = files["stem"].name

    history = load_json(files["history"])
    metrics_payload = load_json(files["metrics"]) if files["metrics"].exists() else {}
    condition_summary = load_json(files["condition_summary"]) if files["condition_summary"].exists() else {}
    output_root = find_output_root(run_dir)
    llm_eval_map, llm_eval_source = load_llm_eval_map(output_root)

    run_report = metrics_payload.get("run_report", {})
    headline = run_report.get("headline_metrics", history.get("metrics", {}))
    derived = run_report.get("derived_metrics", history.get("derived_metrics", {}))
    appendix = run_report.get("appendix_debug", history.get("appendix_debug", {}))
    validation_stats = history.get("validation_stats", {})
    trajectory = history.get("decision_trajectory", [])
    failed_attempts = list(history.get("failed_attempts") or [])

    round0 = history.get("round0", [])
    rounds = history.get("rounds", [])
    slots = list(round0) + list(rounds)
    scenario_id = str(history.get("scenario_id") or run_report.get("scenario_id") or "")
    condition_id = str(history.get("condition", {}).get("condition_id") or run_report.get("condition_id") or "")
    run_id = str(history.get("run_id", stem) or stem)
    llm_eval = llm_eval_map.get((scenario_id, condition_id, run_id))
    llm_audit_card = render_llm_audit_card(
        llm_eval,
        llm_eval_source=llm_eval_source,
        stored_trust_hygiene_rate=headline.get("TrustHygieneRate"),
    )

    agent_accept_counts: Dict[str, int] = defaultdict(int)
    agent_turn_counts: Dict[str, int] = defaultdict(int)
    label_counts = Counter()
    for slot in slots:
        agent = str(slot.get("agent", ""))
        if agent:
            agent_turn_counts[agent] += 1
            if slot.get("accept") is True:
                agent_accept_counts[agent] += 1
        label = slot.get("top1_label")
        if label:
            label_counts[str(label)] += 1

    agree_series = [1 if bool(s.get("agreement_exact_with_signoff")) else 0 for s in trajectory]
    all_accept_series = [1 if bool(s.get("all_accept")) else 0 for s in trajectory]
    false_agree_series = [1 if bool(s.get("false_agreement_without_signoff")) else 0 for s in trajectory]

    trajectory_chart = svg_line_chart_binary(
        {
            "agreement_exact_with_signoff": agree_series,
            "all_accept": all_accept_series,
            "false_agreement_without_signoff": false_agree_series,
        }
    )

    accept_rate_counts = []
    for agent, total in sorted(agent_turn_counts.items()):
        pct = int(round(100 * agent_accept_counts.get(agent, 0) / total)) if total else 0
        accept_rate_counts.append((agent, pct))
    accept_chart = svg_bar_chart(accept_rate_counts, "Agent Accept Rate (%)")
    label_chart = svg_bar_chart(sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])), "Top-1 Label Frequency")

    signoff_agents: List[str] = []
    for slot in slots:
        agent_name = str(slot.get("agent", "")).strip()
        if agent_name and agent_name not in signoff_agents:
            signoff_agents.append(agent_name)
    for snap in trajectory:
        accept_map = snap.get("by_agent_accept", {})
        if not isinstance(accept_map, dict):
            continue
        for agent_name in accept_map.keys():
            agent_name_str = str(agent_name).strip()
            if agent_name_str and agent_name_str not in signoff_agents:
                signoff_agents.append(agent_name_str)

    signoff_header_cells = "".join(
        f"<th>{html.escape(agent_name)} signoff</th>" for agent_name in signoff_agents
    )

    trajectory_rows = []
    for snap in trajectory:
        ce = snap.get("committee_exact")
        if isinstance(ce, dict):
            ce_str = f"{ce.get('label','')}/{ce.get('severity','')}"
        else:
            ce_str = str(ce)
        accept_map = snap.get("by_agent_accept", {})
        if not isinstance(accept_map, dict):
            accept_map = {}
        signoff_cells: List[str] = []
        for agent_name in signoff_agents:
            raw_accept = accept_map.get(agent_name)
            if raw_accept is True:
                accept_str = "True"
            elif raw_accept is False:
                accept_str = "False"
            else:
                accept_str = "NA"
            signoff_cells.append(f"<td>{accept_str}</td>")
        signoff_cells_html = "".join(signoff_cells)
        trajectory_rows.append(
            "<tr>"
            f"<td>{snap.get('turn_index','')}</td>"
            f"<td>{html.escape(str(snap.get('speaker','')))}</td>"
            f"{signoff_cells_html}"
            f"<td>{html.escape(str(snap.get('committee_type','')))}</td>"
            f"<td>{html.escape(ce_str)}</td>"
            f"<td>{int(bool(snap.get('all_accept')))}</td>"
            f"<td>{int(bool(snap.get('agreement_exact_with_signoff')))}</td>"
            f"<td>{int(bool(snap.get('false_agreement_without_signoff')))}</td>"
            "</tr>"
        )

    turn_cards = [render_turn_card(slot) for slot in slots]
    failed_attempt_cards = [render_failed_attempt_card(item) for item in failed_attempts]

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cyber Run Report - {html.escape(stem)}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #dbe3ee;
      --accent: #0b84f3;
      --accent2: #f39c12;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 10% 0%, #eef5ff, var(--bg) 60%);
    }}
    .container {{ max-width: 1400px; margin: 24px auto; padding: 0 16px 40px; }}
    .hero {{
      background: linear-gradient(135deg, #0b84f3, #3da9fc);
      color: white;
      padding: 18px 22px;
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(11,132,243,0.22);
      margin-bottom: 16px;
    }}
    .meta {{ color: #deecff; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }}
    .card h3 {{ margin: 0 0 10px; font-size: 16px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ color: #334155; font-weight: 600; }}
    .kv-table th {{ width: 34%; }}
    .data-table th {{ width: auto; white-space: nowrap; }}
    .wide {{ grid-column: 1 / -1; }}
    .section-title {{ margin: 16px 0 8px; font-size: 19px; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.45;
    }}
    .turn-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      margin: 8px 0;
      padding: 0 10px 10px;
    }}
    summary {{
      cursor: pointer;
      padding: 10px 0;
      font-weight: 600;
      color: #0f172a;
    }}
    .turn-meta {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 8px;
      margin-bottom: 8px;
    }}
    .turn-meta h4 {{
      margin: 0 0 6px;
      font-size: 13px;
      color: #334155;
    }}
    .turn-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
    }}
    .turn-grid h4 {{ margin: 0 0 6px; font-size: 13px; color: #334155; }}
    .turn-pane {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 8px;
    }}
    .turn-sub {{
      margin-top: 8px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 0 8px 8px;
    }}
    .turn-sub > summary {{
      padding: 8px 0;
      font-weight: 600;
      color: #334155;
    }}
    .chart-legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: #334155; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
    .muted {{ color: #6b7280; font-size: 12px; }}
    @media (max-width: 1000px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .turn-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1 style="margin:0 0 6px;">Cyber Negotiation Report</h1>
      <div class="meta">
        run_id: {html.escape(str(history.get("run_id", stem)))} |
        condition: {html.escape(str(history.get("condition", {}).get("condition_id", "unknown")))} |
        scenario: {html.escape(str(history.get("scenario_id", "unknown")))} |
        status: {html.escape(str(history.get("run_status", "unknown")))}
      </div>
    </div>

    <div class="grid">
      <div class="card"><h3>Headline Metrics</h3><div class="table-wrap"><table class="kv-table">{to_rows(headline)}</table></div></div>
      <div class="card"><h3>Derived Metrics</h3><div class="table-wrap"><table class="kv-table">{to_rows(derived)}</table></div></div>
      <div class="card"><h3>Appendix Debug</h3><div class="table-wrap"><table class="kv-table">{to_rows(appendix)}</table></div></div>
      <div class="card"><h3>Validation Stats</h3><div class="table-wrap"><table class="kv-table">{to_rows(validation_stats)}</table></div></div>
      <div class="card"><h3>Condition Aggregate (Headline)</h3><div class="table-wrap"><table class="kv-table">{to_rows(condition_summary.get("headline_metrics", {}))}</table></div></div>
      <div class="card"><h3>Condition Aggregate (Derived)</h3><div class="table-wrap"><table class="kv-table">{to_rows(condition_summary.get("derived_metrics", {}))}</table></div></div>
      {llm_audit_card}
    </div>

    <h2 class="section-title">Diagrams</h2>
    <div class="grid">
      <div class="card wide">
        <h3>Committee Agreement Trajectory</h3>
        {trajectory_chart}
        <div class="chart-legend">
          <span class="legend-item"><span class="swatch" style="background:#0b84f3;"></span>exact_agreement_with_signoff</span>
          <span class="legend-item"><span class="swatch" style="background:#f39c12;"></span>all_signoff</span>
          <span class="legend-item"><span class="swatch" style="background:#2ecc71;"></span>unanimous_exact_without_signoff</span>
        </div>
        <p class="muted">X-axis is trajectory step index (0 = round0 snapshot).</p>
      </div>
      <div class="card">
        <h3>Agent Accept Rate</h3>
        {accept_chart}
      </div>
      <div class="card">
        <h3>Top-1 Label Frequency</h3>
        {label_chart}
      </div>
    </div>

    <h2 class="section-title">Decision Trajectory</h2>
    <div class="card">
      <div class="table-wrap"><table class="data-table">
        <thead>
          <tr>
            <th>turn_index</th><th>speaker</th>{signoff_header_cells}<th>committee_type</th><th>committee_exact</th><th>all_signoff</th><th>exact_agreement_with_signoff</th><th>unanimous_exact_without_signoff</th>
          </tr>
        </thead>
        <tbody>
          {''.join(trajectory_rows)}
        </tbody>
      </table></div>
      <p class="muted">unanimous_exact_without_signoff = all agents currently predict the same exact outcome, but at least one agent set accept=false.</p>
    </div>

    <h2 class="section-title">Failed Attempts</h2>
    <div class="card">
      <p class="muted">These are rejected or failed generation attempts captured separately from the accepted public turns. Trust-hygiene issues can come from here even when the final visible turn stream looks clean.</p>
      {''.join(failed_attempt_cards) if failed_attempt_cards else "<p class='muted'>No failed attempts recorded for this run.</p>"}
    </div>

    <h2 class="section-title">Full Turn Data</h2>
    {''.join(turn_cards)}
  </div>
</body>
</html>
"""

    output = Path(args.output).resolve() if args.output else (run_dir / f"report_{stem}.html")
    output.write_text(html_doc, encoding="utf-8")
    print(f"Report written to: {output}")
    print(f"History source: {files['history']}")
    print(f"Metrics source: {files['metrics']}")
    print(f"Condition summary source: {files['condition_summary']}")


if __name__ == "__main__":
    main()
