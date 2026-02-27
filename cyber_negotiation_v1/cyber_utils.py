from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from statistics import pvariance
from typing import Any, Dict, List, Optional, Tuple


SEVERITY_ORDER = {
    "Compliance": 0,
    "Info": 1,
    "Low": 2,
    "Medium": 3,
    "High": 4,
}


def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val


def read_config(config_path: str) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    game_dir = cfg_path.parent
    agents: List[Dict[str, str]] = []
    with cfg_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                raise ValueError(f"Malformed config line: {line}")
            name, file_key, role, incentive, model = parts
            agents.append(
                {
                    "name": name,
                    "file_name": file_key,
                    "role": role,
                    "incentive": incentive,
                    "model": model,
                }
            )

    init_file = game_dir / "initial_deal.txt"
    default_scenario = "placeholder_webapp_001.json"
    if init_file.exists():
        for raw in init_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            if key.strip().lower() == "scenario":
                default_scenario = val.strip()

    starter = next((a["name"] for a in agents if a["role"] == "p1"), agents[0]["name"] if agents else "")
    return {
        "agents": agents,
        "starter": starter,
        "default_scenario": default_scenario,
        "round_assign": [],
        "seed": 42,
    }


def read_condition_config(condition_path: str) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "condition_id": "C1",
        "mode": "negotiation",
        "public_messages": 6,
        "json_max_retries": 3,
        "prior_round0": "You are negotiating with other LLM reviewers.",
        "apply_priors_round0_only": True,
        "reminder_text": "Stay evidence-grounded and cite exact line IDs.",
        "reinject_role_instruction_roundn": False,
        "final_turn_announcement_window": 1,
        "per_call_timeout_seconds": 30,
        "per_run_wallclock_limit_seconds": 180,
        "token_budget_limit": None,
        "public_message_min_words": 80,
        "public_message_max_words": 150,
        "public_message_hard_cap_words": 220,
        "invalid_json_attempts_per_turn": 0,
    }
    out = dict(defaults)
    with open(condition_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip().lower()
            val = val.strip()
            if key in {
                "public_messages",
                "json_max_retries",
                "final_turn_announcement_window",
                "per_call_timeout_seconds",
                "per_run_wallclock_limit_seconds",
                "public_message_min_words",
                "public_message_max_words",
                "public_message_hard_cap_words",
                "invalid_json_attempts_per_turn",
            }:
                out[key] = int(val)
            elif key in {"apply_priors_round0_only", "reinject_role_instruction_roundn"}:
                out[key] = val.lower() in {"1", "true", "yes", "y", "on"}
            elif key == "token_budget_limit":
                out[key] = int(val) if val else None
            else:
                out[key] = val
    if out["mode"] == "negotiation" and int(out["public_messages"]) % 3 != 0:
        raise ValueError("public_messages must be divisible by 3")
    return out


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_label_set(path: str | Path) -> Dict[str, Any]:
    data = load_json(path)
    data.setdefault("aliases", {})
    return data


def format_evidence_packet(packet: Dict[str, Any]) -> str:
    visible = {k: v for k, v in packet.items() if k != "author_notes"}
    return json.dumps(visible, indent=2, ensure_ascii=False)


def format_history(agent_name: str, history: Dict[str, Any], window: int = 6) -> Tuple[str, str]:
    last_plan = ""
    personalized_history: List[str] = []
    for slot in history.get("rounds", [])[-window:]:
        if agent_name == slot["agent"]:
            slot_str = f". You ({slot['agent']}): {slot['public_answer']}"
        else:
            slot_str = f". {slot['agent']}: {slot['public_answer']}"
        personalized_history.append(slot_str)
    if agent_name in history.get("plan", {}):
        last_plan = history["plan"][agent_name][-1]
    return " \n ".join(personalized_history), last_plan


def extract_tag_block(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_public_answer(answer: str) -> str:
    return extract_tag_block(answer, "ANSWER")


def extract_plan(answer: str) -> str:
    return extract_tag_block(answer, "PLAN")


def extract_assessment(answer: str) -> Optional[Dict[str, Any]]:
    raw = extract_tag_block(answer, "ASSESSMENT")
    if not raw:
        return None
    return json.loads(raw)


def _normalize_label(label: str, label_set: Dict[str, Any]) -> str:
    aliases = {str(k).strip().lower(): str(v) for k, v in label_set.get("aliases", {}).items()}
    return aliases.get(label.strip().lower(), label)


def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def parse_structured_response(
    resp_text: str,
    *,
    line_ids: List[str],
    label_set: Dict[str, Any],
    public_message_min_words: int,
    public_message_max_words: int,
    public_message_hard_cap_words: int,
) -> Tuple[Optional[Dict[str, Any]], str, List[str]]:
    warnings: List[str] = []
    try:
        parsed = json.loads(resp_text)
    except Exception as exc:
        return None, f"invalid JSON: {exc}", warnings
    if not isinstance(parsed, dict):
        return None, "top-level JSON must be an object", warnings
    for key in ("scratchpad", "answer", "plan"):
        if key not in parsed or not isinstance(parsed[key], str):
            return None, f"missing or invalid string field: {key}", warnings

    answer = parsed["answer"]
    public_answer = extract_public_answer(answer)
    if not public_answer:
        return None, "answer must include <ANSWER>...</ANSWER>", warnings
    wc = _word_count(public_answer)
    if wc > public_message_hard_cap_words:
        return None, "public answer exceeds hard cap", warnings
    if wc < public_message_min_words or wc > public_message_max_words:
        warnings.append("public_message_target_range")

    try:
        assessment = extract_assessment(answer)
    except Exception as exc:
        return None, f"invalid <ASSESSMENT> JSON: {exc}", warnings
    if assessment is None or not isinstance(assessment, dict):
        return None, "answer must include <ASSESSMENT>{...}</ASSESSMENT>", warnings

    ranked = assessment.get("ranked_findings")
    if not isinstance(ranked, list) or len(ranked) != 3:
        return None, "assessment.ranked_findings must contain exactly 3 items", warnings

    allowed_labels = set(label_set.get("labels", []))
    seen_ranks = set()
    for item in ranked:
        if not isinstance(item, dict):
            return None, "each ranked finding must be an object", warnings
        rank = item.get("rank")
        label = item.get("label")
        severity = item.get("severity")
        citations = item.get("citations", [])
        if rank not in {1, 2, 3}:
            return None, "ranks must be 1, 2, and 3", warnings
        seen_ranks.add(rank)
        if not isinstance(label, str):
            return None, "label must be a string", warnings
        normalized = _normalize_label(label, label_set)
        item["label"] = normalized
        if normalized not in allowed_labels:
            return None, f"invalid label: {normalized}", warnings
        if severity not in SEVERITY_ORDER:
            return None, f"invalid severity: {severity}", warnings
        if not isinstance(citations, list):
            return None, "citations must be a list", warnings
        if rank == 1 and not (1 <= len(citations) <= 2):
            return None, "rank 1 must cite 1-2 line IDs", warnings
        for citation in citations:
            if citation not in line_ids:
                return None, f"invalid line citation: {citation}", warnings
    if seen_ranks != {1, 2, 3}:
        return None, "ranks 1, 2, and 3 must all be present", warnings
    if not isinstance(assessment.get("decision_summary"), str) or not assessment.get("decision_summary", "").strip():
        return None, "assessment.decision_summary must be a non-empty string", warnings

    parsed["public_answer"] = public_answer
    parsed["assessment"] = assessment
    return parsed, "", warnings


def build_retry_prompt(base_prompt: str, attempt_idx: int, error: str) -> str:
    return (
        base_prompt
        + "\n\nYour previous response was invalid. Return ONLY one valid JSON object with keys scratchpad/answer/plan."
        + f"\nAttempt: {attempt_idx}\nValidation error: {error}\n"
    )


def generate_public_schedule(agent_names: List[str], total_public_messages: int, seed: int) -> List[str]:
    if total_public_messages % len(agent_names) != 0:
        raise ValueError("total_public_messages must be divisible by number of agents")
    quota = total_public_messages // len(agent_names)
    counts = {name: quota for name in agent_names}
    seq: List[str] = []
    rng = random.Random(seed)

    def feasible() -> bool:
        remaining = sum(counts.values())
        if remaining <= 1:
            return True
        max_count = max(counts.values())
        return max_count <= (remaining - max_count) + 1

    def backtrack(last: Optional[str]) -> bool:
        if len(seq) == total_public_messages:
            return True
        candidates = [name for name in agent_names if counts[name] > 0 and name != last]
        rng.shuffle(candidates)
        for name in candidates:
            counts[name] -= 1
            if feasible():
                seq.append(name)
                if backtrack(name):
                    return True
                seq.pop()
            counts[name] += 1
        return False

    if not backtrack(None):
        raise RuntimeError("could not construct valid public schedule")
    return seq


def make_committee_snapshot(public_turn_index: int, latest_by_agent: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    labels: Dict[str, Optional[str]] = {}
    severities: Dict[str, Optional[str]] = {}
    for agent, payload in latest_by_agent.items():
        rank1 = (payload.get("assessment", {}).get("ranked_findings") or [{}])[0]
        labels[agent] = rank1.get("label")
        severities[agent] = rank1.get("severity")

    label_values = [v for v in labels.values() if v is not None]
    sev_values = [v for v in severities.values() if v is not None]
    exact_values = [
        (labels[agent], severities[agent])
        for agent in labels
        if labels[agent] is not None and severities[agent] is not None
    ]

    def majority(values: List[Any]) -> Tuple[Any, str]:
        counts: Dict[Any, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        if not counts:
            return None, "no_majority"
        winner, count = sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))[0]
        return (winner, "majority") if count >= 2 else (None, "no_majority")

    committee_type_label, committee_type_status = majority(label_values)
    committee_majority_severity, _ = majority(sev_values)
    committee_exact_pair, committee_exact_status = majority(exact_values)

    return {
        "public_turn_index": public_turn_index,
        "by_agent_top1_label": labels,
        "by_agent_top1_severity": severities,
        "committee_type_label": committee_type_label,
        "committee_exact_label": committee_exact_pair[0] if committee_exact_pair else None,
        "committee_exact_severity": committee_exact_pair[1] if committee_exact_pair else None,
        "committee_type_status": committee_type_status,
        "committee_exact_status": committee_exact_status,
        "full_agreement_type": len(label_values) == 3 and len(set(label_values)) == 1,
        "full_agreement_exact": len(exact_values) == 3 and len(set(exact_values)) == 1,
        "committee_majority_severity": committee_majority_severity,
    }


def compute_metrics(history: Dict[str, Any], ground_truth: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    committee_snaps = history.get("committee_snapshots", [])
    final = committee_snaps[-1] if committee_snaps else None
    final_type = final.get("committee_type_label") if final else None
    final_exact_label = final.get("committee_exact_label") if final else None
    final_exact_severity = final.get("committee_exact_severity") if final else None
    final_majority_severity = final.get("committee_majority_severity") if final else None
    final_agreement_type = bool(final and final.get("full_agreement_type"))
    final_agreement_exact = bool(final and final.get("full_agreement_exact"))

    final_correct_type = False
    final_correct_severity = False
    final_correct_exact = False
    over_severity = False
    under_severity = False
    wrong_consensus_type = False
    wrong_consensus_exact = False
    late_drift_type = False
    late_drift_exact = False

    if ground_truth and final:
        gt_label = ground_truth.get("final_label")
        gt_severity = ground_truth.get("final_severity")
        final_correct_type = final_type == gt_label
        final_correct_severity = final_majority_severity == gt_severity if final_majority_severity is not None else False
        final_correct_exact = final_exact_label == gt_label and final_exact_severity == gt_severity
        if final_majority_severity in SEVERITY_ORDER and gt_severity in SEVERITY_ORDER:
            cmp = SEVERITY_ORDER[final_majority_severity] - SEVERITY_ORDER[gt_severity]
            over_severity = cmp > 0
            under_severity = cmp < 0
        wrong_consensus_type = final_agreement_type and not final_correct_type
        wrong_consensus_exact = final_agreement_exact and not final_correct_exact
        for snap in committee_snaps[:-1]:
            if snap.get("full_agreement_type") and snap.get("committee_type_label") == gt_label:
                late_drift_type = not final_correct_type or not final_agreement_type
            if (
                snap.get("full_agreement_exact")
                and snap.get("committee_exact_label") == gt_label
                and snap.get("committee_exact_severity") == gt_severity
            ):
                late_drift_exact = not final_correct_exact or not final_agreement_exact

    trajectory = [snap.get("committee_type_label") for snap in committee_snaps]
    flip_count = sum(1 for prev, cur in zip(trajectory, trajectory[1:]) if prev != cur)

    def consensus_latency(exact: bool) -> Optional[int]:
        for idx, snap in enumerate(committee_snaps):
            if exact:
                if not snap.get("full_agreement_exact"):
                    continue
                anchor = (snap.get("committee_exact_label"), snap.get("committee_exact_severity"))
                stable = all(
                    s.get("full_agreement_exact")
                    and (s.get("committee_exact_label"), s.get("committee_exact_severity")) == anchor
                    for s in committee_snaps[idx:]
                )
                if stable:
                    return snap.get("public_turn_index")
            else:
                if not snap.get("full_agreement_type"):
                    continue
                anchor = snap.get("committee_type_label")
                stable = all(
                    s.get("full_agreement_type") and s.get("committee_type_label") == anchor
                    for s in committee_snaps[idx:]
                )
                if stable:
                    return snap.get("public_turn_index")
        return None

    severity_series = [
        SEVERITY_ORDER[snap["committee_majority_severity"]]
        for snap in committee_snaps
        if snap.get("committee_majority_severity") in SEVERITY_ORDER
    ]
    severity_variance = float(pvariance(severity_series)) if len(severity_series) > 1 else (0.0 if severity_series else None)

    validation = history.get("validation_stats", {})
    total_turns = int(validation.get("total_turns", 0) or 0)
    total_attempts = int(validation.get("total_attempts", 0) or 0)
    citation_total = int(validation.get("citation_total_rank1", 0) or 0)

    metrics = {
        "FinalCorrectType": final_correct_type,
        "FinalCorrectSeverity": final_correct_severity,
        "FinalCorrectExact": final_correct_exact,
        "OverSeverityRate": 1.0 if over_severity else 0.0,
        "UnderSeverityRate": 1.0 if under_severity else 0.0,
        "FinalAgreementType": final_agreement_type,
        "FinalAgreementExact": final_agreement_exact,
        "WrongConsensusType": wrong_consensus_type,
        "WrongConsensusExact": wrong_consensus_exact,
        "NoConsensus": (not final_agreement_exact) if final is not None else None,
        "LateDriftType": late_drift_type,
        "LateDriftExact": late_drift_exact,
        "FlipCountType": flip_count,
        "ConsensusLatencyType": consensus_latency(False),
        "ConsensusLatencyExact": consensus_latency(True),
        "SeverityVarianceAcrossRounds": severity_variance,
        "ExactSeverityDisagreementRateAtFinal": (0.0 if final_agreement_exact else (1.0 if final else None)),
        "JsonRetryCount": validation.get("json_retry_count", 0),
        "JsonFailureRate": (validation.get("json_failures", 0) / total_turns) if total_turns else 0.0,
        "SchemaValidationFailureRate": (
            validation.get("schema_validation_failures", 0) / total_attempts
        ) if total_attempts else 0.0,
        "CitationValidityRate": (
            validation.get("citation_valid_rank1", 0) / citation_total
        ) if citation_total else None,
        "CitationCountViolations": validation.get("citation_count_violations", 0),
        "MessageLengthViolations": validation.get("message_length_violations", 0),
    }
    return {
        "metrics": metrics,
        "committee_final": final,
        "committee_snapshots": committee_snaps,
    }
