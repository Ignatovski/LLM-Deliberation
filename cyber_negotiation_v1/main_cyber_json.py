from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cyber_agent import CyberAgent
from cyber_save_utils import create_outfiles, save_conversation, write_file
from cyber_utils import (
    NO_CONSENSUS,
    aggregate_condition_results,
    build_retry_prompt,
    compute_metrics,
    format_evidence_packet,
    format_history,
    generate_public_schedule,
    load_env_file,
    load_json,
    load_label_set,
    make_committee_snapshot,
    parse_structured_response,
    read_condition_config,
    read_config,
)


JSON_PROMPT = """You must respond with ONE JSON object, nothing else:
{
  "scratchpad": "<SCRATCHPAD>...</SCRATCHPAD>",
  "answer": "<ANSWER>Your public negotiation message only.</ANSWER>\n<ASSESSMENT>{\"ranked_findings\":[...],\"decision_summary\":\"...\",\"accept\":true,\"block_reason\":null}</ASSESSMENT>",
  "plan": "<PLAN>...</PLAN>"
}

Rules:
- scratchpad: private reasoning only; keep it concise.
- answer: must contain exactly one public <ANSWER>...</ANSWER> block and one hidden <ASSESSMENT>{...}</ASSESSMENT> JSON object.
- plan: private next-step notes; omit or use "<PLAN></PLAN>" if none.
- assessment.ranked_findings must contain ranks 1, 2, and 3.
- Every ranked finding must include a `citations` list (use `[]` if none for non-rank-1 findings).
- The rank-1 finding must include 1-2 line-ID citations (for example: ["L003","L010"]).
- Severity is required for ranks 1-3 and must be one of Compliance, Info, Low, Medium, High.
- Labels must come from the configured label set.
- assessment.accept must be true or false.
- If assessment.accept is false, assessment.block_reason must be a short non-empty string explaining why you cannot sign off.
- If assessment.accept is true, assessment.block_reason must be null or an empty string.
- Public message must stay report-defensible, obey the configured length limits, and must not expose private scratchpad or hidden planning.
- Output must be valid JSON; no extra text, comments, or Markdown.
If you cannot comply, reformat to valid JSON with the fields above; never return other text.
"""


class CyberInitialPromptJSON:
    def __init__(
        self,
        game_dir: str,
        agent_name: str,
        agent_file_name: str,
        behavior_pack: str,
        scenario: Dict[str, Any],
        condition: Dict[str, Any],
        label_set: Dict[str, Any],
    ):
        del agent_name
        global_text_path = os.path.join(game_dir, "global_instructions.txt")
        with open(global_text_path, "r", encoding="utf-8") as f:
            self.global_text = f.read().strip()

        instruction_path = os.path.join(
            game_dir, "individual_instructions", behavior_pack, f"{agent_file_name}.txt"
        )
        with open(instruction_path, "r", encoding="utf-8") as f:
            self.personal_text = f.read().strip()

        parts = [
            self.global_text,
            self.personal_text,
            "Allowed finding labels: " + ", ".join(label_set.get("labels", [])),
        ]
        if condition.get("prior_round0"):
            parts.append(str(condition["prior_round0"]).strip())
        parts.append("Evidence packet (all evidence visible from the start):\n" + format_evidence_packet(scenario))
        self.initial_prompt = "\n\n".join(part for part in parts if part)

    def return_initial_prompt(self):
        return self.initial_prompt


class CyberSlotPromptJSON:
    def __init__(self, agent_name: str, game_dir: str, agent_file_name: str, condition: Dict[str, Any]):
        self.agent_name = agent_name
        self.game_dir = game_dir
        self.agent_file_name = agent_file_name
        self.condition = condition
        instruction_path = os.path.join(
            game_dir, "individual_instructions", "cooperative", f"{agent_file_name}.txt"
        )
        self.personal_text = Path(instruction_path).read_text(encoding="utf-8").strip()

    def build_round0_prompt(self, history: Dict[str, Any]):
        del history
        return (
            "Round 0 independent assessment. All evidence is already visible. "
            "Produce your own initial cyber finding assessment before public negotiation begins.\n\n"
            + JSON_PROMPT
        )

    def build_slot_prompt(self, history: Dict[str, Any], round_idx: int, *_):
        history.setdefault("rounds", [])
        history.setdefault("plan", {})
        history_text, last_plan = format_history(self.agent_name, history, window=6)
        prompt = "Review the latest public negotiation history:\n"
        prompt += f"<HISTORY>{history_text}</HISTORY>\n" if history_text else "<HISTORY></HISTORY>\n"
        if self.condition.get("reminder_text"):
            prompt += f"Reminder: {str(self.condition['reminder_text']).strip()}\n"
        if last_plan:
            prompt += f"Your previous notes were <PREV_PLAN>{last_plan}</PREV_PLAN>.\n"
        if history.get("_current_final_turn"):
            prompt += "This is the final public turn. Commit to your most defensible assessment.\n"
        prompt += f"Current public turn index: {round_idx + 1}\n"
        prompt += "Return the required JSON object only.\n\n" + JSON_PROMPT
        return prompt


def ensure_text(response) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "content") and isinstance(response.content, str):
        return response.content
    return str(response)


def save_state(
    history: Dict[str, Any],
    agent_name: str,
    prompt: str,
    parsed_obj: Dict[str, Any],
    *,
    phase: str,
    extra_fields: Optional[Dict[str, Any]] = None,
):
    scratch = parsed_obj.get("scratchpad", "")
    answer = parsed_obj.get("answer", "")
    plan = parsed_obj.get("plan", "")
    full_answer = "\n".join(part for part in [scratch, answer, plan] if part)
    return save_conversation(history, agent_name, full_answer, prompt, phase=phase, extra_fields=extra_fields)


def derive_role_id(file_name: str) -> str:
    lowered = file_name.lower()
    if lowered.endswith("_r"):
        return "R"
    if lowered.endswith("_c"):
        return "C"
    if lowered.endswith("_k"):
        return "K"
    return file_name[-1:].upper()


def main():
    parser = argparse.ArgumentParser(description="Cyber negotiation game (polynomial-style JSON runner)")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--agents_num", type=int, default=0)
    parser.add_argument("--rounds_num", type=int, default=6, help="Legacy CLI field; condition files are authoritative")
    parser.add_argument("--output_dir", type=str, default="./games_descriptions/cyber_game/output/")
    parser.add_argument("--game_dir", type=str, default="./games_descriptions/cyber_game")
    parser.add_argument("--config_file", type=str, default="config.txt", help="Polynomial-style agent config file")
    parser.add_argument("--condition_file", type=str, default="conditions/C1.txt", help="Polynomial-style key=value condition file")
    parser.add_argument("--scenario_file", type=str, default="", help="Scenario JSON file name under scenarios/")
    parser.add_argument("--ground_truth_file", type=str, default="", help="Ground-truth JSON file name under ground_truth/")
    parser.add_argument("--label_set_file", type=str, default="label_sets/default_findings_v1.json")
    parser.add_argument("--exp_name", type=str, default="cyber_run")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--output_file", type=str, default="history.json")
    parser.add_argument("--azure", action="store_true", help="Use Azure OpenAI for OpenAI model names")
    parser.add_argument("--model", type=str, default="", help="Optional model override for all agents")
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument("--max_attempts_per_turn", type=int, default=0, help="0 means use json_max_retries from the condition file")
    parser.add_argument("--retry_sleep", type=float, default=0.0)
    args = parser.parse_args()

    load_env_file(args.env_file)

    condition_path = os.path.join(args.game_dir, args.condition_file)
    condition = read_condition_config(condition_path)
    config_name = args.config_file
    if config_name == "config.txt":
        config_name = str(condition["config_file"])
    config_path = os.path.join(args.game_dir, config_name)
    config = read_config(config_path)
    max_attempts_per_turn = int(args.max_attempts_per_turn or condition["json_max_retries"])

    scenario_name = args.scenario_file or config.get("default_scenario", "placeholder_webapp_001.json")
    scenario_path = scenario_name if os.path.isabs(scenario_name) else os.path.join(args.game_dir, "scenarios", scenario_name)
    scenario = load_json(scenario_path)

    ground_truth_name = args.ground_truth_file or f"{Path(scenario_name).stem}.json"
    ground_truth_path = (
        ground_truth_name
        if os.path.isabs(ground_truth_name)
        else os.path.join(args.game_dir, "ground_truth", ground_truth_name)
    )
    ground_truth = load_json(ground_truth_path) if os.path.exists(ground_truth_path) else None

    label_set_path = (
        args.label_set_file if os.path.isabs(args.label_set_file) else os.path.join(args.game_dir, args.label_set_file)
    )
    label_set = load_label_set(label_set_path)

    agents_cfg = config["agents"]
    requested_agents = len(agents_cfg) if args.agents_num <= 0 else args.agents_num
    if requested_agents != len(agents_cfg):
        raise SystemExit("agents_num must match number of agents in config")

    public_messages = 0 if str(condition["mode"]).lower() == "baseline" else int(condition["public_messages"])
    if str(condition["mode"]).lower() == "baseline":
        public_messages = 0
    if len(agents_cfg) and public_messages % len(agents_cfg) != 0:
        raise SystemExit("public_messages / rounds_num must be divisible by agents_num")

    random.seed(config.get("seed", 42))
    agent_names = [agent["name"] for agent in agents_cfg]
    starter = config["starter"]
    ordered_names = agent_names[:]
    if starter in ordered_names:
        idx = ordered_names.index(starter)
        ordered_names = ordered_names[idx:] + ordered_names[:idx]
    schedule_seed = int(config.get("seed", 42)) + 101
    public_schedule = generate_public_schedule(ordered_names, public_messages, seed=schedule_seed)

    line_ids = [line["id"] for line in scenario.get("lines", [])]
    agents: Dict[str, Dict[str, Any]] = {}
    for agent in agents_cfg:
        model_name = args.model or agent["model"]
        init_prompt = CyberInitialPromptJSON(
            args.game_dir,
            agent["name"],
            agent["file_name"],
            agent["incentive"],
            scenario,
            condition,
            label_set,
        )
        round_prompt = CyberSlotPromptJSON(agent["name"], args.game_dir, agent["file_name"], condition)
        agents[agent["name"]] = {
            "instance": CyberAgent(
                init_prompt,
                round_prompt,
                agent["name"],
                args.temp,
                model=model_name,
                azure=args.azure,
                role_id=derive_role_id(agent["file_name"]),
            )
        }

    output_dir = os.path.join(args.output_dir, args.exp_name)
    round_assign, _, public_start, history = create_outfiles(args, output_dir)
    del round_assign
    history.setdefault("content", {})
    history["content"].setdefault("round0", [])
    history["content"].setdefault("rounds", [])
    history["content"].setdefault("plan", {})
    history["content"].setdefault("finished_rounds", 0)
    history["content"].setdefault("finished_public_rounds", 0)
    history["content"].setdefault("failed_attempts", [])
    history["content"].setdefault("decision_trajectory", [])
    history["content"]["committee_snapshots"] = history["content"]["decision_trajectory"]
    history["content"]["slot_assignment"] = public_schedule
    history["content"]["condition"] = condition
    history["content"]["scenario"] = {k: v for k, v in scenario.items() if k != "author_notes"}
    history["content"]["scenario_id"] = scenario.get("scenario_id")
    history["content"]["scenario_title"] = scenario.get("title")
    history["content"]["schedule_seed"] = schedule_seed
    if ground_truth:
        history["content"]["ground_truth"] = ground_truth
    history["content"]["validation_stats"] = {
        "total_turns": 0,
        "total_attempts": 0,
        "successful_turns": 0,
        "schema_failure_turns": 0,
        "json_retry_count": 0,
        "citation_violation_turns": 0,
        "invalid_public_turns": 0,
        "leakage_turns": 0,
        "citation_mention_turns": 0,
    }
    history["content"]["run_status"] = "running"

    stem = Path(history["file"]).stem
    history["content"]["run_id"] = stem

    def record_failed_attempt(agent_name: str, phase: str, attempt: int, prompt: str, response_text: str, error: str) -> None:
        history["content"]["failed_attempts"].append(
            {
                "agent": agent_name,
                "phase": phase,
                "attempt": attempt,
                "prompt": prompt,
                "response_text": response_text,
                "error": error,
            }
        )

    def finalize_validation(validation: Dict[str, Any], *, attempts: int, schema_failed: bool) -> Dict[str, Any]:
        stats = history["content"]["validation_stats"]
        stats["total_turns"] += 1
        stats["total_attempts"] += attempts
        stats["json_retry_count"] += max(0, attempts - 1)
        if schema_failed:
            stats["schema_failure_turns"] += 1
        else:
            stats["successful_turns"] += 1
        if validation.get("citation_violation"):
            stats["citation_violation_turns"] += 1
        if validation.get("invalid_public_message"):
            stats["invalid_public_turns"] += 1
        if validation.get("leakage_flag"):
            stats["leakage_turns"] += 1
        if validation.get("citation_mention"):
            stats["citation_mention_turns"] += 1
        validation["schema_failure_after_retries"] = bool(schema_failed)
        return validation

    run_completed = True

    for agent in agents_cfg:
        attempts = 0
        parsed = None
        prompt = ""
        error = ""
        response_text = ""
        validation: Dict[str, Any] = {}
        while True:
            attempts += 1
            prompt, response = agents[agent["name"]]["instance"].execute_round0(history["content"])
            response_text = ensure_text(response)
            parsed, error, validation = parse_structured_response(
                response_text,
                line_ids=line_ids,
                label_set=label_set,
                public_message_max_words=int(condition["public_message_max_words"]),
                forbidden_public_tokens=list(condition["forbidden_public_tokens"]),
            )
            if parsed is not None:
                break
            record_failed_attempt(agent["name"], "round0", attempts, prompt, response_text, error)
            if max_attempts_per_turn and attempts >= max_attempts_per_turn:
                break
            prompt = build_retry_prompt(prompt, attempts, error)
            if args.retry_sleep:
                time.sleep(args.retry_sleep)
        validation = finalize_validation(validation, attempts=attempts, schema_failed=parsed is None)
        if parsed is None:
            history["content"]["run_status"] = "aborted_invalid_output"
            history["content"]["stopped_reason"] = "invalid_round0_output_after_retries"
            history["content"]["failed_turn"] = {
                "phase": "round0",
                "agent": agent["name"],
                "attempts": attempts,
                "error": error,
                "response_text": response_text,
            }
            write_file(history["content"], history["file"])
            run_completed = False
            break
        history = save_state(
            history,
            agent["name"],
            prompt,
            parsed,
            phase="round0",
            extra_fields={
                "phase": "round0",
                "turn_index": 0,
                "attempts": attempts,
                "validation": validation,
                "top1_label": parsed["assessment"]["ranked_findings"][0]["label"],
                "top1_severity": parsed["assessment"]["ranked_findings"][0]["severity"],
                "top1_exact": {
                    "label": parsed["assessment"]["ranked_findings"][0]["label"],
                    "severity": parsed["assessment"]["ranked_findings"][0]["severity"],
                },
                "rank1_citations": list(parsed["assessment"]["ranked_findings"][0].get("citations") or []),
                "accept": parsed["assessment"]["accept"],
                "block_reason": parsed["assessment"]["block_reason"],
            },
        )

    latest_by_agent = {
        slot["agent"]: {"assessment": slot["assessment"]}
        for slot in history["content"].get("round0", [])
        if slot.get("assessment")
    }
    if run_completed and latest_by_agent and not history["content"]["decision_trajectory"]:
        history["content"]["decision_trajectory"].append(
            make_committee_snapshot(0, "round0", latest_by_agent, public_turn_index=None, speaker=None)
        )
        write_file(history["content"], history["file"])

    for public_idx, current_agent in enumerate(public_schedule[public_start:], start=public_start):
        if not run_completed:
            break
        history["content"]["_current_final_turn"] = public_idx >= (
            len(public_schedule) - int(condition["final_turn_announcement_window"])
        )
        attempts = 0
        parsed = None
        prompt = ""
        error = ""
        response_text = ""
        validation = {}
        while True:
            attempts += 1
            prompt, response = agents[current_agent]["instance"].execute_round(history["content"], public_idx)
            response_text = ensure_text(response)
            parsed, error, validation = parse_structured_response(
                response_text,
                line_ids=line_ids,
                label_set=label_set,
                public_message_max_words=int(condition["public_message_max_words"]),
                forbidden_public_tokens=list(condition["forbidden_public_tokens"]),
            )
            if parsed is not None:
                break
            record_failed_attempt(current_agent, "public", attempts, prompt, response_text, error)
            if max_attempts_per_turn and attempts >= max_attempts_per_turn:
                break
            prompt = build_retry_prompt(prompt, attempts, error)
            if args.retry_sleep:
                time.sleep(args.retry_sleep)
        validation = finalize_validation(validation, attempts=attempts, schema_failed=parsed is None)
        if parsed is None:
            history["content"]["run_status"] = "aborted_invalid_output"
            history["content"]["stopped_reason"] = "invalid_public_output_after_retries"
            history["content"]["failed_turn"] = {
                "phase": "public",
                "agent": current_agent,
                "public_turn_index": public_idx,
                "attempts": attempts,
                "error": error,
                "response_text": response_text,
            }
            write_file(history["content"], history["file"])
            run_completed = False
            break

        history = save_state(
            history,
            current_agent,
            prompt,
            parsed,
            phase="public",
            extra_fields={
                "phase": "public",
                "turn_index": public_idx + 1,
                "public_turn_index": public_idx,
                "is_final_public_turn": bool(history["content"].get("_current_final_turn")),
                "attempts": attempts,
                "validation": validation,
                "top1_label": parsed["assessment"]["ranked_findings"][0]["label"],
                "top1_severity": parsed["assessment"]["ranked_findings"][0]["severity"],
                "top1_exact": {
                    "label": parsed["assessment"]["ranked_findings"][0]["label"],
                    "severity": parsed["assessment"]["ranked_findings"][0]["severity"],
                },
                "rank1_citations": list(parsed["assessment"]["ranked_findings"][0].get("citations") or []),
                "accept": parsed["assessment"]["accept"],
                "block_reason": parsed["assessment"]["block_reason"],
            },
        )
        latest_by_agent[current_agent] = {"assessment": parsed["assessment"]}
        history["content"]["decision_trajectory"].append(
            make_committee_snapshot(
                public_idx + 1,
                "public",
                latest_by_agent,
                public_turn_index=public_idx,
                speaker=current_agent,
            )
        )
        write_file(history["content"], history["file"])

    history["content"].pop("_current_final_turn", None)
    if run_completed and history["content"]["run_status"] == "running":
        history["content"]["run_status"] = "completed"
    summary = compute_metrics(history["content"], ground_truth)

    run_report = {
        "run_id": stem,
        "condition_id": condition["condition_id"],
        "scenario_id": scenario.get("scenario_id"),
        "headline_metrics": summary["headline_metrics"],
        "derived_metrics": summary["derived_metrics"],
        "appendix_debug": summary["appendix_debug"],
        "committee_final": summary["committee_final"],
        "decision_trajectory": summary["decision_trajectory"],
        "validation_stats": history["content"]["validation_stats"],
        "failed_attempts": history["content"].get("failed_attempts", []),
    }
    condition_aggregate = aggregate_condition_results([run_report], condition_id=condition["condition_id"])

    history["content"]["metrics"] = summary["headline_metrics"]
    history["content"]["derived_metrics"] = summary["derived_metrics"]
    history["content"]["appendix_debug"] = summary["appendix_debug"]
    history["content"]["committee_final"] = summary["committee_final"]
    history["content"]["condition_aggregate"] = condition_aggregate
    write_file(history["content"], history["file"])

    out_dir = Path(history["file"]).parent
    metrics_path = out_dir / f"metrics_{stem}.json"
    condition_json_path = out_dir / f"condition_summary_{stem}.json"
    condition_csv_path = out_dir / f"condition_headline_{stem}.csv"
    public_history_path = out_dir / f"public_history_{stem}.json"
    expert_csv_path = out_dir / f"expert_review_{stem}.csv"

    metrics_payload = {
        "run_report": run_report,
        "condition_aggregate": condition_aggregate,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    condition_json_path.write_text(json.dumps(condition_aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    condition_csv_row = {
        "condition_id": condition_aggregate["headline_metrics"]["condition_id"],
        "run_count": condition_aggregate["headline_metrics"]["run_count"],
        "FinalCorrectExact": condition_aggregate["headline_metrics"]["FinalCorrectExact"],
        "FinalCorrectType": condition_aggregate["headline_metrics"]["FinalCorrectType"],
        "FinalAgreementExact": condition_aggregate["headline_metrics"]["FinalAgreementExact"],
        "AnyAgreementExact": condition_aggregate["headline_metrics"]["AnyAgreementExact"],
        "SeverityBias": condition_aggregate["headline_metrics"]["SeverityBias"],
        "SeverityBiasMissingCount": condition_aggregate["headline_metrics"]["SeverityBiasMissingCount"],
        "TrustHygieneRate": condition_aggregate["headline_metrics"]["TrustHygieneRate"],
    }
    with condition_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(condition_csv_row.keys()))
        writer.writeheader()
        writer.writerow(condition_csv_row)

    public_history = [
        {
            "turn_index": slot.get("turn_index"),
            "public_turn_index": slot.get("public_turn_index"),
            "agent": slot["agent"],
            "public_answer": slot["public_answer"],
        }
        for slot in history["content"].get("rounds", [])
    ]
    public_history_path.write_text(json.dumps(public_history, indent=2, ensure_ascii=False), encoding="utf-8")

    final_committee = summary["committee_final"] or {}
    final_committee_exact = final_committee.get("committee_exact")
    final_committee_type = final_committee.get("committee_type")
    final_label = NO_CONSENSUS
    final_severity = ""
    if isinstance(final_committee_exact, dict):
        final_label = final_committee_exact["label"]
        final_severity = final_committee_exact["severity"]
    elif final_committee_type and final_committee_type != NO_CONSENSUS:
        final_label = final_committee_type

    expert_row = {
        "packet_id": scenario.get("scenario_id"),
        "condition_id": condition["condition_id"],
        "run_id": stem,
        "final_committee_label": final_label,
        "final_committee_severity": final_severity,
        "final_agreement_exact": summary["headline_metrics"].get("FinalAgreementExact"),
        "history_reference": str(Path(history["file"]).resolve()),
        "public_history_reference": str(public_history_path.resolve()),
        "q1_finding_type_correctness": "",
        "q1_reviewer_type": "",
        "q2_severity_appropriateness": "",
        "q2_reviewer_severity": "",
        "q3_evidence_support": "",
        "q4_report_defensibility": "",
        "q5_main_issue": "",
        "q6_minimal_fix_needed": "",
        "comment": "",
        "reviewer": "",
    }
    with expert_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(expert_row.keys()))
        writer.writeheader()
        writer.writerow(expert_row)

    print("=====")
    print(f"History file: {history['file']}")
    print(f"Metrics file: {metrics_path}")
    print(f"Condition summary JSON: {condition_json_path}")
    print(f"Condition headline CSV: {condition_csv_path}")
    print(f"Public history file: {public_history_path}")
    print(f"Expert review CSV: {expert_csv_path}")


if __name__ == "__main__":
    main()
