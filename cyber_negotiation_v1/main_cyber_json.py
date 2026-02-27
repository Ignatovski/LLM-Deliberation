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
  "scratchpad": "<SCRATCHPAD>…</SCRATCHPAD>",
  "answer": "<ANSWER>Your public negotiation message only.</ANSWER>\n<ASSESSMENT>{\"ranked_findings\":[...],\"decision_summary\":\"...\"}</ASSESSMENT>",
  "plan": "<PLAN>…</PLAN>"
}

Rules:
- scratchpad: private reasoning only; keep it concise.
- answer: must contain exactly one public <ANSWER>...</ANSWER> block and one hidden <ASSESSMENT>{...}</ASSESSMENT> JSON object.
- plan: private next-step notes; omit or use "<PLAN></PLAN>" if none.
- assessment.ranked_findings must contain ranks 1, 2, and 3.
- Severity is required for ranks 1-3 and must be one of Compliance, Info, Low, Medium, High.
- Rank 1 must cite 1-2 valid evidence line IDs.
- Public message should be natural discussion text, 80-150 words, and must not expose private scratchpad or hidden planning.
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
            (
                f"Visible scenario metadata: scenario_id={scenario['scenario_id']}, title={scenario['title']}, "
                f"source_family={scenario['source_family']}, difficulty={scenario['difficulty']}"
            ),
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
        if self.condition.get("reinject_role_instruction_roundn"):
            prompt += "Role reminder:\n" + self.personal_text + "\n"
        if self.condition.get("reminder_text"):
            prompt += f"Reminder: {str(self.condition['reminder_text']).strip()}\n"
        if last_plan:
            prompt += f"Your previous notes were <PREV_PLAN>{last_plan}</PREV_PLAN>.\n"
        if history.get("_current_final_turn"):
            prompt += "This is the final public turn. Commit to your most defensible assessment.\n"
        prompt += f"Current public turn index: {round_idx}\n"
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


def make_fallback_parsed(label_set: Dict[str, Any], line_ids: List[str], message: str, reason: str) -> Dict[str, Any]:
    top_label = "NoFinding" if "NoFinding" in label_set.get("labels", []) else label_set.get("labels", ["Other"])[0]
    second_label = "Other" if "Other" in label_set.get("labels", []) else top_label
    assessment = {
        "ranked_findings": [
            {
                "rank": 1,
                "label": top_label,
                "severity": "Info",
                "citations": line_ids[:1],
                "confidence": 0.0,
                "rationale": reason,
            },
            {
                "rank": 2,
                "label": second_label,
                "severity": "Info",
                "citations": [],
                "confidence": 0.0,
                "rationale": "Fallback secondary candidate.",
            },
            {
                "rank": 3,
                "label": top_label,
                "severity": "Info",
                "citations": [],
                "confidence": 0.0,
                "rationale": "Fallback tertiary candidate.",
            },
        ],
        "decision_summary": "Fallback assessment emitted after repeated invalid JSON output.",
    }
    return {
        "scratchpad": f"<SCRATCHPAD>{reason}</SCRATCHPAD>",
        "answer": "<ANSWER>" + message + "</ANSWER>\n<ASSESSMENT>" + json.dumps(assessment, ensure_ascii=False) + "</ASSESSMENT>",
        "plan": "<PLAN></PLAN>",
        "public_answer": message,
        "assessment": assessment,
    }


def main():
    parser = argparse.ArgumentParser(description="Cyber negotiation game (polynomial-style JSON runner)")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--agents_num", type=int, default=3)
    parser.add_argument("--rounds_num", type=int, default=6, help="Fallback public message count if condition file omits it")
    parser.add_argument("--output_dir", type=str, default="./games_descriptions/cyber_game/output/")
    parser.add_argument("--game_dir", type=str, default="./games_descriptions/cyber_game")
    parser.add_argument("--config_file", type=str, default="config.txt", help="Polynomial-style agent config file")
    parser.add_argument("--condition_file", type=str, default="conditions/C1.txt", help="Polynomial-style key=value condition file")
    parser.add_argument("--scenario_file", type=str, default="", help="Scenario JSON file name under scenarios/")
    parser.add_argument("--ground_truth_file", type=str, default="", help="Ground-truth JSON file name under ground_truth/")
    parser.add_argument("--label_set_file", type=str, default="label_sets/default_findings_v1.json")
    parser.add_argument("--exp_name", type=str, default="cyber_demo_json")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--output_file", type=str, default="history.json")
    parser.add_argument("--azure", action="store_true", help="Use Azure Responses provider for non-mock non-Claude models")
    parser.add_argument("--model", type=str, default="", help="Optional model override for all agents")
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument("--max_attempts_per_turn", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=0.0)
    args = parser.parse_args()

    load_env_file(args.env_file)

    config_path = os.path.join(args.game_dir, args.config_file)
    condition_path = os.path.join(args.game_dir, args.condition_file)
    config = read_config(config_path)
    condition = read_condition_config(condition_path)

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
    if args.agents_num != len(agents_cfg):
        raise SystemExit("agents_num must match number of agents in config")

    public_messages = int(condition.get("public_messages") or args.rounds_num)
    if public_messages % args.agents_num != 0:
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
                line_ids=line_ids,
                label_set=label_set,
                timeout_seconds=int(condition.get("per_call_timeout_seconds", 30)),
                invalid_json_attempts_per_turn=int(condition.get("invalid_json_attempts_per_turn", 0)),
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
    history["content"].setdefault("validation_failures", [])
    history["content"].setdefault("committee_snapshots", [])
    history["content"]["slot_assignment"] = public_schedule
    history["content"]["condition"] = condition
    history["content"]["scenario"] = {k: v for k, v in scenario.items() if k != "author_notes"}
    history["content"]["scenario_id"] = scenario.get("scenario_id")
    history["content"]["scenario_title"] = scenario.get("title")
    if ground_truth:
        history["content"]["ground_truth"] = ground_truth
    history["content"]["validation_stats"] = {
        "total_turns": 0,
        "total_attempts": 0,
        "successful_turns": 0,
        "failed_turns": 0,
        "json_retry_count": 0,
        "json_failures": 0,
        "schema_validation_failures": 0,
        "citation_count_violations": 0,
        "message_length_violations": 0,
        "citation_total_rank1": 0,
        "citation_valid_rank1": 0,
    }

    def update_validation(success: bool, attempts: int, warnings: List[str], parsed: Optional[Dict[str, Any]], error: str) -> None:
        stats = history["content"]["validation_stats"]
        stats["total_turns"] += 1
        stats["total_attempts"] += attempts
        stats["json_retry_count"] += max(0, attempts - 1)
        if success:
            stats["successful_turns"] += 1
        else:
            stats["failed_turns"] += 1
            stats["json_failures"] += 1
        if "public_message_target_range" in warnings:
            stats["message_length_violations"] += 1
        if parsed is not None:
            rank1 = next(
                (item for item in parsed["assessment"]["ranked_findings"] if item.get("rank") == 1),
                None,
            )
            if rank1 is not None:
                citations = rank1.get("citations", [])
                stats["citation_total_rank1"] += len(citations)
                stats["citation_valid_rank1"] += len(citations)
        if error:
            history["content"]["validation_failures"].append(error)

    run_started = time.monotonic()
    token_budget_limit = condition.get("token_budget_limit")
    total_attempt_counter = 0

    for agent in agents_cfg:
        attempts = 0
        parsed = None
        prompt = ""
        warnings: List[str] = []
        error = ""
        while True:
            attempts += 1
            prompt, response = agents[agent["name"]]["instance"].execute_round0(history["content"])
            parsed, error, warnings = parse_structured_response(
                ensure_text(response),
                line_ids=line_ids,
                label_set=label_set,
                public_message_min_words=int(condition["public_message_min_words"]),
                public_message_max_words=int(condition["public_message_max_words"]),
                public_message_hard_cap_words=int(condition["public_message_hard_cap_words"]),
            )
            if parsed is not None:
                break
            if args.max_attempts_per_turn and attempts >= args.max_attempts_per_turn:
                break
            prompt = build_retry_prompt(prompt, attempts, error)
            if args.retry_sleep:
                time.sleep(args.retry_sleep)
        total_attempt_counter += attempts
        update_validation(parsed is not None, attempts, warnings, parsed, error if parsed is None else "")
        if parsed is None:
            parsed = make_fallback_parsed(
                label_set,
                line_ids,
                "I cannot support a final committee claim from this turn because my previous output was invalid. Keep the discussion anchored in the packet and rely on the remaining committee record.",
                "Structured output failure during Round 0.",
            )
        history = save_state(
            history,
            agent["name"],
            prompt,
            parsed,
            phase="round0",
            extra_fields={"phase": "round0", "attempts": attempts, "warnings": warnings},
        )

    latest_by_agent = {
        slot["agent"]: {"assessment": slot["assessment"]}
        for slot in history["content"].get("round0", [])
        if slot.get("assessment")
    }

    for public_idx, current_agent in enumerate(public_schedule[public_start:], start=public_start):
        if condition.get("per_run_wallclock_limit_seconds") and (
            time.monotonic() - run_started
        ) > int(condition["per_run_wallclock_limit_seconds"]):
            history["content"]["stopped_reason"] = "wallclock_limit_exceeded"
            break
        if token_budget_limit is not None and total_attempt_counter > int(token_budget_limit):
            history["content"]["stopped_reason"] = "token_budget_placeholder_exceeded"
            break

        history["content"]["_current_final_turn"] = public_idx >= (
            len(public_schedule) - int(condition["final_turn_announcement_window"])
        )
        attempts = 0
        parsed = None
        prompt = ""
        warnings = []
        error = ""
        while True:
            attempts += 1
            prompt, response = agents[current_agent]["instance"].execute_round(history["content"], public_idx)
            parsed, error, warnings = parse_structured_response(
                ensure_text(response),
                line_ids=line_ids,
                label_set=label_set,
                public_message_min_words=int(condition["public_message_min_words"]),
                public_message_max_words=int(condition["public_message_max_words"]),
                public_message_hard_cap_words=int(condition["public_message_hard_cap_words"]),
            )
            if parsed is not None:
                break
            if args.max_attempts_per_turn and attempts >= args.max_attempts_per_turn:
                break
            prompt = build_retry_prompt(prompt, attempts, error)
            if args.retry_sleep:
                time.sleep(args.retry_sleep)
        total_attempt_counter += attempts
        update_validation(parsed is not None, attempts, warnings, parsed, error if parsed is None else "")
        if parsed is None:
            parsed = make_fallback_parsed(
                label_set,
                line_ids,
                "I am holding my prior position because my previous output was invalid. The safest course is to keep the committee anchored in cited packet lines and avoid inventing new claims.",
                "Structured output failure during public negotiation.",
            )

        history = save_state(
            history,
            current_agent,
            prompt,
            parsed,
            phase="public",
            extra_fields={
                "phase": "public",
                "public_turn_index": public_idx,
                "is_final_public_turn": bool(history["content"].get("_current_final_turn")),
                "attempts": attempts,
                "warnings": warnings,
            },
        )
        latest_by_agent[current_agent] = {"assessment": parsed["assessment"]}
        history["content"]["committee_snapshots"].append(make_committee_snapshot(public_idx, latest_by_agent))
        write_file(history["content"], history["file"])

    history["content"].pop("_current_final_turn", None)
    summary = compute_metrics(history["content"], ground_truth)
    history["content"]["metrics"] = summary["metrics"]
    history["content"]["committee_final"] = summary["committee_final"]
    write_file(history["content"], history["file"])

    stem = Path(history["file"]).stem
    out_dir = Path(history["file"]).parent
    metrics_path = out_dir / f"metrics_{stem}.json"
    public_history_path = out_dir / f"public_history_{stem}.json"
    expert_csv_path = out_dir / f"expert_review_{stem}.csv"

    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    public_history = [
        {
            "public_turn_index": slot.get("public_turn_index"),
            "agent": slot["agent"],
            "public_answer": slot["public_answer"],
        }
        for slot in history["content"].get("rounds", [])
    ]
    public_history_path.write_text(json.dumps(public_history, indent=2, ensure_ascii=False), encoding="utf-8")

    expert_row = {
        "run_id": stem,
        "condition_id": condition.get("condition_id"),
        "scenario_id": scenario.get("scenario_id"),
        "final_committee_label": (summary["committee_final"] or {}).get("committee_exact_label")
        or (summary["committee_final"] or {}).get("committee_type_label"),
        "final_committee_severity": (summary["committee_final"] or {}).get("committee_exact_severity")
        or (summary["committee_final"] or {}).get("committee_majority_severity"),
        "final_agreement_exact": (summary["committee_final"] or {}).get("full_agreement_exact"),
        "transcript_reference": str(Path(history["file"]).resolve()),
        "agent_final_outputs_reference": str(Path(history["file"]).resolve()),
        "expert_label": "",
        "expert_severity": "",
        "expert_score_argument_quality": "",
        "expert_score_evidence_relevance": "",
        "expert_score_defensibility_for_report": "",
        "expert_would_accept_in_QA": "",
        "expert_comments": "",
    }
    with expert_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(expert_row.keys()))
        writer.writeheader()
        writer.writerow(expert_row)

    print("=====")
    print(f"History file: {history['file']}")
    print(f"Metrics file: {metrics_path}")
    print(f"Public history file: {public_history_path}")
    print(f"Expert review CSV: {expert_csv_path}")
    print("Final metrics:", json.dumps(summary["metrics"], ensure_ascii=False))
    if summary["committee_final"]:
        print("Final committee:", json.dumps(summary["committee_final"], ensure_ascii=False))


if __name__ == "__main__":
    main()
