from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def write_file(log_dict, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(log_dict, outfile, indent=2, ensure_ascii=False)


def create_outfiles(args, output_dir):
    history = {}
    os.makedirs(output_dir, exist_ok=True)
    if args.restart:
        history["file"] = os.path.join(output_dir, args.output_file)
        with open(history["file"], "r", encoding="utf-8") as file:
            history["content"] = json.load(file)
        round_start = int(history["content"].get("finished_rounds", 0))
        round_assign = history["content"].get("slot_assignment", [])
        public_start = int(history["content"].get("finished_public_rounds", 0))
    else:
        time_str = time.strftime("%H_%M_%S", time.localtime())
        output_file = os.path.join(output_dir, args.output_file.split(".json")[0] + time_str + ".json")
        round_start = 0
        public_start = 0
        history = {"file": output_file, "content": {}}
        round_assign = []
    return round_assign, round_start, public_start, history


def save_conversation(
    history,
    agent_name,
    full_answer,
    prompt,
    phase="public",
    extra_fields: Optional[Dict] = None,
    *,
    public_answer: str,
    private_notes: str,
    private_plan: str,
    assessment: Optional[Dict[str, Any]],
):
    history["content"].setdefault("rounds", [])
    history["content"].setdefault("round0", [])
    history["content"].setdefault("plan", {})
    history["content"].setdefault("finished_rounds", 0)
    history["content"].setdefault("finished_public_rounds", 0)
    if phase == "public":
        history["content"]["finished_rounds"] += 1
        history["content"]["finished_public_rounds"] += 1
    elif phase == "round0":
        history["content"]["finished_rounds"] += 1

    slot = {
        "agent": agent_name,
        "prompt": prompt,
        "full_answer": full_answer,
        "public_answer": public_answer,
        "private_notes": private_notes,
        "private_plan": private_plan,
        "assessment": assessment,
    }
    if extra_fields:
        slot.update(extra_fields)
    if phase == "round0":
        history["content"]["round0"].append(slot)
    else:
        history["content"]["rounds"].append(slot)

    if private_plan:
        history["content"]["plan"].setdefault(agent_name, []).append(private_plan)

    write_file(history["content"], history["file"])
    return history
