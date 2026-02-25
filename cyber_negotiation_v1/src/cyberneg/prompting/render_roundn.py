from __future__ import annotations

import json
from typing import Optional

from ..core.schemas import PromptTemplateConfig, PublicTurnRecord, RoleInstructionConfig
from .json_contracts import render_json_contract_text


def render_roundn_prompt(
    *,
    roundn_prompt: PromptTemplateConfig,
    public_history: list[PublicTurnRecord],
    own_previous_private_notes: Optional[str],
    own_previous_private_plan: Optional[str],
    reminder_text: str,
    is_final_public_turn: bool,
    role_instruction: Optional[RoleInstructionConfig] = None,
    reinject_role_instruction: bool = False,
) -> str:
    sections: list[str] = [roundn_prompt.template.strip()]
    if reinject_role_instruction and role_instruction is not None:
        sections.append("Role-specific reminder:\n" + role_instruction.instruction_text.strip())
    if public_history:
        sections.append(
            "Public history:\n" + json.dumps([h.model_dump(mode="json") for h in public_history], indent=2)
        )
    else:
        sections.append("Public history:\n[]")
    if reminder_text.strip():
        sections.append("Reminder:\n" + reminder_text.strip())
    if own_previous_private_notes:
        sections.append("Your previous private notes (visible only to you):\n" + own_previous_private_notes)
    if own_previous_private_plan:
        sections.append("Your previous private plan (visible only to you):\n" + own_previous_private_plan)
    if is_final_public_turn:
        sections.append(
            "FINAL PUBLIC TURN ANNOUNCEMENT: This is the final public negotiation turn. Commit to your most defensible assessment."
        )
    sections.append(render_json_contract_text())
    return "\n\n".join(sections)
