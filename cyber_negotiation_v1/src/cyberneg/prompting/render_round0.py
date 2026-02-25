from __future__ import annotations

import json

from ..core.schemas import EvidencePacket, PromptTemplateConfig, RoleInstructionConfig
from .json_contracts import render_json_contract_text


def render_round0_prompt(
    *,
    global_prompt: PromptTemplateConfig,
    role_instruction: RoleInstructionConfig,
    prior_text: str,
    evidence_packet: EvidencePacket,
) -> str:
    visible_packet = json.dumps(evidence_packet.visible_payload(), indent=2, ensure_ascii=False)
    sections = [
        global_prompt.template.strip(),
        role_instruction.instruction_text.strip(),
    ]
    if prior_text.strip():
        sections.append(prior_text.strip())
    sections.append("Evidence packet (all evidence visible in V1):\n" + visible_packet)
    sections.append(render_json_contract_text())
    return "\n\n".join(s for s in sections if s)

