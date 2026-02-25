from __future__ import annotations

import json
from pathlib import Path

from ..core.schemas import AgentTurnOutput, EvidencePacket, ExpertReviewRow, GroundTruth


def schema_bundle() -> dict[str, dict]:
    return {
        "agent_turn_output": AgentTurnOutput.model_json_schema(),
        "evidence_packet": EvidencePacket.model_json_schema(),
        "ground_truth": GroundTruth.model_json_schema(),
        "expert_review_row": ExpertReviewRow.model_json_schema(),
    }


def render_json_contract_text() -> str:
    return (
        "Return ONLY one valid JSON object matching the AgentTurnOutput schema.\n"
        "No markdown, no prose outside JSON, no code fences.\n"
        "Fields:\n"
        "- private_notes (string; hidden from other agents)\n"
        "- private_plan (string; hidden from other agents)\n"
        "- public_message (string; shared)\n"
        "- assessment (object with ranked_findings [ranks 1..3], decision_summary)\n"
        "Rank 1 must cite 1-2 valid evidence line IDs. Severity is required for ranks 1-3.\n"
    )


def export_json_schemas(out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, schema in schema_bundle().items():
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        paths[name] = str(path)
    return paths

