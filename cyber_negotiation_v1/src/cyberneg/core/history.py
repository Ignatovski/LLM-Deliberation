from __future__ import annotations

from typing import Optional

from .enums import RoleId, TurnPhase
from .schemas import AgentTurnOutput, PublicTurnRecord, TurnResultLog


def latest_private_plan(turns: list[TurnResultLog], role_id: RoleId) -> Optional[str]:
    for turn in reversed(turns):
        if turn.role_id == role_id and turn.final_output is not None:
            return turn.final_output.private_plan
    return None


def latest_private_notes(turns: list[TurnResultLog], role_id: RoleId) -> Optional[str]:
    for turn in reversed(turns):
        if turn.role_id == role_id and turn.final_output is not None:
            return turn.final_output.private_notes
    return None


def visible_public_history(turns: list[TurnResultLog]) -> list[PublicTurnRecord]:
    records: list[PublicTurnRecord] = []
    for turn in turns:
        if turn.phase != TurnPhase.PUBLIC or turn.final_output is None or turn.public_turn_index is None:
            continue
        records.append(
            PublicTurnRecord(
                public_turn_index=turn.public_turn_index,
                role_id=turn.role_id,
                public_message=turn.final_output.public_message,
            )
        )
    return records


def latest_assessment_by_role(turns: list[TurnResultLog]) -> dict[RoleId, AgentTurnOutput]:
    out: dict[RoleId, AgentTurnOutput] = {}
    for turn in turns:
        if turn.final_output is None:
            continue
        out[turn.role_id] = turn.final_output
    return out

