from __future__ import annotations

import pytest
from pydantic import ValidationError

from cyberneg.core.schemas import ConditionConfig, EvidencePacket


def test_evidence_packet_rejects_duplicate_line_ids() -> None:
    with pytest.raises(ValidationError):
        EvidencePacket.model_validate(
            {
                "scenario_id": "s1",
                "title": "dup ids",
                "source_family": "generic webapp",
                "difficulty": "easy",
                "label_set_id": "default",
                "lines": [
                    {"id": "L001", "text": "a"},
                    {"id": "L001", "text": "b"},
                ],
            }
        )


def test_negotiation_condition_requires_public_messages_divisible_by_three() -> None:
    with pytest.raises(ValidationError):
        ConditionConfig.model_validate(
            {
                "condition_id": "C1",
                "enabled": True,
                "mode": "negotiation",
                "models_by_role": {"R": "m1", "C": "m1", "K": "m1"},
                "runtime": {
                    "public_messages": 5,
                    "json_max_retries": 3,
                    "public_message_min_words": 80,
                    "public_message_max_words": 150,
                    "public_message_hard_cap_words": 220,
                },
            }
        )

