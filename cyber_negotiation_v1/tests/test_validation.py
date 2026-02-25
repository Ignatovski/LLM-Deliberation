from __future__ import annotations

import json

from cyberneg.core.enums import RoleId, TurnPhase
from cyberneg.core.schemas import LabelSetConfig, ProviderAttemptLog
from cyberneg.core.validators import build_turn_validation_context, strict_json_retry_loop
from cyberneg.providers.base import BaseProvider, ProviderCallContext, ProviderResponse
from cyberneg.providers.mock_provider import MockProvider


class SequenceProvider(BaseProvider):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(provider_name="seq", provider_kind="mock", model_name="seq")
        self._responses = list(responses)
        self._idx = 0

    def generate(self, prompt: str, ctx: ProviderCallContext) -> ProviderResponse:
        text = self._responses[self._idx]
        self._idx += 1
        return ProviderResponse(text=text)


def _ctx() -> ProviderCallContext:
    return ProviderCallContext(
        provider_name="mock",
        provider_kind="mock",
        model_name="mock",
        phase=TurnPhase.PUBLIC,
        role_id=RoleId.R,
        public_turn_index=0,
        turn_id="t0",
        metadata={
            "line_ids": ["L001", "L002", "L003"],
            "label_set_labels": ["XSS_Reflected", "NoFinding", "Other"],
        },
    )


def _validation_ctx():
    label_set = LabelSetConfig(
        label_set_id="ls1",
        labels=["XSS_Reflected", "NoFinding", "Other"],
        aliases={"sqli": "Other"},
    )
    return build_turn_validation_context(
        label_set,
        ["L001", "L002", "L003"],
        public_message_min_words=1,
        public_message_max_words=200,
        public_message_hard_cap_words=300,
    )


def _attempt_factory(attempt_idx: int, prompt_text: str) -> ProviderAttemptLog:
    return ProviderAttemptLog(
        attempt_index=attempt_idx,
        provider_name="mock",
        provider_kind="mock",
        model_name="mock",
        phase=TurnPhase.PUBLIC,
        role_id=RoleId.R,
        public_turn_index=0,
        prompt_text=prompt_text,
    )


def test_strict_json_retry_loop_retries_invalid_json_then_succeeds() -> None:
    provider = MockProvider(invalid_json_attempts_per_turn=1)
    result = strict_json_retry_loop(
        provider=provider,
        provider_context=_ctx(),
        base_prompt="Return JSON",
        validation_context=_validation_ctx(),
        max_retries=3,
        attempt_log_factory=_attempt_factory,
    )

    assert result.success is True
    assert len(result.attempts) == 2
    assert result.attempts[0].json_valid is False
    assert result.attempts[0].validation_errors[0].code == "invalid_json"
    assert result.attempts[1].schema_valid is True


def test_strict_json_retry_loop_logs_schema_business_validation_errors() -> None:
    invalid_payload = {
        "private_notes": "n",
        "private_plan": "p",
        "public_message": "short but valid length for this test",
        "assessment": {
            "ranked_findings": [
                {"rank": 1, "label": "MadeUpLabel", "severity": "Low", "citations": []},
                {"rank": 2, "label": "Other", "severity": "Info", "citations": []},
                {"rank": 3, "label": "NoFinding", "severity": "Info", "citations": []},
            ],
            "decision_summary": "summary",
        },
    }
    valid_payload = {
        "private_notes": "n",
        "private_plan": "p",
        "public_message": "this public message has enough words for the test and remains within the configured bounds",
        "assessment": {
            "ranked_findings": [
                {"rank": 1, "label": "XSS_Reflected", "severity": "Low", "citations": ["L001"]},
                {"rank": 2, "label": "Other", "severity": "Info", "citations": []},
                {"rank": 3, "label": "NoFinding", "severity": "Info", "citations": []},
            ],
            "decision_summary": "summary",
        },
    }
    provider = SequenceProvider([json.dumps(invalid_payload), json.dumps(valid_payload)])
    result = strict_json_retry_loop(
        provider=provider,
        provider_context=_ctx(),
        base_prompt="Return JSON",
        validation_context=_validation_ctx(),
        max_retries=2,
        attempt_log_factory=_attempt_factory,
    )

    assert result.success is True
    assert len(result.attempts) == 2
    first_errors = {e.code for e in result.attempts[0].validation_errors}
    assert "invalid_label" in first_errors
    assert "rank1_citation_count" in first_errors
    assert result.attempts[0].json_valid is True
    assert result.attempts[0].schema_valid is False

