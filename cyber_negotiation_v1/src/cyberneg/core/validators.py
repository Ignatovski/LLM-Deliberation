from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from .schemas import (
    AgentTurnOutput,
    LabelSetConfig,
    ProviderAttemptLog,
    ProviderUsage,
    ValidationMessage,
    utc_now_iso,
)
from ..providers.base import BaseProvider, ProviderCallContext, ProviderResponse


@dataclass
class TurnValidationContext:
    allowed_labels: set[str]
    label_aliases: dict[str, str]
    valid_line_ids: set[str]
    public_message_min_words: int
    public_message_max_words: int
    public_message_hard_cap_words: int


@dataclass
class StrictRetryResult:
    success: bool
    output: Optional[AgentTurnOutput]
    attempts: list[ProviderAttemptLog]
    final_status: str
    warnings: list[ValidationMessage]


def build_turn_validation_context(
    label_set: LabelSetConfig,
    line_ids: list[str],
    *,
    public_message_min_words: int,
    public_message_max_words: int,
    public_message_hard_cap_words: int,
) -> TurnValidationContext:
    return TurnValidationContext(
        allowed_labels=set(label_set.labels),
        label_aliases={k.strip().lower(): v for k, v in label_set.aliases.items()},
        valid_line_ids=set(line_ids),
        public_message_min_words=public_message_min_words,
        public_message_max_words=public_message_max_words,
        public_message_hard_cap_words=public_message_hard_cap_words,
    )


def _word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def _norm_label(raw: str, ctx: TurnValidationContext) -> tuple[str, bool]:
    key = raw.strip().lower()
    if key in ctx.label_aliases:
        return ctx.label_aliases[key], True
    return raw, False


def _validate_business_rules(model: AgentTurnOutput, ctx: TurnValidationContext) -> tuple[list[ValidationMessage], list[ValidationMessage]]:
    errors: list[ValidationMessage] = []
    warnings: list[ValidationMessage] = []

    # Label normalization / validation
    for item in model.assessment.ranked_findings:
        normed, aliased = _norm_label(item.label, ctx)
        if aliased:
            item.label = normed
        if item.label not in ctx.allowed_labels:
            errors.append(
                ValidationMessage(
                    code="invalid_label",
                    message=f"Label '{item.label}' is not in the scenario label set",
                    details={"rank": item.rank},
                )
            )
        if item.rank == 1 and not (1 <= len(item.citations) <= 2):
            errors.append(
                ValidationMessage(
                    code="rank1_citation_count",
                    message="Rank 1 must include 1-2 citations",
                    details={"count": len(item.citations)},
                )
            )
        for cid in item.citations:
            if cid not in ctx.valid_line_ids:
                errors.append(
                    ValidationMessage(
                        code="invalid_citation_line_id",
                        message=f"Citation '{cid}' not found in evidence line IDs",
                        details={"rank": item.rank, "citation": cid},
                    )
                )

    wc = _word_count(model.public_message)
    if wc > ctx.public_message_hard_cap_words:
        errors.append(
            ValidationMessage(
                code="public_message_hard_cap",
                message="Public message exceeds hard cap",
                details={"word_count": wc, "hard_cap": ctx.public_message_hard_cap_words},
            )
        )
    if wc < ctx.public_message_min_words or wc > ctx.public_message_max_words:
        warnings.append(
            ValidationMessage(
                code="public_message_target_range",
                message="Public message is outside target length range",
                details={
                    "word_count": wc,
                    "target_min": ctx.public_message_min_words,
                    "target_max": ctx.public_message_max_words,
                },
            )
        )

    return errors, warnings


def validate_turn_output_json(raw_text: str, ctx: TurnValidationContext) -> tuple[Optional[AgentTurnOutput], list[ValidationMessage], list[ValidationMessage], bool]:
    try:
        parsed_obj = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, [ValidationMessage(code="invalid_json", message=str(exc))], [], False

    try:
        model = AgentTurnOutput.model_validate(parsed_obj)
    except ValidationError as exc:
        errs = [
            ValidationMessage(
                code="schema_validation_error",
                message=e.get("msg", "schema validation error"),
                details={"loc": [str(x) for x in e.get("loc", ())], "type": e.get("type")},
            )
            for e in exc.errors()
        ]
        return None, errs, [], True

    biz_errors, warnings = _validate_business_rules(model, ctx)
    if biz_errors:
        return None, biz_errors, warnings, True
    return model, [], warnings, True


def build_retry_prompt(base_prompt: str, attempt_idx: int, errors: list[ValidationMessage]) -> str:
    error_lines = "\n".join(f"- {e.code}: {e.message}" for e in errors)
    return (
        base_prompt
        + "\n\nYour previous response was invalid. Return ONLY valid JSON matching the required schema.\n"
        + f"Attempt: {attempt_idx}\nValidation errors:\n{error_lines}\n"
    )


def strict_json_retry_loop(
    *,
    provider: BaseProvider,
    provider_context: ProviderCallContext,
    base_prompt: str,
    validation_context: TurnValidationContext,
    max_retries: int,
    attempt_log_factory: Callable[[int, str], ProviderAttemptLog],
) -> StrictRetryResult:
    attempts: list[ProviderAttemptLog] = []
    warnings: list[ValidationMessage] = []
    current_prompt = base_prompt

    if max_retries <= 0:
        raise ValueError("max_retries must be > 0")

    for attempt_idx in range(1, max_retries + 1):
        log = attempt_log_factory(attempt_idx, current_prompt)
        t0 = time.time()
        try:
            response: ProviderResponse = provider.generate(current_prompt, provider_context)
            t1 = time.time()
            log.finished_at = utc_now_iso()
            log.duration_ms = int((t1 - t0) * 1000)
            log.raw_response_text = response.text
            log.response_metadata = dict(response.response_metadata or {})
            log.request_metadata = dict(response.request_metadata or {})
            log.usage = (
                ProviderUsage.model_validate(response.usage)
                if isinstance(response.usage, dict)
                else response.usage
            )
            parsed, errs, warns, json_ok = validate_turn_output_json(response.text, validation_context)
            log.json_valid = json_ok
            log.schema_valid = parsed is not None
            log.validation_errors = errs
            log.validation_warnings = warns
            attempts.append(log)
            if parsed is not None:
                warnings.extend(warns)
                return StrictRetryResult(
                    success=True,
                    output=parsed,
                    attempts=attempts,
                    final_status="success",
                    warnings=warnings,
                )
            current_prompt = build_retry_prompt(base_prompt, attempt_idx, errs)
        except Exception as exc:  # noqa: BLE001 - logging exact provider/runtime failure
            t1 = time.time()
            log.finished_at = utc_now_iso()
            log.duration_ms = int((t1 - t0) * 1000)
            log.exception_type = type(exc).__name__
            log.exception_message = str(exc)
            attempts.append(log)
            current_prompt = build_retry_prompt(
                base_prompt,
                attempt_idx,
                [ValidationMessage(code="provider_exception", message=str(exc))],
            )

    return StrictRetryResult(
        success=False,
        output=None,
        attempts=attempts,
        final_status="exhausted_retries",
        warnings=warnings,
    )
