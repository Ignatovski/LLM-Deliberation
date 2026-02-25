from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.enums import ConditionMode, RoleId, TurnPhase, TurnStatus
from ..core.metrics import compute_run_metrics
from ..core.schemas import ConditionConfig, EvidencePacket, GroundTruth, ProviderAttemptLog, TurnResultLog, ValidationStats
from ..core.validators import build_turn_validation_context, strict_json_retry_loop
from ..io.exports import (
    build_summary_table_json,
    export_expert_review_csv,
    export_run_metrics_files,
    export_siemens_overview_package,
    generate_basic_plots,
)
from ..io.loaders import LoadedExperimentBundle, build_provider_from_model_ref
from ..io.manifests import finalize_manifest, make_run_manifest
from ..io.storage import ensure_dir, write_json, write_jsonl
from ..prompting.json_contracts import export_json_schemas, render_json_contract_text
from ..providers.base import ProviderCallContext


@dataclass
class BaselineRunResult:
    run_dir: Path
    manifest: Any
    run_metrics: Any


def _attempts_total_tokens(attempts: list[ProviderAttemptLog]) -> int:
    total = 0
    for attempt in attempts:
        usage = attempt.usage
        if usage is None:
            continue
        if usage.total_tokens is not None:
            total += int(usage.total_tokens)
        elif usage.input_tokens is not None or usage.output_tokens is not None:
            total += int(usage.input_tokens or 0) + int(usage.output_tokens or 0)
    return total


def _attempt_factory(provider, phase: TurnPhase, role_id: RoleId):
    def factory(attempt_idx: int, prompt_text: str) -> ProviderAttemptLog:
        return ProviderAttemptLog(
            attempt_index=attempt_idx,
            provider_name=provider.provider_name,
            provider_kind=provider.provider_kind,
            model_name=provider.model_name,
            phase=phase,
            role_id=role_id,
            prompt_text=prompt_text,
        )

    return factory


def _update_validation_stats(stats: ValidationStats, turn: TurnResultLog) -> None:
    stats.total_turns += 1
    stats.total_attempts += len(turn.attempts)
    stats.json_retry_count += max(0, len(turn.attempts) - 1)
    if turn.status == TurnStatus.SUCCESS:
        stats.successful_turns += 1
    else:
        stats.failed_turns += 1
        if turn.status == TurnStatus.FAILED_JSON:
            stats.json_failures += 1
    for a in turn.attempts:
        if a.json_valid and not a.schema_valid:
            stats.schema_validation_failures += 1
    for warn in turn.validation_warnings:
        if warn.code == "public_message_target_range":
            stats.message_length_violations += 1
    if turn.final_output is not None:
        rank1 = next((x for x in turn.final_output.assessment.ranked_findings if x.rank == 1), None)
        if rank1 is not None:
            stats.citation_total_rank1 += len(rank1.citations)
            stats.citation_valid_rank1 += len(rank1.citations)


def _baseline_prompt(bundle: LoadedExperimentBundle, condition: ConditionConfig, scenario: EvidencePacket) -> str:
    import json

    sections = [bundle.prompts["baseline"].template.strip()]
    if condition.priors.text.strip():
        sections.append(condition.priors.text.strip())
    sections.append("Evidence packet (all evidence visible in V1):\n" + json.dumps(scenario.visible_payload(), indent=2))
    sections.append(render_json_contract_text())
    return "\n\n".join(sections)


def run_baseline_once(
    *,
    bundle: LoadedExperimentBundle,
    condition: ConditionConfig,
    scenario: EvidencePacket,
    ground_truth: GroundTruth | None,
    session_dir: Path,
    run_seed: int,
) -> BaselineRunResult:
    if condition.mode != ConditionMode.BASELINE:
        raise ValueError("run_baseline_once requires a baseline condition")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_dir = ensure_dir(session_dir / condition.condition_id / scenario.scenario_id / run_id)
    transcript_dir = ensure_dir(run_dir / "transcript")
    provider_dir = ensure_dir(run_dir / "provider")

    runtime = condition.runtime
    run_started = time.monotonic()
    label_set = bundle.label_sets[scenario.label_set_id]
    line_ids = [ln.id for ln in scenario.lines]
    provider = build_provider_from_model_ref(
        bundle=bundle,
        model_ref=condition.baseline_model or "",
        timeout_override=runtime.provider_timeout_seconds,
        mock_behavior_override=bundle.experiment.mock.behavior.model_dump(mode="json"),
    )

    manifest = make_run_manifest(
        run_id=run_id,
        experiment_id=bundle.experiment.experiment_id,
        condition_id=condition.condition_id,
        scenario_id=scenario.scenario_id,
        mode=condition.mode,
        seed=run_seed,
        config_refs=bundle.config_refs,
        runtime_limits=runtime.model_dump(mode="json"),
    )
    manifest.provider_models_by_role = {
        "baseline": {
            "provider_name": provider.provider_name,
            "provider_kind": provider.provider_kind,
            "model_name": provider.model_name,
            "model_ref": condition.baseline_model,
        }
    }
    if bundle.experiment.runtime_overrides.export_json_schemas:
        export_json_schemas(run_dir / "schemas")

    validation_ctx = build_turn_validation_context(
        label_set,
        line_ids,
        public_message_min_words=runtime.public_message_min_words,
        public_message_max_words=runtime.public_message_max_words,
        public_message_hard_cap_words=runtime.public_message_hard_cap_words,
    )
    prompt = _baseline_prompt(bundle, condition, scenario)

    provider_ctx = ProviderCallContext(
        provider_name=provider.provider_name,
        provider_kind=provider.provider_kind,
        model_name=provider.model_name,
        phase=TurnPhase.BASELINE,
        role_id=RoleId.R,
        public_turn_index=None,
        turn_id="baseline",
        timeout_seconds=runtime.provider_timeout_seconds,
        metadata={
            "line_ids": line_ids,
            "label_set_labels": label_set.labels,
            "scenario_id": scenario.scenario_id,
            "condition_id": condition.condition_id,
            "phase": "baseline",
            "is_final_public_turn": False,
        },
    )
    retry_result = strict_json_retry_loop(
        provider=provider,
        provider_context=provider_ctx,
        base_prompt=prompt,
        validation_context=validation_ctx,
        max_retries=runtime.json_max_retries,
        attempt_log_factory=_attempt_factory(provider, TurnPhase.BASELINE, RoleId.R),
    )
    if retry_result.success:
        status = TurnStatus.SUCCESS
    else:
        last = retry_result.attempts[-1] if retry_result.attempts else None
        if last and last.exception_type:
            status = TurnStatus.FAILED_RUNTIME
        elif last and last.json_valid:
            status = TurnStatus.FAILED_SCHEMA
        else:
            status = TurnStatus.FAILED_JSON

    turn = TurnResultLog(
        turn_id="baseline",
        phase=TurnPhase.BASELINE,
        role_id=RoleId.R,
        attempts=retry_result.attempts,
        status=status,
        final_output=retry_result.output,
        validation_warnings=retry_result.warnings,
    )
    validation_stats = ValidationStats()
    _update_validation_stats(validation_stats, turn)
    stop_reason: str | None = None
    if runtime.per_run_wallclock_limit_seconds is not None:
        if (time.monotonic() - run_started) > runtime.per_run_wallclock_limit_seconds:
            stop_reason = (
                "Per-run wall-clock limit exceeded "
                f"({runtime.per_run_wallclock_limit_seconds}s)"
            )
    if stop_reason is None and runtime.token_budget_limit is not None:
        total_tokens = _attempts_total_tokens(turn.attempts)
        if total_tokens > runtime.token_budget_limit:
            stop_reason = f"Per-run token budget limit exceeded ({total_tokens} > {runtime.token_budget_limit})"
    if stop_reason is not None and status == TurnStatus.SUCCESS:
        status = TurnStatus.FAILED_RUNTIME
        turn.status = status

    final_label = None
    final_severity = None
    if retry_result.output is not None:
        top1 = next((x for x in retry_result.output.assessment.ranked_findings if x.rank == 1), None)
        if top1 is not None:
            final_label = top1.label
            final_severity = top1.severity

    run_metrics = compute_run_metrics(
        run_id=run_id,
        condition_id=condition.condition_id,
        scenario_id=scenario.scenario_id,
        mode=ConditionMode.BASELINE,
        committee_snapshots=[],
        ground_truth=ground_truth,
        validation_stats=validation_stats,
        baseline_top1_label=final_label,
        baseline_top1_severity=final_severity,
    )

    write_json(transcript_dir / "turns.json", [turn.model_dump(mode="json")])
    write_json(transcript_dir / "public_history.json", [])
    write_json(transcript_dir / "committee_snapshots.json", [])
    write_json(provider_dir / "attempts.json", [a.model_dump(mode="json") for a in turn.attempts])
    write_jsonl(provider_dir / "attempts.jsonl", [a.model_dump(mode="json") for a in turn.attempts])
    metric_paths = export_run_metrics_files(run_dir, run_metrics)
    build_summary_table_json(run_dir, run_metrics)
    plot_paths = generate_basic_plots(run_dir, run_metrics) if bundle.experiment.runtime_overrides.generate_plots else {}
    expert_csv = export_expert_review_csv(
        run_dir,
        run_id=run_id,
        condition_id=condition.condition_id,
        scenario_id=scenario.scenario_id,
        final_committee_label=final_label,
        final_committee_severity=(final_severity.value if final_severity else None),
        final_agreement_exact=None,
    )
    overview_paths = export_siemens_overview_package(run_dir, run_metrics)

    manifest = finalize_manifest(
        manifest,
        status="success" if status == TurnStatus.SUCCESS and stop_reason is None else "failed",
        output_paths={
            "run_dir": str(run_dir),
            "turns_json": str(transcript_dir / "turns.json"),
            "provider_attempts_json": str(provider_dir / "attempts.json"),
            "provider_attempts_jsonl": str(provider_dir / "attempts.jsonl"),
            "expert_review_csv": expert_csv,
            **metric_paths,
            **plot_paths,
            **overview_paths,
        },
        error=(
            None
            if status == TurnStatus.SUCCESS and stop_reason is None
            else (stop_reason or "Baseline failed due to strict JSON retry exhaustion and/or provider error")
        ),
    )
    write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))

    return BaselineRunResult(run_dir=Path(run_dir), manifest=manifest, run_metrics=run_metrics)
