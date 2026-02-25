from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.aggregator import build_committee_snapshot
from ..core.enums import ConditionMode, RoleId, TurnPhase, TurnStatus
from ..core.history import latest_assessment_by_role, latest_private_notes, latest_private_plan, visible_public_history
from ..core.metrics import compute_run_metrics
from ..core.scheduler import generate_public_schedule, round0_roles
from ..core.schemas import (
    ConditionConfig,
    EvidencePacket,
    GroundTruth,
    ProviderAttemptLog,
    RunManifest,
    TurnResultLog,
    ValidationMessage,
    ValidationStats,
)
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
from ..prompting.json_contracts import export_json_schemas
from ..prompting.render_round0 import render_round0_prompt
from ..prompting.render_roundn import render_roundn_prompt
from ..providers.base import ProviderCallContext


@dataclass
class NegotiationRunResult:
    run_dir: Path
    manifest: RunManifest
    metrics_json_path: Path
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


def _all_attempts_total_tokens(turns: list[TurnResultLog]) -> int:
    return sum(_attempts_total_tokens(turn.attempts) for turn in turns)


def _stop_guard_reason(*, runtime, run_started: float, turns: list[TurnResultLog]) -> str | None:
    if runtime.per_run_wallclock_limit_seconds is not None:
        if (time.monotonic() - run_started) > runtime.per_run_wallclock_limit_seconds:
            return f"Per-run wall-clock limit exceeded ({runtime.per_run_wallclock_limit_seconds}s)"
    if runtime.token_budget_limit is not None:
        total_tokens = _all_attempts_total_tokens(turns)
        if total_tokens > runtime.token_budget_limit:
            return f"Per-run token budget limit exceeded ({total_tokens} > {runtime.token_budget_limit})"
    return None


def _turn_attempt_factory(
    *,
    provider_name: str,
    provider_kind: str,
    model_name: str,
    phase: TurnPhase,
    role_id: Optional[RoleId],
    public_turn_index: Optional[int],
) -> Any:
    def factory(attempt_idx: int, prompt_text: str) -> ProviderAttemptLog:
        return ProviderAttemptLog(
            attempt_index=attempt_idx,
            provider_name=provider_name,
            provider_kind=provider_kind,
            model_name=model_name,
            phase=phase,
            role_id=role_id,
            public_turn_index=public_turn_index,
            prompt_text=prompt_text,
        )

    return factory


def _summarize_failed_status(turn_attempts: list[ProviderAttemptLog]) -> TurnStatus:
    if not turn_attempts:
        return TurnStatus.FAILED_RUNTIME
    last = turn_attempts[-1]
    if last.exception_type:
        return TurnStatus.FAILED_RUNTIME
    if last.json_valid and not last.schema_valid:
        return TurnStatus.FAILED_SCHEMA
    return TurnStatus.FAILED_JSON


def _update_validation_stats(stats: ValidationStats, turn: TurnResultLog) -> None:
    stats.total_turns += 1
    stats.total_attempts += len(turn.attempts)
    if turn.status == TurnStatus.SUCCESS:
        stats.successful_turns += 1
    else:
        stats.failed_turns += 1
        if turn.status == TurnStatus.FAILED_JSON:
            stats.json_failures += 1
    stats.json_retry_count += max(0, len(turn.attempts) - 1)

    for attempt in turn.attempts:
        if attempt.json_valid and not attempt.schema_valid:
            stats.schema_validation_failures += 1
        for err in attempt.validation_errors:
            if err.code == "rank1_citation_count":
                stats.citation_count_violations += 1

    for warn in turn.validation_warnings:
        if warn.code == "public_message_target_range":
            stats.message_length_violations += 1

    if turn.final_output is not None:
        rank1 = next((x for x in turn.final_output.assessment.ranked_findings if x.rank == 1), None)
        if rank1 is not None:
            stats.citation_total_rank1 += len(rank1.citations)
            # If the turn succeeded, citations are valid by construction.
            stats.citation_valid_rank1 += len(rank1.citations)


def _turn_to_public_json(turns: list[TurnResultLog]) -> list[dict[str, Any]]:
    return [h.model_dump(mode="json") for h in visible_public_history(turns)]


def run_negotiation_once(
    *,
    bundle: LoadedExperimentBundle,
    condition: ConditionConfig,
    scenario: EvidencePacket,
    ground_truth: GroundTruth | None,
    session_dir: Path,
    run_seed: int,
) -> NegotiationRunResult:
    if condition.mode != ConditionMode.NEGOTIATION:
        raise ValueError("run_negotiation_once requires a negotiation condition")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_dir = ensure_dir(session_dir / condition.condition_id / scenario.scenario_id / run_id)
    transcript_dir = ensure_dir(run_dir / "transcript")
    provider_dir = ensure_dir(run_dir / "provider")

    label_set = bundle.label_sets[scenario.label_set_id]
    line_ids = [ln.id for ln in scenario.lines]
    runtime = condition.runtime
    run_started = time.monotonic()

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

    if bundle.experiment.runtime_overrides.export_json_schemas:
        schema_paths = export_json_schemas(run_dir / "schemas")
        manifest.output_paths["schemas"] = str(run_dir / "schemas")
        manifest.config_refs["schema_paths"] = schema_paths

    providers_by_role: dict[RoleId, Any] = {}
    for role in (RoleId.R, RoleId.C, RoleId.K):
        model_ref = condition.models_by_role[role]  # validated by schema
        provider = build_provider_from_model_ref(
            bundle=bundle,
            model_ref=model_ref,
            timeout_override=runtime.provider_timeout_seconds,
            mock_behavior_override=bundle.experiment.mock.behavior.model_dump(mode="json"),
        )
        providers_by_role[role] = provider
        manifest.provider_models_by_role[role.value] = {
            "provider_name": provider.provider_name,
            "provider_kind": provider.provider_kind,
            "model_name": provider.model_name,
            "model_ref": model_ref,
        }

    public_schedule = generate_public_schedule(runtime.public_messages or 0, seed=run_seed + 101)
    # Respect config final-turn announcement window, overriding default scheduler flags if provided.
    final_window = runtime.final_turn_announcement_window or 1
    public_schedule.final_turn_flags = [
        idx >= (len(public_schedule.public_order) - max(1, min(final_window, len(public_schedule.public_order))))
        for idx in range(len(public_schedule.public_order))
    ]
    manifest.scheduler = public_schedule

    turns: list[TurnResultLog] = []
    committee_snaps: list[Any] = []
    validation_stats = ValidationStats()
    stop_reason: str | None = None
    validation_ctx = build_turn_validation_context(
        label_set,
        line_ids,
        public_message_min_words=runtime.public_message_min_words,
        public_message_max_words=runtime.public_message_max_words,
        public_message_hard_cap_words=runtime.public_message_hard_cap_words,
    )

    # Phase A: Round 0 independent assessments (all agents see full evidence from start)
    for role in round0_roles():
        stop_reason = _stop_guard_reason(runtime=runtime, run_started=run_started, turns=turns)
        if stop_reason is not None:
            break
        provider = providers_by_role[role]
        role_cfg = bundle.roles[role.value]
        prompt = render_round0_prompt(
            global_prompt=bundle.prompts["round0_global"],
            role_instruction=role_cfg,
            prior_text=(condition.priors.text if condition.priors.apply_in_round0_only else ""),
            evidence_packet=scenario,
        )
        turn_id = f"round0_{role.value}"
        provider_ctx = ProviderCallContext(
            provider_name=provider.provider_name,
            provider_kind=provider.provider_kind,
            model_name=provider.model_name,
            phase=TurnPhase.ROUND0,
            role_id=role,
            public_turn_index=None,
            turn_id=turn_id,
            timeout_seconds=runtime.provider_timeout_seconds,
            metadata={
                "line_ids": line_ids,
                "label_set_labels": label_set.labels,
                "scenario_id": scenario.scenario_id,
                "condition_id": condition.condition_id,
                "phase": "round0",
                "is_final_public_turn": False,
            },
        )
        retry_result = strict_json_retry_loop(
            provider=provider,
            provider_context=provider_ctx,
            base_prompt=prompt,
            validation_context=validation_ctx,
            max_retries=runtime.json_max_retries,
            attempt_log_factory=_turn_attempt_factory(
                provider_name=provider.provider_name,
                provider_kind=provider.provider_kind,
                model_name=provider.model_name,
                phase=TurnPhase.ROUND0,
                role_id=role,
                public_turn_index=None,
            ),
        )
        turn = TurnResultLog(
            turn_id=turn_id,
            phase=TurnPhase.ROUND0,
            role_id=role,
            attempts=retry_result.attempts,
            status=TurnStatus.SUCCESS if retry_result.success else _summarize_failed_status(retry_result.attempts),
            final_output=retry_result.output,
            validation_warnings=retry_result.warnings,
        )
        turns.append(turn)
        _update_validation_stats(validation_stats, turn)
        if turn.status != TurnStatus.SUCCESS:
            break
        stop_reason = _stop_guard_reason(runtime=runtime, run_started=run_started, turns=turns)
        if stop_reason is not None:
            break

    # Phase B: public negotiation turns
    if stop_reason is None and all(t.status == TurnStatus.SUCCESS for t in turns):
        for public_idx, role in enumerate(public_schedule.public_order):
            stop_reason = _stop_guard_reason(runtime=runtime, run_started=run_started, turns=turns)
            if stop_reason is not None:
                break
            provider = providers_by_role[role]
            role_cfg = bundle.roles[role.value]
            prompt = render_roundn_prompt(
                roundn_prompt=bundle.prompts["roundn"],
                public_history=visible_public_history(turns),
                own_previous_private_notes=latest_private_notes(turns, role),
                own_previous_private_plan=latest_private_plan(turns, role),
                reminder_text=condition.prompt_options.reminder_text,
                is_final_public_turn=public_schedule.final_turn_flags[public_idx],
                role_instruction=role_cfg,
                reinject_role_instruction=condition.prompt_options.re_inject_role_instruction_in_roundn,
            )
            turn_id = f"public_{public_idx:03d}_{role.value}"
            provider_ctx = ProviderCallContext(
                provider_name=provider.provider_name,
                provider_kind=provider.provider_kind,
                model_name=provider.model_name,
                phase=TurnPhase.PUBLIC,
                role_id=role,
                public_turn_index=public_idx,
                turn_id=turn_id,
                timeout_seconds=runtime.provider_timeout_seconds,
                metadata={
                    "line_ids": line_ids,
                    "label_set_labels": label_set.labels,
                    "scenario_id": scenario.scenario_id,
                    "condition_id": condition.condition_id,
                    "phase": "public",
                    "is_final_public_turn": public_schedule.final_turn_flags[public_idx],
                },
            )
            retry_result = strict_json_retry_loop(
                provider=provider,
                provider_context=provider_ctx,
                base_prompt=prompt,
                validation_context=validation_ctx,
                max_retries=runtime.json_max_retries,
                attempt_log_factory=_turn_attempt_factory(
                    provider_name=provider.provider_name,
                    provider_kind=provider.provider_kind,
                    model_name=provider.model_name,
                    phase=TurnPhase.PUBLIC,
                    role_id=role,
                    public_turn_index=public_idx,
                ),
            )
            turn = TurnResultLog(
                turn_id=turn_id,
                phase=TurnPhase.PUBLIC,
                role_id=role,
                public_turn_index=public_idx,
                is_final_public_turn=public_schedule.final_turn_flags[public_idx],
                attempts=retry_result.attempts,
                status=TurnStatus.SUCCESS if retry_result.success else _summarize_failed_status(retry_result.attempts),
                final_output=retry_result.output,
                validation_warnings=retry_result.warnings,
            )
            turns.append(turn)
            _update_validation_stats(validation_stats, turn)
            if turn.status != TurnStatus.SUCCESS:
                break

            snap = build_committee_snapshot(public_idx, latest_assessment_by_role(turns))
            committee_snaps.append(snap)
            stop_reason = _stop_guard_reason(runtime=runtime, run_started=run_started, turns=turns)
            if stop_reason is not None:
                break

    success = (
        stop_reason is None
        and len(turns) >= len(round0_roles())
        and
        all(t.status == TurnStatus.SUCCESS for t in turns)
        and len([t for t in turns if t.phase == TurnPhase.PUBLIC]) == len(public_schedule.public_order)
    )

    run_metrics = compute_run_metrics(
        run_id=run_id,
        condition_id=condition.condition_id,
        scenario_id=scenario.scenario_id,
        mode=ConditionMode.NEGOTIATION,
        committee_snapshots=committee_snaps,
        ground_truth=ground_truth,
        validation_stats=validation_stats,
    )

    write_json(transcript_dir / "turns.json", [t.model_dump(mode="json") for t in turns])
    write_json(transcript_dir / "public_history.json", _turn_to_public_json(turns))
    write_json(transcript_dir / "committee_snapshots.json", [s.model_dump(mode="json") for s in committee_snaps])
    write_json(provider_dir / "attempts.json", [a.model_dump(mode="json") for t in turns for a in t.attempts])
    write_jsonl(provider_dir / "attempts.jsonl", [a.model_dump(mode="json") for t in turns for a in t.attempts])

    metric_paths = export_run_metrics_files(run_dir, run_metrics)
    build_summary_table_json(run_dir, run_metrics)
    plot_paths = generate_basic_plots(run_dir, run_metrics) if bundle.experiment.runtime_overrides.generate_plots else {}
    expert_csv = export_expert_review_csv(
        run_dir,
        run_id=run_id,
        condition_id=condition.condition_id,
        scenario_id=scenario.scenario_id,
        final_committee_label=run_metrics.committee_final.get("committee_exact_label")
        or run_metrics.committee_final.get("committee_type_label"),
        final_committee_severity=run_metrics.committee_final.get("committee_exact_severity")
        or run_metrics.committee_final.get("committee_majority_severity"),
        final_agreement_exact=run_metrics.committee_final.get("full_agreement_exact"),
    )
    overview_paths = export_siemens_overview_package(run_dir, run_metrics)

    manifest = finalize_manifest(
        manifest,
        status="success" if success else "failed",
        output_paths={
            "run_dir": str(run_dir),
            "turns_json": str(transcript_dir / "turns.json"),
            "public_history_json": str(transcript_dir / "public_history.json"),
            "committee_snapshots_json": str(transcript_dir / "committee_snapshots.json"),
            "provider_attempts_json": str(provider_dir / "attempts.json"),
            "provider_attempts_jsonl": str(provider_dir / "attempts.jsonl"),
            "expert_review_csv": expert_csv,
            **metric_paths,
            **plot_paths,
            **overview_paths,
        },
        error=(None if success else (stop_reason or "Run failed due to strict JSON retry exhaustion and/or provider error")),
    )
    write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))

    return NegotiationRunResult(
        run_dir=Path(run_dir),
        manifest=manifest,
        metrics_json_path=Path(metric_paths["run_metrics_json"]),
        run_metrics=run_metrics,
    )
