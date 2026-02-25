from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ConditionMode, RoleId, Severity, TurnPhase, TurnStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceLine(StrictModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EvidencePacket(StrictModel):
    scenario_id: str
    title: str
    source_family: str
    difficulty: Literal["easy", "medium", "hard"]
    label_set_id: str
    lines: list[EvidenceLine]
    author_notes: Optional[str] = None

    @model_validator(mode="after")
    def _unique_line_ids(self) -> "EvidencePacket":
        ids = [ln.id for ln in self.lines]
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence packet line IDs must be unique")
        return self

    def visible_payload(self) -> dict[str, Any]:
        data = self.model_dump()
        data.pop("author_notes", None)
        return data


class GroundTruth(StrictModel):
    scenario_id: str
    final_label: str
    final_severity: Severity
    ambiguity_notes: Optional[str] = None
    expert_notes: Optional[str] = None
    adjudication_metadata: Optional[dict[str, Any]] = None


class RankedFinding(StrictModel):
    rank: int = Field(ge=1, le=3)
    label: str
    severity: Severity
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)
    rationale: Optional[str] = None


class StructuredAssessment(StrictModel):
    ranked_findings: list[RankedFinding] = Field(min_length=3, max_length=3)
    decision_summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ranks(self) -> "StructuredAssessment":
        ranks = [item.rank for item in self.ranked_findings]
        if sorted(ranks) != [1, 2, 3]:
            raise ValueError("ranked_findings must contain exactly ranks 1, 2, and 3")
        return self


class AgentTurnOutput(StrictModel):
    private_notes: str
    private_plan: str
    public_message: str
    assessment: StructuredAssessment


class ProviderUsage(StrictModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    raw: Optional[dict[str, Any]] = None


class ValidationMessage(StrictModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ProviderAttemptLog(StrictModel):
    attempt_index: int
    provider_name: str
    provider_kind: str
    model_name: str
    phase: TurnPhase
    role_id: Optional[RoleId] = None
    public_turn_index: Optional[int] = None
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str = Field(default_factory=utc_now_iso)
    duration_ms: Optional[int] = None
    prompt_text: str
    raw_response_text: Optional[str] = None
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage: Optional[ProviderUsage] = None
    json_valid: bool = False
    schema_valid: bool = False
    validation_errors: list[ValidationMessage] = Field(default_factory=list)
    validation_warnings: list[ValidationMessage] = Field(default_factory=list)
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None


class TurnResultLog(StrictModel):
    turn_id: str
    phase: TurnPhase
    role_id: RoleId
    public_turn_index: Optional[int] = None
    is_final_public_turn: bool = False
    attempts: list[ProviderAttemptLog]
    status: TurnStatus
    final_output: Optional[AgentTurnOutput] = None
    validation_warnings: list[ValidationMessage] = Field(default_factory=list)


class PublicTurnRecord(StrictModel):
    public_turn_index: int
    role_id: RoleId
    public_message: str
    timestamp: str = Field(default_factory=utc_now_iso)


class CommitteeSnapshot(StrictModel):
    public_turn_index: int
    by_agent_top1_label: dict[RoleId, Optional[str]]
    by_agent_top1_severity: dict[RoleId, Optional[Severity]]
    committee_type_label: Optional[str] = None
    committee_exact_label: Optional[str] = None
    committee_exact_severity: Optional[Severity] = None
    committee_type_status: Literal["majority", "no_majority"]
    committee_exact_status: Literal["majority", "no_majority"]
    full_agreement_type: bool
    full_agreement_exact: bool
    committee_majority_severity: Optional[Severity] = None


class SchedulerPlan(StrictModel):
    order_seed: int
    public_order: list[RoleId]
    final_turn_flags: list[bool]
    role_counts: dict[RoleId, int]


class RunManifest(StrictModel):
    run_id: str
    experiment_id: str
    condition_id: str
    scenario_id: str
    mode: ConditionMode
    seed: int
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: Optional[str] = None
    status: Literal["success", "failed"] = "success"
    repo_adaptation_note: str = "Implemented as isolated subproject under cyber_negotiation_v1/"
    config_refs: dict[str, Any] = Field(default_factory=dict)
    provider_models_by_role: dict[str, dict[str, str]] = Field(default_factory=dict)
    scheduler: Optional[SchedulerPlan] = None
    runtime_limits: dict[str, Any] = Field(default_factory=dict)
    output_paths: dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None


class ValidationStats(StrictModel):
    total_turns: int = 0
    total_attempts: int = 0
    successful_turns: int = 0
    failed_turns: int = 0
    json_retry_count: int = 0
    json_failures: int = 0
    schema_validation_failures: int = 0
    citation_count_violations: int = 0
    message_length_violations: int = 0
    citation_total_rank1: int = 0
    citation_valid_rank1: int = 0


class RunMetrics(StrictModel):
    run_id: str
    condition_id: str
    scenario_id: str
    mode: ConditionMode
    metrics: dict[str, Any]
    validation: ValidationStats
    committee_final: dict[str, Any] = Field(default_factory=dict)
    per_turn_committee: list[CommitteeSnapshot] = Field(default_factory=list)


class AggregateMetricsReport(StrictModel):
    scope: Literal["per_scenario", "per_condition", "overall"]
    key: str
    count_runs: int
    aggregates: dict[str, Any]


class ExpertReviewRow(StrictModel):
    run_id: str
    condition_id: str
    scenario_id: str
    final_committee_label: Optional[str] = None
    final_committee_severity: Optional[Severity] = None
    final_agreement_exact: Optional[bool] = None
    transcript_reference: str
    agent_final_outputs_reference: str
    expert_label: Optional[str] = None
    expert_severity: Optional[str] = None
    expert_score_argument_quality: Optional[float] = None
    expert_score_evidence_relevance: Optional[float] = None
    expert_score_defensibility_for_report: Optional[float] = None
    expert_would_accept_in_QA: Optional[str] = None
    expert_comments: Optional[str] = None


class ProviderCatalogEntry(StrictModel):
    kind: str
    enabled: bool = True
    env_mapping: dict[str, list[str]] = Field(default_factory=dict)


class ProviderCatalog(StrictModel):
    providers: dict[str, ProviderCatalogEntry]


class ModelCatalogEntry(StrictModel):
    provider: str
    model_name: str
    timeout_seconds: Optional[int] = None


class ModelCatalog(StrictModel):
    models: dict[str, ModelCatalogEntry]


class RoleInstructionConfig(StrictModel):
    role_id: RoleId
    name: str
    short_name: Optional[str] = None
    instruction_text: str


class PromptTemplateConfig(StrictModel):
    name: str
    template: str


class LabelSetConfig(StrictModel):
    label_set_id: str
    labels: list[str]
    aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_labels(self) -> "LabelSetConfig":
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("Label set labels must be unique")
        return self


class PriorsConfig(StrictModel):
    apply_in_round0_only: bool = True
    text: str = ""


class PromptOptionsConfig(StrictModel):
    re_inject_role_instruction_in_roundn: bool = False
    reminder_text: str = ""


class RuntimeConfig(StrictModel):
    public_messages: Optional[int] = None
    json_max_retries: int = 3
    provider_timeout_seconds: Optional[int] = None
    per_run_wallclock_limit_seconds: Optional[int] = None
    token_budget_limit: Optional[int] = None
    final_turn_announcement_window: Optional[int] = 1
    parallel_runs: Optional[bool] = False
    public_message_min_words: int = 80
    public_message_max_words: int = 150
    public_message_hard_cap_words: int = 220


class ConditionConfig(StrictModel):
    condition_id: str
    enabled: bool = True
    mode: ConditionMode
    priors: PriorsConfig = Field(default_factory=PriorsConfig)
    models_by_role: Optional[dict[RoleId, str]] = None
    baseline_model: Optional[str] = None
    prompt_options: PromptOptionsConfig = Field(default_factory=PromptOptionsConfig)
    runtime: RuntimeConfig

    @model_validator(mode="after")
    def _validate_mode_specific_fields(self) -> "ConditionConfig":
        if self.mode == ConditionMode.NEGOTIATION:
            if not self.models_by_role:
                raise ValueError("Negotiation condition requires models_by_role")
            missing = {RoleId.R, RoleId.C, RoleId.K} - set(self.models_by_role.keys())
            if missing:
                raise ValueError(f"Negotiation condition missing role model mappings: {sorted(m.value for m in missing)}")
            if self.runtime.public_messages is None:
                raise ValueError("Negotiation condition requires runtime.public_messages")
            if self.runtime.public_messages <= 0:
                raise ValueError("runtime.public_messages must be > 0")
            if self.runtime.public_messages % 3 != 0:
                raise ValueError("runtime.public_messages must be divisible by 3 to ensure equal turns")
        elif self.mode == ConditionMode.BASELINE:
            if not self.baseline_model:
                raise ValueError("Baseline condition requires baseline_model")
        return self


class ConditionSetConfig(StrictModel):
    condition_set_id: str
    description: Optional[str] = None
    conditions: list[ConditionConfig]

    @model_validator(mode="after")
    def _unique_condition_ids(self) -> "ConditionSetConfig":
        ids = [c.condition_id for c in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("Condition IDs must be unique")
        return self


class MockBehaviorConfig(StrictModel):
    invalid_json_attempts_per_turn: int = 0
    deterministic_seed_offset: int = 0


class MockConfig(StrictModel):
    enabled: bool = True
    behavior: MockBehaviorConfig = Field(default_factory=MockBehaviorConfig)


class RuntimeOverridesConfig(StrictModel):
    export_json_schemas: bool = True
    generate_plots: bool = True


class ExperimentConfig(StrictModel):
    experiment_id: str
    condition_set_path: str
    provider_catalog_path: str
    model_catalog_path: str
    label_set_paths: list[str]
    scenario_paths: list[str]
    ground_truth_paths: list[str] = Field(default_factory=list)
    role_paths: dict[RoleId, str]
    prompt_paths: dict[str, str]
    output_root: str = "outputs"
    seed: int = 0
    mock: MockConfig = Field(default_factory=MockConfig)
    runtime_overrides: RuntimeOverridesConfig = Field(default_factory=RuntimeOverridesConfig)

