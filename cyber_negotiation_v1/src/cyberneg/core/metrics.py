from __future__ import annotations

import math
from collections import defaultdict
from statistics import pvariance
from typing import Any, Iterable, Optional

from .enums import ConditionMode, SEVERITY_ORDER, Severity
from .schemas import AggregateMetricsReport, CommitteeSnapshot, GroundTruth, RunMetrics, ValidationStats


def _severity_cmp(a: Severity, b: Severity) -> int:
    return SEVERITY_ORDER[a] - SEVERITY_ORDER[b]


def _final_snapshot(snaps: list[CommitteeSnapshot]) -> CommitteeSnapshot | None:
    return snaps[-1] if snaps else None


def _flip_count_type(snaps: list[CommitteeSnapshot]) -> int:
    traj = [snap.committee_type_label for snap in snaps]
    flips = 0
    for prev, cur in zip(traj, traj[1:]):
        if prev != cur:
            flips += 1
    return flips


def _consensus_latency(
    snaps: list[CommitteeSnapshot],
    exact: bool,
) -> Optional[int]:
    if not snaps:
        return None
    for idx, snap in enumerate(snaps):
        if exact:
            if not snap.full_agreement_exact:
                continue
            anchor = (snap.committee_exact_label, snap.committee_exact_severity)
            stable = all(
                s.full_agreement_exact and (s.committee_exact_label, s.committee_exact_severity) == anchor
                for s in snaps[idx:]
            )
            if stable:
                return snap.public_turn_index
        else:
            if not snap.full_agreement_type:
                continue
            anchor = snap.committee_type_label
            stable = all(s.full_agreement_type and s.committee_type_label == anchor for s in snaps[idx:])
            if stable:
                return snap.public_turn_index
    return None


def _late_drift_type(snaps: list[CommitteeSnapshot], gt: GroundTruth) -> bool:
    if not snaps:
        return False
    final = snaps[-1]
    final_ok = final.full_agreement_type and final.committee_type_label == gt.final_label
    if final_ok:
        return False
    for snap in snaps[:-1]:
        if snap.full_agreement_type and snap.committee_type_label == gt.final_label:
            return True
    return False


def _late_drift_exact(snaps: list[CommitteeSnapshot], gt: GroundTruth) -> bool:
    if not snaps:
        return False
    final = snaps[-1]
    final_ok = (
        final.full_agreement_exact
        and final.committee_exact_label == gt.final_label
        and final.committee_exact_severity == gt.final_severity
    )
    if final_ok:
        return False
    for snap in snaps[:-1]:
        if (
            snap.full_agreement_exact
            and snap.committee_exact_label == gt.final_label
            and snap.committee_exact_severity == gt.final_severity
        ):
            return True
    return False


def _severity_variance(snaps: list[CommitteeSnapshot]) -> Optional[float]:
    vals: list[int] = []
    for snap in snaps:
        sev = snap.committee_majority_severity
        if sev is None:
            continue
        vals.append(SEVERITY_ORDER[sev])
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return float(pvariance(vals))


def compute_run_metrics(
    *,
    run_id: str,
    condition_id: str,
    scenario_id: str,
    mode: ConditionMode,
    committee_snapshots: list[CommitteeSnapshot],
    ground_truth: GroundTruth | None,
    validation_stats: ValidationStats,
    baseline_top1_label: str | None = None,
    baseline_top1_severity: Severity | None = None,
) -> RunMetrics:
    final = _final_snapshot(committee_snapshots)

    if mode == ConditionMode.BASELINE:
        final_agreement_type = None
        final_agreement_exact = None
        final_type = baseline_top1_label
        final_exact_label = baseline_top1_label
        final_exact_sev = baseline_top1_severity
        final_majority_sev = baseline_top1_severity
    else:
        final_agreement_type = bool(final and final.full_agreement_type)
        final_agreement_exact = bool(final and final.full_agreement_exact)
        final_type = final.committee_type_label if final else None
        final_exact_label = final.committee_exact_label if final else None
        final_exact_sev = final.committee_exact_severity if final else None
        final_majority_sev = final.committee_majority_severity if final else None

    final_correct_type = False
    final_correct_severity = False
    final_correct_exact = False
    wrong_consensus_type = False
    wrong_consensus_exact = False
    over_severity = False
    under_severity = False
    late_drift_type = False
    late_drift_exact = False

    if ground_truth is not None and final is not None:
        final_correct_type = final_type == ground_truth.final_label
        if final_majority_sev is not None:
            final_correct_severity = final_majority_sev == ground_truth.final_severity
            cmp = _severity_cmp(final_majority_sev, ground_truth.final_severity)
            over_severity = cmp > 0
            under_severity = cmp < 0
        final_correct_exact = (
            final_exact_label == ground_truth.final_label and final_exact_sev == ground_truth.final_severity
        )
        if mode == ConditionMode.NEGOTIATION:
            wrong_consensus_type = bool(final_agreement_type and not final_correct_type)
            wrong_consensus_exact = bool(final_agreement_exact and not final_correct_exact)
            late_drift_type = _late_drift_type(committee_snapshots, ground_truth)
            late_drift_exact = _late_drift_exact(committee_snapshots, ground_truth)
    elif ground_truth is not None and mode == ConditionMode.BASELINE:
        final_correct_type = final_type == ground_truth.final_label
        if final_majority_sev is not None:
            final_correct_severity = final_majority_sev == ground_truth.final_severity
            cmp = _severity_cmp(final_majority_sev, ground_truth.final_severity)
            over_severity = cmp > 0
            under_severity = cmp < 0
        final_correct_exact = (
            final_exact_label == ground_truth.final_label and final_exact_sev == ground_truth.final_severity
        )

    metrics: dict[str, Any] = {
        "FinalCorrectType": final_correct_type,
        "FinalCorrectSeverity": final_correct_severity,
        "FinalCorrectExact": final_correct_exact,
        "OverSeverityRate": 1.0 if over_severity else 0.0,
        "UnderSeverityRate": 1.0 if under_severity else 0.0,
        "FinalAgreementType": final_agreement_type,
        "FinalAgreementExact": final_agreement_exact,
        "WrongConsensusType": (wrong_consensus_type if mode == ConditionMode.NEGOTIATION else None),
        "WrongConsensusExact": (wrong_consensus_exact if mode == ConditionMode.NEGOTIATION else None),
        "NoConsensus": (None if mode == ConditionMode.BASELINE else (not bool(final_agreement_exact))),
        "LateDriftType": (late_drift_type if mode == ConditionMode.NEGOTIATION else None),
        "LateDriftExact": (late_drift_exact if mode == ConditionMode.NEGOTIATION else None),
        "FlipCountType": (_flip_count_type(committee_snapshots) if mode == ConditionMode.NEGOTIATION else None),
        "ConsensusLatencyType": (
            _consensus_latency(committee_snapshots, exact=False) if mode == ConditionMode.NEGOTIATION else None
        ),
        "ConsensusLatencyExact": (
            _consensus_latency(committee_snapshots, exact=True) if mode == ConditionMode.NEGOTIATION else None
        ),
        "SeverityVarianceAcrossRounds": (
            _severity_variance(committee_snapshots)
            if mode == ConditionMode.NEGOTIATION
            else (0.0 if final_majority_sev is not None else None)
        ),
        "ExactSeverityDisagreementRateAtFinal": (
            (0.0 if bool(final_agreement_exact) else (1.0 if final is not None else None))
            if mode == ConditionMode.NEGOTIATION
            else None
        ),
        "JsonRetryCount": validation_stats.json_retry_count,
        "JsonFailureRate": (
            validation_stats.json_failures / validation_stats.total_turns if validation_stats.total_turns else 0.0
        ),
        "SchemaValidationFailureRate": (
            validation_stats.schema_validation_failures / validation_stats.total_attempts
            if validation_stats.total_attempts
            else 0.0
        ),
        "CitationValidityRate": (
            validation_stats.citation_valid_rank1 / validation_stats.citation_total_rank1
            if validation_stats.citation_total_rank1
            else None
        ),
        "CitationCountViolations": validation_stats.citation_count_violations,
        "MessageLengthViolations": validation_stats.message_length_violations,
    }
    committee_final = {
        "committee_type_label": final_type,
        "committee_exact_label": final_exact_label,
        "committee_exact_severity": final_exact_sev.value if isinstance(final_exact_sev, Severity) else None,
        "committee_majority_severity": (final_majority_sev.value if isinstance(final_majority_sev, Severity) else None),
        "full_agreement_type": final_agreement_type,
        "full_agreement_exact": final_agreement_exact,
    }
    return RunMetrics(
        run_id=run_id,
        condition_id=condition_id,
        scenario_id=scenario_id,
        mode=mode,
        metrics=metrics,
        validation=validation_stats,
        committee_final=committee_final,
        per_turn_committee=committee_snapshots,
    )


def _avg(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_run_metrics(reports: Iterable[RunMetrics]) -> list[AggregateMetricsReport]:
    reports_list = list(reports)
    out: list[AggregateMetricsReport] = []

    def build(scope: str, key_fn) -> None:
        groups: dict[str, list[RunMetrics]] = defaultdict(list)
        for rep in reports_list:
            groups[str(key_fn(rep))].append(rep)
        for key, items in groups.items():
            metric_names = sorted({m for rep in items for m in rep.metrics.keys()})
            agg: dict[str, Any] = {}
            for name in metric_names:
                vals = [rep.metrics.get(name) for rep in items]
                numeric = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                boolish = [int(v) for v in vals if isinstance(v, bool)]
                non_null = [v for v in vals if v is not None]
                if boolish and len(boolish) == len([v for v in vals if isinstance(v, bool)]):
                    agg[name] = _avg([float(v) for v in boolish])
                elif numeric:
                    agg[name] = _avg(numeric)
                else:
                    agg[name] = None if not non_null else non_null[-1]
            out.append(
                AggregateMetricsReport(
                    scope=scope,  # type: ignore[arg-type]
                    key=key,
                    count_runs=len(items),
                    aggregates=agg,
                )
            )

    build("per_scenario", lambda rep: rep.scenario_id)
    build("per_condition", lambda rep: rep.condition_id)
    build("overall", lambda rep: "overall")
    return out
