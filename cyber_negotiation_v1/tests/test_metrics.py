from __future__ import annotations

import pytest

from cyberneg.core.enums import ConditionMode, RoleId, Severity
from cyberneg.core.metrics import compute_run_metrics
from cyberneg.core.schemas import CommitteeSnapshot, GroundTruth, ValidationStats


def _snap(
    turn: int,
    *,
    label: str | None,
    exact_label: str | None,
    exact_sev: Severity | None,
    full_type: bool,
    full_exact: bool,
    majority_sev: Severity | None,
) -> CommitteeSnapshot:
    return CommitteeSnapshot(
        public_turn_index=turn,
        by_agent_top1_label={RoleId.R: label, RoleId.C: label, RoleId.K: label},
        by_agent_top1_severity={RoleId.R: exact_sev, RoleId.C: exact_sev, RoleId.K: exact_sev},
        committee_type_label=label,
        committee_exact_label=exact_label,
        committee_exact_severity=exact_sev,
        committee_type_status="majority",
        committee_exact_status="majority" if exact_label is not None and exact_sev is not None else "no_majority",
        full_agreement_type=full_type,
        full_agreement_exact=full_exact,
        committee_majority_severity=majority_sev,
    )


def test_metrics_wrong_consensus_and_late_drift() -> None:
    gt = GroundTruth(scenario_id="s1", final_label="XSS_Reflected", final_severity=Severity.MEDIUM)
    snaps = [
        _snap(
            0,
            label="XSS_Reflected",
            exact_label="XSS_Reflected",
            exact_sev=Severity.MEDIUM,
            full_type=True,
            full_exact=True,
            majority_sev=Severity.MEDIUM,
        ),
        _snap(
            1,
            label="NoFinding",
            exact_label="NoFinding",
            exact_sev=Severity.INFO,
            full_type=True,
            full_exact=True,
            majority_sev=Severity.INFO,
        ),
    ]
    rep = compute_run_metrics(
        run_id="r1",
        condition_id="C1",
        scenario_id="s1",
        mode=ConditionMode.NEGOTIATION,
        committee_snapshots=snaps,
        ground_truth=gt,
        validation_stats=ValidationStats(),
    )
    m = rep.metrics
    assert m["WrongConsensusType"] is True
    assert m["WrongConsensusExact"] is True
    assert m["LateDriftType"] is True
    assert m["LateDriftExact"] is True
    assert m["FinalCorrectType"] is False


def test_metrics_latency_flipcount_and_severity_variance() -> None:
    gt = GroundTruth(scenario_id="s1", final_label="XSS_Reflected", final_severity=Severity.MEDIUM)
    snaps = [
        _snap(
            0,
            label="NoFinding",
            exact_label="NoFinding",
            exact_sev=Severity.LOW,
            full_type=True,
            full_exact=True,
            majority_sev=Severity.LOW,
        ),
        _snap(
            1,
            label="XSS_Reflected",
            exact_label="XSS_Reflected",
            exact_sev=Severity.MEDIUM,
            full_type=True,
            full_exact=True,
            majority_sev=Severity.MEDIUM,
        ),
        _snap(
            2,
            label="XSS_Reflected",
            exact_label="XSS_Reflected",
            exact_sev=Severity.MEDIUM,
            full_type=True,
            full_exact=True,
            majority_sev=Severity.MEDIUM,
        ),
    ]
    rep = compute_run_metrics(
        run_id="r2",
        condition_id="C1",
        scenario_id="s1",
        mode=ConditionMode.NEGOTIATION,
        committee_snapshots=snaps,
        ground_truth=gt,
        validation_stats=ValidationStats(),
    )
    m = rep.metrics
    assert m["FlipCountType"] == 1
    assert m["ConsensusLatencyType"] == 1
    assert m["ConsensusLatencyExact"] == 1
    assert m["SeverityVarianceAcrossRounds"] == pytest.approx(2 / 9, rel=1e-6)

