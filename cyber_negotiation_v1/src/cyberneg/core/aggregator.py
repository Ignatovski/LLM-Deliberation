from __future__ import annotations

from collections import Counter
from typing import Optional

from .enums import ROLE_ORDER, RoleId, Severity
from .schemas import AgentTurnOutput, CommitteeSnapshot


def top1_label(output: AgentTurnOutput | None) -> Optional[str]:
    if output is None:
        return None
    for item in output.assessment.ranked_findings:
        if item.rank == 1:
            return item.label
    return None


def top1_severity(output: AgentTurnOutput | None) -> Optional[Severity]:
    if output is None:
        return None
    for item in output.assessment.ranked_findings:
        if item.rank == 1:
            return item.severity
    return None


def _majority_str(values: list[str | None]) -> tuple[Optional[str], str]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, "no_majority"
    counts = Counter(clean)
    label, count = counts.most_common(1)[0]
    return (label, "majority") if count >= 2 else (None, "no_majority")


def _majority_severity(values: list[Severity | None]) -> tuple[Optional[Severity], str]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, "no_majority"
    counts = Counter(clean)
    sev, count = counts.most_common(1)[0]
    return (sev, "majority") if count >= 2 else (None, "no_majority")


def _majority_exact(
    pairs: list[tuple[str, Severity] | None],
) -> tuple[Optional[tuple[str, Severity]], str]:
    clean = [v for v in pairs if v is not None]
    if not clean:
        return None, "no_majority"
    counts = Counter(clean)
    pair, count = counts.most_common(1)[0]
    return (pair, "majority") if count >= 2 else (None, "no_majority")


def build_committee_snapshot(
    public_turn_index: int,
    latest_outputs_by_role: dict[RoleId, AgentTurnOutput],
) -> CommitteeSnapshot:
    label_map: dict[RoleId, Optional[str]] = {role: top1_label(latest_outputs_by_role.get(role)) for role in ROLE_ORDER}
    sev_map: dict[RoleId, Optional[Severity]] = {
        role: top1_severity(latest_outputs_by_role.get(role)) for role in ROLE_ORDER
    }
    labels = [label_map[r] for r in ROLE_ORDER]
    sevs = [sev_map[r] for r in ROLE_ORDER]
    exact_pairs = [
        (label_map[r], sev_map[r]) if label_map[r] is not None and sev_map[r] is not None else None for r in ROLE_ORDER
    ]
    exact_pairs_clean = [p for p in exact_pairs if p is not None]

    committee_label, committee_type_status = _majority_str(labels)
    committee_sev, _ = _majority_severity(sevs)
    committee_exact_pair, committee_exact_status = _majority_exact(exact_pairs_clean)

    non_null_labels = [v for v in labels if v is not None]
    full_agreement_type = len(non_null_labels) == 3 and len(set(non_null_labels)) == 1
    non_null_exact = [v for v in exact_pairs if v is not None]
    full_agreement_exact = len(non_null_exact) == 3 and len(set(non_null_exact)) == 1

    exact_label = committee_exact_pair[0] if committee_exact_pair else None
    exact_sev = committee_exact_pair[1] if committee_exact_pair else None

    return CommitteeSnapshot(
        public_turn_index=public_turn_index,
        by_agent_top1_label=label_map,
        by_agent_top1_severity=sev_map,
        committee_type_label=committee_label,
        committee_exact_label=exact_label,
        committee_exact_severity=exact_sev,
        committee_type_status=committee_type_status,  # type: ignore[arg-type]
        committee_exact_status=committee_exact_status,  # type: ignore[arg-type]
        full_agreement_type=full_agreement_type,
        full_agreement_exact=full_agreement_exact,
        committee_majority_severity=committee_sev,
    )

