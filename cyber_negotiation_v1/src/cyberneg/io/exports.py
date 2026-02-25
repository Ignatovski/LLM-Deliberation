from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.schemas import AggregateMetricsReport, ExpertReviewRow, RunMetrics
from .storage import ensure_dir, write_csv, write_json


def run_metrics_to_flat_row(run_metrics: RunMetrics) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_metrics.run_id,
        "condition_id": run_metrics.condition_id,
        "scenario_id": run_metrics.scenario_id,
        "mode": run_metrics.mode.value,
    }
    row.update(run_metrics.metrics)
    row.update(
        {
            "JsonRetryCount": run_metrics.validation.json_retry_count,
            "JsonFailureCount": run_metrics.validation.json_failures,
            "SchemaValidationFailureCount": run_metrics.validation.schema_validation_failures,
            "CitationCountViolations": run_metrics.validation.citation_count_violations,
            "MessageLengthViolations": run_metrics.validation.message_length_violations,
        }
    )
    row.update({f"committee_final_{k}": v for k, v in run_metrics.committee_final.items()})
    return row


def export_run_metrics_files(run_dir: str | Path, run_metrics: RunMetrics) -> dict[str, str]:
    metrics_dir = ensure_dir(Path(run_dir) / "metrics")
    json_path = write_json(metrics_dir / "run_metrics.json", run_metrics.model_dump(mode="json"))
    csv_path = write_csv(metrics_dir / "run_metrics.csv", [run_metrics_to_flat_row(run_metrics)])
    return {"run_metrics_json": str(json_path), "run_metrics_csv": str(csv_path)}


def export_aggregate_metrics_files(base_dir: str | Path, reports: list[AggregateMetricsReport]) -> dict[str, str]:
    metrics_dir = ensure_dir(Path(base_dir) / "metrics")
    json_path = write_json(metrics_dir / "aggregate_metrics.json", [r.model_dump(mode="json") for r in reports])
    rows = [r.model_dump(mode="json") for r in reports]
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        base = {"scope": row["scope"], "key": row["key"], "count_runs": row["count_runs"]}
        for mk, mv in row["aggregates"].items():
            out = dict(base)
            out["metric"] = mk
            out["value"] = mv
            csv_rows.append(out)
    csv_path = write_csv(metrics_dir / "aggregate_metrics.csv", csv_rows)
    return {"aggregate_metrics_json": str(json_path), "aggregate_metrics_csv": str(csv_path)}


def export_expert_review_csv(
    run_dir: str | Path,
    *,
    run_id: str,
    condition_id: str,
    scenario_id: str,
    final_committee_label: str | None,
    final_committee_severity: str | None,
    final_agreement_exact: bool | None,
) -> str:
    export_dir = ensure_dir(Path(run_dir) / "exports")
    row = ExpertReviewRow(
        run_id=run_id,
        condition_id=condition_id,
        scenario_id=scenario_id,
        final_committee_label=final_committee_label,
        final_committee_severity=final_committee_severity,  # type: ignore[arg-type]
        final_agreement_exact=final_agreement_exact,
        transcript_reference=str(Path(run_dir) / "transcript" / "public_history.json"),
        agent_final_outputs_reference=str(Path(run_dir) / "transcript" / "turns.json"),
    )
    write_csv(export_dir / "expert_review.csv", [row.model_dump(mode="json")])
    return str(export_dir / "expert_review.csv")


def export_siemens_overview_package(run_dir: str | Path, run_metrics: RunMetrics) -> dict[str, str]:
    base = ensure_dir(Path(run_dir) / "siemens_overview")
    summary = {
        "run_id": run_metrics.run_id,
        "condition_id": run_metrics.condition_id,
        "scenario_id": run_metrics.scenario_id,
        "final_committee": run_metrics.committee_final,
        "headline_metrics": {
            k: run_metrics.metrics.get(k)
            for k in [
                "FinalCorrectType",
                "FinalCorrectSeverity",
                "FinalCorrectExact",
                "FinalAgreementType",
                "FinalAgreementExact",
                "WrongConsensusType",
                "WrongConsensusExact",
                "LateDriftType",
                "LateDriftExact",
            ]
        },
    }
    json_path = write_json(base / "overview_summary.json", summary)
    csv_path = write_csv(base / "overview_metrics_table.csv", [run_metrics_to_flat_row(run_metrics)])
    return {"overview_json": str(json_path), "overview_csv": str(csv_path)}


def generate_basic_plots(run_dir: str | Path, run_metrics: RunMetrics) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    plots_dir = ensure_dir(Path(run_dir) / "plots")
    out: dict[str, str] = {}

    snaps = run_metrics.per_turn_committee
    if snaps:
        turns = [s.public_turn_index for s in snaps]
        agree_type = [1 if s.full_agreement_type else 0 for s in snaps]
        agree_exact = [1 if s.full_agreement_exact else 0 for s in snaps]
        sev_vals = [
            (None if s.committee_majority_severity is None else {"Compliance": 0, "Info": 1, "Low": 2, "Medium": 3, "High": 4}[s.committee_majority_severity.value])
            for s in snaps
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(turns, agree_type, marker="o", label="AgreementType")
        ax.plot(turns, agree_exact, marker="o", label="AgreementExact")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Public turn")
        ax.set_ylabel("Agreement")
        ax.set_title("Agreement Trajectory")
        ax.legend()
        p = plots_dir / "agreement_trajectory.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out["agreement_trajectory"] = str(p)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(turns, [v if v is not None else float("nan") for v in sev_vals], marker="o")
        ax.set_xlabel("Public turn")
        ax.set_ylabel("Severity ordinal")
        ax.set_title("Committee Severity Trajectory")
        p = plots_dir / "severity_trajectory.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out["severity_trajectory"] = str(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["JsonRetryCount", "JsonFailureRate", "SchemaValidationFailureRate", "MessageLengthViolations"]
    vals = [run_metrics.metrics.get(k) or 0 for k in keys]
    ax.bar(range(len(keys)), vals)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_title("Validation / Reliability Signals")
    p = plots_dir / "validation_signals.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out["validation_signals"] = str(p)

    return out


def build_summary_table_json(run_dir: str | Path, run_metrics: RunMetrics) -> str:
    p = Path(run_dir) / "metrics" / "run_metrics_table.json"
    ensure_dir(p.parent)
    try:
        import pandas as pd

        table = pd.DataFrame([run_metrics_to_flat_row(run_metrics)])
        p.write_text(table.to_json(orient="records", indent=2), encoding="utf-8")
    except Exception:
        p.write_text(__import__("json").dumps([run_metrics_to_flat_row(run_metrics)], indent=2), encoding="utf-8")
    return str(p)
