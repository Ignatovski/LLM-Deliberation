from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.metrics import aggregate_run_metrics
from ..io.exports import export_aggregate_metrics_files
from ..io.loaders import LoadedExperimentBundle, load_env_file_if_present, load_experiment_bundle
from ..io.storage import ensure_dir, utc_ts_compact, write_csv, write_json, write_latest_pointer
from .run_condition import ConditionRunResult, run_condition


@dataclass
class ExperimentRunResult:
    session_dir: Path
    output_root: Path
    condition_results: list[ConditionRunResult]
    aggregate_paths: dict[str, str]
    aggregate_reports: list[Any]
    combined_expert_review_csv: Optional[str]
    session_manifest_path: Path


def resolve_output_root(bundle: LoadedExperimentBundle, output_root_override: str | Path | None = None) -> Path:
    raw = str(output_root_override) if output_root_override is not None else bundle.experiment.output_root
    p = Path(raw)
    return p if p.is_absolute() else (bundle.project_root / p)


def create_session_dir(
    *,
    bundle: LoadedExperimentBundle,
    output_root_override: str | Path | None = None,
    session_name: str | None = None,
) -> tuple[Path, Path]:
    output_root = ensure_dir(resolve_output_root(bundle, output_root_override))
    session_leaf = session_name or utc_ts_compact()
    session_dir = ensure_dir(output_root / bundle.experiment.experiment_id / session_leaf)
    return output_root, session_dir


def _concat_expert_csvs(condition_results: list[ConditionRunResult], out_csv_path: Path) -> Optional[str]:
    rows: list[dict[str, Any]] = []
    for cr in condition_results:
        if not cr.combined_expert_review_csv:
            continue
        p = Path(cr.combined_expert_review_csv)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(dict(row))
    if not rows:
        return None
    write_csv(out_csv_path, rows)
    return str(out_csv_path)


def _selected_conditions(bundle: LoadedExperimentBundle, condition_ids: list[str] | None) -> list[Any]:
    conditions = bundle.condition_set.conditions
    if condition_ids:
        wanted = set(condition_ids)
        selected = [c for c in conditions if c.condition_id in wanted]
        missing = sorted(wanted - {c.condition_id for c in selected})
        if missing:
            raise KeyError(f"Unknown condition IDs: {missing}")
        disabled = [c.condition_id for c in selected if not c.enabled]
        if disabled:
            raise ValueError(f"Selected conditions are disabled in config: {disabled}")
        return selected
    return [c for c in conditions if c.enabled]


def run_experiment(
    *,
    bundle: LoadedExperimentBundle,
    condition_ids: list[str] | None = None,
    scenario_ids: list[str] | None = None,
    output_root_override: str | Path | None = None,
    session_name: str | None = None,
    parallel_conditions: bool = False,
    max_workers: int | None = None,
) -> ExperimentRunResult:
    output_root, session_dir = create_session_dir(
        bundle=bundle, output_root_override=output_root_override, session_name=session_name
    )
    conditions = _selected_conditions(bundle, condition_ids)
    indexed = list(enumerate(conditions))
    ordered_results: list[ConditionRunResult | None] = [None] * len(indexed)

    def job(pair: tuple[int, Any]) -> tuple[int, ConditionRunResult]:
        idx, condition = pair
        # Spread condition seeds to keep deterministic but distinct schedules.
        seed_base = int(bundle.experiment.seed) + (idx * 10_000)
        result = run_condition(
            bundle=bundle,
            condition=condition,
            session_dir=session_dir,
            scenario_ids=scenario_ids,
            seed_base=seed_base,
            parallel=None,
            max_workers=max_workers,
        )
        return idx, result

    if parallel_conditions and len(indexed) > 1:
        workers = max_workers or min(4, len(indexed))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(job, item) for item in indexed]
            for fut in as_completed(futures):
                idx, result = fut.result()
                ordered_results[idx] = result
    else:
        for item in indexed:
            idx, result = job(item)
            ordered_results[idx] = result

    condition_results = [x for x in ordered_results if x is not None]
    run_metrics = [rm for cr in condition_results for rm in cr.run_metrics]
    aggregate_reports = aggregate_run_metrics(run_metrics) if run_metrics else []

    session_summary_dir = ensure_dir(session_dir / "_session_summary")
    aggregate_paths = export_aggregate_metrics_files(session_summary_dir, aggregate_reports) if aggregate_reports else {}
    combined_expert = _concat_expert_csvs(condition_results, session_summary_dir / "expert_review_combined.csv")

    session_manifest = {
        "experiment_id": bundle.experiment.experiment_id,
        "session_dir": str(session_dir),
        "output_root": str(output_root),
        "conditions": [
            {
                "condition_id": cr.condition_id,
                "condition_summary_dir": str(cr.condition_summary_dir),
                "run_count": len(cr.run_results),
                "aggregate_paths": cr.aggregate_paths,
                "combined_expert_review_csv": cr.combined_expert_review_csv,
            }
            for cr in condition_results
        ],
        "aggregate_paths": aggregate_paths,
        "combined_expert_review_csv": combined_expert,
        "config_refs": bundle.config_refs,
    }
    session_manifest_path = write_json(session_summary_dir / "session_manifest.json", session_manifest)
    write_latest_pointer(output_root, session_dir)

    return ExperimentRunResult(
        session_dir=session_dir,
        output_root=output_root,
        condition_results=condition_results,
        aggregate_paths=aggregate_paths,
        aggregate_reports=[r.model_dump(mode="json") for r in aggregate_reports],
        combined_expert_review_csv=combined_expert,
        session_manifest_path=Path(session_manifest_path),
    )


def run_experiment_from_config(
    *,
    config_path: str | Path,
    env_file: str | Path | None = None,
    **kwargs: Any,
) -> ExperimentRunResult:
    if env_file:
        load_env_file_if_present(env_file)
    bundle = load_experiment_bundle(config_path)
    return run_experiment(bundle=bundle, **kwargs)

