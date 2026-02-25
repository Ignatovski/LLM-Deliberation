from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.metrics import aggregate_run_metrics
from ..core.schemas import ConditionConfig, RunMetrics
from ..io.exports import export_aggregate_metrics_files
from ..io.loaders import LoadedExperimentBundle
from ..io.storage import ensure_dir, write_csv, write_json
from .run_baseline import BaselineRunResult, run_baseline_once
from .run_negotiation import NegotiationRunResult, run_negotiation_once


@dataclass
class ConditionRunResult:
    condition_id: str
    session_dir: Path
    condition_summary_dir: Path
    run_results: list[Any]
    run_metrics: list[RunMetrics]
    aggregate_paths: dict[str, str]
    aggregate_reports: list[Any]
    combined_expert_review_csv: Optional[str]


def _scenario_run_seed(seed_base: int, scenario_index: int) -> int:
    return int(seed_base) + int(scenario_index)


def _run_single(
    *,
    bundle: LoadedExperimentBundle,
    condition: ConditionConfig,
    scenario_id: str,
    session_dir: Path,
    run_seed: int,
) -> BaselineRunResult | NegotiationRunResult:
    scenario = bundle.scenarios[scenario_id]
    gt = bundle.ground_truths.get(scenario_id)
    if condition.mode.value == "baseline":
        return run_baseline_once(
            bundle=bundle,
            condition=condition,
            scenario=scenario,
            ground_truth=gt,
            session_dir=session_dir,
            run_seed=run_seed,
        )
    return run_negotiation_once(
        bundle=bundle,
        condition=condition,
        scenario=scenario,
        ground_truth=gt,
        session_dir=session_dir,
        run_seed=run_seed,
    )


def _concat_expert_rows(run_results: list[Any], out_csv_path: Path) -> Optional[str]:
    rows: list[dict[str, Any]] = []
    for rr in run_results:
        csv_path = rr.manifest.output_paths.get("expert_review_csv") if getattr(rr, "manifest", None) else None
        if not csv_path:
            continue
        p = Path(csv_path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(dict(row))
    if not rows:
        return None
    write_csv(out_csv_path, rows)
    return str(out_csv_path)


def _write_condition_index(
    *,
    condition_summary_dir: Path,
    bundle: LoadedExperimentBundle,
    condition: ConditionConfig,
    run_results: list[Any],
    aggregate_paths: dict[str, str],
    combined_expert_review_csv: Optional[str],
) -> str:
    payload = {
        "experiment_id": bundle.experiment.experiment_id,
        "condition_id": condition.condition_id,
        "mode": condition.mode.value,
        "run_count": len(run_results),
        "run_dirs": [str(getattr(rr, "run_dir", "")) for rr in run_results],
        "run_manifests": [getattr(rr, "manifest", None).model_dump(mode="json") for rr in run_results],
        "aggregate_paths": aggregate_paths,
        "combined_expert_review_csv": combined_expert_review_csv,
    }
    p = write_json(condition_summary_dir / "condition_index.json", payload)
    return str(p)


def run_condition(
    *,
    bundle: LoadedExperimentBundle,
    condition: ConditionConfig,
    session_dir: str | Path,
    scenario_ids: Optional[list[str]] = None,
    seed_base: Optional[int] = None,
    parallel: Optional[bool] = None,
    max_workers: Optional[int] = None,
) -> ConditionRunResult:
    session_dir = ensure_dir(session_dir)
    selected_ids = scenario_ids or list(bundle.scenarios.keys())
    missing = [sid for sid in selected_ids if sid not in bundle.scenarios]
    if missing:
        raise KeyError(f"Unknown scenario IDs for condition {condition.condition_id}: {missing}")

    use_parallel = bool(condition.runtime.parallel_runs) if parallel is None else bool(parallel)
    seed_start = bundle.experiment.seed if seed_base is None else seed_base
    indexed = list(enumerate(selected_ids))

    ordered_results: list[Any | None] = [None] * len(indexed)

    def job(pair: tuple[int, str]) -> tuple[int, Any]:
        idx, scenario_id = pair
        rr = _run_single(
            bundle=bundle,
            condition=condition,
            scenario_id=scenario_id,
            session_dir=session_dir,
            run_seed=_scenario_run_seed(seed_start, idx),
        )
        return idx, rr

    if use_parallel and len(indexed) > 1:
        workers = max_workers or min(8, len(indexed))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(job, item) for item in indexed]
            for fut in as_completed(futures):
                idx, rr = fut.result()
                ordered_results[idx] = rr
    else:
        for item in indexed:
            idx, rr = job(item)
            ordered_results[idx] = rr

    run_results = [rr for rr in ordered_results if rr is not None]
    run_metrics = [rr.run_metrics for rr in run_results]
    aggregate_reports = aggregate_run_metrics(run_metrics) if run_metrics else []

    condition_summary_dir = ensure_dir(session_dir / condition.condition_id / "_condition_summary")
    aggregate_paths = (
        export_aggregate_metrics_files(condition_summary_dir, aggregate_reports) if aggregate_reports else {}
    )
    combined_expert = _concat_expert_rows(run_results, condition_summary_dir / "expert_review_combined.csv")
    _write_condition_index(
        condition_summary_dir=condition_summary_dir,
        bundle=bundle,
        condition=condition,
        run_results=run_results,
        aggregate_paths=aggregate_paths,
        combined_expert_review_csv=combined_expert,
    )

    return ConditionRunResult(
        condition_id=condition.condition_id,
        session_dir=Path(session_dir),
        condition_summary_dir=condition_summary_dir,
        run_results=run_results,
        run_metrics=run_metrics,
        aggregate_paths=aggregate_paths,
        aggregate_reports=[r.model_dump(mode="json") for r in aggregate_reports],
        combined_expert_review_csv=combined_expert,
    )

