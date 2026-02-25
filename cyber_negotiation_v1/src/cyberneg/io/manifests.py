from __future__ import annotations

from typing import Any

from ..core.enums import ConditionMode
from ..core.schemas import RunManifest, utc_now_iso


def make_run_manifest(
    *,
    run_id: str,
    experiment_id: str,
    condition_id: str,
    scenario_id: str,
    mode: ConditionMode,
    seed: int,
    config_refs: dict[str, Any],
    runtime_limits: dict[str, Any],
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        experiment_id=experiment_id,
        condition_id=condition_id,
        scenario_id=scenario_id,
        mode=mode,
        seed=seed,
        config_refs=config_refs,
        runtime_limits=runtime_limits,
    )


def finalize_manifest(manifest: RunManifest, *, status: str, output_paths: dict[str, str], error: str | None = None) -> RunManifest:
    manifest.finished_at = utc_now_iso()
    manifest.status = "success" if status == "success" else "failed"
    manifest.output_paths = output_paths
    manifest.error = error
    return manifest
