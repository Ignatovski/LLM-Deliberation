from __future__ import annotations

import json
from pathlib import Path

from cyberneg.io.loaders import load_experiment_bundle
from cyberneg.runner.run_experiment import run_experiment


def test_mock_end_to_end_experiment_runs_baseline_and_negotiation(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "examples" / "configs" / "mock_experiment.yaml"
    bundle = load_experiment_bundle(config_path)

    # Keep tests independent from optional plotting deps.
    bundle.experiment.runtime_overrides.generate_plots = False
    bundle.experiment.runtime_overrides.export_json_schemas = True

    result = run_experiment(
        bundle=bundle,
        output_root_override=tmp_path / "outputs",
        session_name="test_session",
        parallel_conditions=False,
    )

    assert result.session_dir.exists()
    assert result.session_manifest_path.exists()
    assert len(result.condition_results) == 2

    all_manifests = [rr.manifest for cr in result.condition_results for rr in cr.run_results]
    modes = {m.mode.value for m in all_manifests}
    assert modes == {"negotiation", "baseline"}

    # Inspect the negotiation run artifacts.
    neg_run = next(
        rr
        for cr in result.condition_results
        for rr in cr.run_results
        if rr.manifest.mode.value == "negotiation"
    )
    turns_path = neg_run.run_dir / "transcript" / "turns.json"
    public_history_path = neg_run.run_dir / "transcript" / "public_history.json"
    attempts_path = neg_run.run_dir / "provider" / "attempts.json"
    assert turns_path.exists()
    assert public_history_path.exists()
    assert attempts_path.exists()

    turns = json.loads(turns_path.read_text(encoding="utf-8"))
    public_history = json.loads(public_history_path.read_text(encoding="utf-8"))
    attempts = json.loads(attempts_path.read_text(encoding="utf-8"))

    assert len(public_history) == 6
    assert len(turns) >= 9  # 3 round0 + 6 public when successful
    assert len(attempts) > len(turns)  # mock config injects one invalid JSON attempt per turn

    # Public history should not expose private fields.
    assert all("private_notes" not in msg for msg in public_history)
    assert all("private_plan" not in msg for msg in public_history)

    # Session-level aggregates and expert CSV should exist.
    assert result.aggregate_paths["aggregate_metrics_json"]
    assert Path(result.aggregate_paths["aggregate_metrics_json"]).exists()
    assert result.combined_expert_review_csv is not None
    assert Path(result.combined_expert_review_csv).exists()

