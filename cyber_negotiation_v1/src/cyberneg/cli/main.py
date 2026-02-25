from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..core.metrics import aggregate_run_metrics
from ..core.schemas import RunMetrics
from ..core.scheduler import generate_public_schedule
from ..io.exports import export_aggregate_metrics_files
from ..io.loaders import dump_bundle_summary, load_env_file_if_present, load_experiment_bundle
from ..io.storage import ensure_dir, write_csv, write_json
from ..runner.run_condition import run_condition
from ..runner.run_experiment import create_session_dir, run_experiment


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _default_env_candidates(config_path: Path) -> list[Path]:
    candidates: list[Path] = []
    cur = config_path.resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        candidates.append(parent / ".env")
    return candidates


def _load_bundle_with_env(config_path: str, env_file: str | None) -> Any:
    cfg_path = Path(config_path).resolve()
    if env_file:
        load_env_file_if_present(env_file)
    else:
        for p in _default_env_candidates(cfg_path):
            if p.exists():
                load_env_file_if_present(p)
    return load_experiment_bundle(cfg_path)


def _find_run_metrics_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("run_metrics.json") if p.parent.name == "metrics"]
    return sorted(files)


def _find_expert_review_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("expert_review.csv") if p.parent.name == "exports"]
    return sorted(files)


def _load_run_metrics(root: Path) -> list[RunMetrics]:
    out: list[RunMetrics] = []
    for path in _find_run_metrics_files(root):
        out.append(RunMetrics.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    return out


def _cmd_validate_config(args: argparse.Namespace) -> int:
    bundle = _load_bundle_with_env(args.config, args.env_file)
    _print_json(
        {
            "ok": True,
            "bundle_summary": dump_bundle_summary(bundle),
            "config_refs": bundle.config_refs,
        }
    )
    return 0


def _cmd_dry_run(args: argparse.Namespace) -> int:
    bundle = _load_bundle_with_env(args.config, args.env_file)
    conds = [c for c in bundle.condition_set.conditions if c.enabled]
    if args.condition_id:
        conds = [c for c in conds if c.condition_id == args.condition_id]
    if not conds:
        raise ValueError("No enabled conditions match the selection")
    scenario_ids = args.scenario_ids or list(bundle.scenarios.keys())
    preview: list[dict[str, Any]] = []
    for condition in conds:
        item: dict[str, Any] = {
            "condition_id": condition.condition_id,
            "mode": condition.mode.value,
            "scenario_ids": scenario_ids,
            "runtime": condition.runtime.model_dump(mode="json"),
        }
        if condition.mode.value == "negotiation":
            sched = generate_public_schedule(condition.runtime.public_messages or 0, seed=bundle.experiment.seed + 101)
            item["scheduler_preview"] = {
                "order_seed": sched.order_seed,
                "public_order": [r.value for r in sched.public_order],
                "role_counts": {k.value: v for k, v in sched.role_counts.items()},
            }
        preview.append(item)
    _print_json({"ok": True, "experiment_id": bundle.experiment.experiment_id, "dry_run": preview})
    return 0


def _cmd_run_scenario(args: argparse.Namespace) -> int:
    bundle = _load_bundle_with_env(args.config, args.env_file)
    condition = next((c for c in bundle.condition_set.conditions if c.condition_id == args.condition_id), None)
    if condition is None:
        raise KeyError(f"Unknown condition_id: {args.condition_id}")
    if args.scenario_id not in bundle.scenarios:
        raise KeyError(f"Unknown scenario_id: {args.scenario_id}")
    _, session_dir = create_session_dir(bundle=bundle, output_root_override=args.output_root, session_name=args.session_name)
    result = run_condition(
        bundle=bundle,
        condition=condition,
        session_dir=session_dir,
        scenario_ids=[args.scenario_id],
        parallel=False,
    )
    _print_json(
        {
            "ok": True,
            "session_dir": str(result.session_dir),
            "condition_id": result.condition_id,
            "run_count": len(result.run_results),
            "aggregate_paths": result.aggregate_paths,
            "condition_summary_dir": str(result.condition_summary_dir),
        }
    )
    return 0


def _cmd_run_condition(args: argparse.Namespace) -> int:
    bundle = _load_bundle_with_env(args.config, args.env_file)
    condition = next((c for c in bundle.condition_set.conditions if c.condition_id == args.condition_id), None)
    if condition is None:
        raise KeyError(f"Unknown condition_id: {args.condition_id}")
    _, session_dir = create_session_dir(bundle=bundle, output_root_override=args.output_root, session_name=args.session_name)
    result = run_condition(
        bundle=bundle,
        condition=condition,
        session_dir=session_dir,
        scenario_ids=args.scenario_ids,
        parallel=args.parallel if args.parallel is not None else None,
        max_workers=args.max_workers,
    )
    _print_json(
        {
            "ok": True,
            "session_dir": str(result.session_dir),
            "condition_id": result.condition_id,
            "run_count": len(result.run_results),
            "aggregate_paths": result.aggregate_paths,
            "condition_summary_dir": str(result.condition_summary_dir),
            "combined_expert_review_csv": result.combined_expert_review_csv,
        }
    )
    return 0


def _cmd_run_experiment(args: argparse.Namespace) -> int:
    bundle = _load_bundle_with_env(args.config, args.env_file)
    result = run_experiment(
        bundle=bundle,
        condition_ids=args.condition_ids,
        scenario_ids=args.scenario_ids,
        output_root_override=args.output_root,
        session_name=args.session_name,
        parallel_conditions=bool(args.parallel_conditions),
        max_workers=args.max_workers,
    )
    _print_json(
        {
            "ok": True,
            "session_dir": str(result.session_dir),
            "output_root": str(result.output_root),
            "condition_count": len(result.condition_results),
            "aggregate_paths": result.aggregate_paths,
            "combined_expert_review_csv": result.combined_expert_review_csv,
            "session_manifest": str(result.session_manifest_path),
        }
    )
    return 0


def _cmd_compute_metrics(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    run_metrics = _load_run_metrics(root)
    aggregate = aggregate_run_metrics(run_metrics)
    out_dir = ensure_dir(Path(args.output_dir).resolve()) if args.output_dir else ensure_dir(root / "_recomputed_metrics")
    paths = export_aggregate_metrics_files(out_dir, aggregate)
    _print_json(
        {
            "ok": True,
            "root": str(root),
            "run_metrics_count": len(run_metrics),
            "output_dir": str(out_dir),
            "aggregate_paths": paths,
        }
    )
    return 0


def _cmd_export_expert_csv(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    files = _find_expert_review_files(root)
    rows: list[dict[str, Any]] = []
    for p in files:
        with p.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(dict(row))
    out_path = Path(args.output_csv).resolve() if args.output_csv else (root / "_expert_review_combined.csv")
    write_csv(out_path, rows)
    _print_json(
        {
            "ok": True,
            "root": str(root),
            "files_found": len(files),
            "rows_written": len(rows),
            "output_csv": str(out_path),
        }
    )
    return 0


def _cmd_launch_dashboard(args: argparse.Namespace) -> int:
    dashboard_path = Path(__file__).resolve().parents[1] / "ui" / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--", "--root", str(Path(args.root))]
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def _cmd_scaffold_scenario(args: argparse.Namespace) -> int:
    out = Path(args.output).resolve()
    if out.exists() and not args.force:
        raise FileExistsError(f"File exists: {out} (use --force to overwrite)")
    template = f"""scenario_id: {args.scenario_id}
title: "Placeholder scenario title"
source_family: "generic webapp"
difficulty: medium
label_set_id: default_findings_v1
lines:
  - id: L001
    text: "Placeholder evidence line with observable behavior."
  - id: L002
    text: "Placeholder HTTP response snippet or log line."
  - id: L003
    text: "Placeholder reproduction note or screenshot transcript."
author_notes: "Hidden notes for scenario authors/adjudicators only."
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template, encoding="utf-8")
    _print_json({"ok": True, "output": str(out), "scenario_id": args.scenario_id})
    return 0


def _cmd_scaffold_condition(args: argparse.Namespace) -> int:
    out = Path(args.output).resolve()
    if out.exists() and not args.force:
        raise FileExistsError(f"File exists: {out} (use --force to overwrite)")
    template = """condition_set_id: condition_set_template_v1
description: "Template condition set - fill in C1..C7 as needed"
conditions:
  - condition_id: C1
    enabled: true
    mode: negotiation
    priors:
      apply_in_round0_only: true
      text: "You are negotiating with other LLMs."
    models_by_role:
      R: mock-default
      C: mock-default
      K: mock-default
    prompt_options:
      re_inject_role_instruction_in_roundn: false
      reminder_text: "Stay evidence-grounded."
    runtime:
      public_messages: 6
      json_max_retries: 3
      provider_timeout_seconds: 30
      per_run_wallclock_limit_seconds: 180
      token_budget_limit: null
      final_turn_announcement_window: 1
      parallel_runs: false
      public_message_min_words: 80
      public_message_max_words: 150
      public_message_hard_cap_words: 220
  - condition_id: C2
    enabled: true
    mode: baseline
    priors:
      apply_in_round0_only: true
      text: "You are performing a one-shot solo review."
    baseline_model: mock-default
    runtime:
      json_max_retries: 3
      provider_timeout_seconds: 30
      per_run_wallclock_limit_seconds: 120
      token_budget_limit: null
      public_message_min_words: 80
      public_message_max_words: 150
      public_message_hard_cap_words: 220
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template, encoding="utf-8")
    _print_json({"ok": True, "output": str(out)})
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyberneg", description="Cyber negotiation V1 scaffold CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, help="Experiment config YAML path")
        p.add_argument("--env-file", default=None, help="Optional .env path")

    p = sub.add_parser("validate-config")
    add_common_config(p)
    p.set_defaults(func=_cmd_validate_config)

    p = sub.add_parser("dry-run")
    add_common_config(p)
    p.add_argument("--condition-id")
    p.add_argument("--scenario-ids", nargs="*")
    p.set_defaults(func=_cmd_dry_run)

    p = sub.add_parser("run-scenario")
    add_common_config(p)
    p.add_argument("--condition-id", required=True)
    p.add_argument("--scenario-id", required=True)
    p.add_argument("--output-root")
    p.add_argument("--session-name")
    p.set_defaults(func=_cmd_run_scenario)

    p = sub.add_parser("run-condition")
    add_common_config(p)
    p.add_argument("--condition-id", required=True)
    p.add_argument("--scenario-ids", nargs="*")
    p.add_argument("--output-root")
    p.add_argument("--session-name")
    p.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-workers", type=int)
    p.set_defaults(func=_cmd_run_condition)

    p = sub.add_parser("run-experiment")
    add_common_config(p)
    p.add_argument("--condition-ids", nargs="*")
    p.add_argument("--scenario-ids", nargs="*")
    p.add_argument("--output-root")
    p.add_argument("--session-name")
    p.add_argument("--parallel-conditions", action="store_true")
    p.add_argument("--max-workers", type=int)
    p.set_defaults(func=_cmd_run_experiment)

    p = sub.add_parser("compute-metrics")
    p.add_argument("--root", required=True, help="Root folder to scan for run_metrics.json files")
    p.add_argument("--output-dir", help="Where to write aggregate metrics artifacts")
    p.set_defaults(func=_cmd_compute_metrics)

    p = sub.add_parser("export-expert-csv")
    p.add_argument("--root", required=True, help="Root folder to scan for expert_review.csv files")
    p.add_argument("--output-csv", help="Combined CSV output path")
    p.set_defaults(func=_cmd_export_expert_csv)

    p = sub.add_parser("launch-dashboard")
    p.add_argument("--root", default="outputs", help="Output root directory")
    p.set_defaults(func=_cmd_launch_dashboard)

    p = sub.add_parser("scaffold-scenario")
    p.add_argument("--output", required=True)
    p.add_argument("--scenario-id", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_scaffold_scenario)

    p = sub.add_parser("scaffold-condition")
    p.add_argument("--output", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_scaffold_condition)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        _print_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

