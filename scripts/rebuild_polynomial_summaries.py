#!/usr/bin/env python3
"""
Rebuild thesis-facing polynomial summaries from the consolidated output root.

Generated artifacts:
  - viewer/metrics_summary.json
  - viewer/dynamics_summary.json
  - viewer/history_manifest.json
  - summarys/metrics_summary.generated.json
  - summarys/metrics_summary.adversarial_obstructive.json
  - summarys/metrics_summary.adversarial_outcome_targeted.json
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASELINE_ROOTS = [
    "polynomial/outputs/output_mix_all_diff",
    "polynomial/outputs/output_mix_split",
    "polynomial/outputs/polynomial_game/output",
    "polynomial/outputs/polynomial_game/output_claude",
    "polynomial/outputs/polynomial_game/output_llama",
    "polynomial/outputs/polynomial_game_all_AI/output",
    "polynomial/outputs/polynomial_game_all_AI/output_claude",
    "polynomial/outputs/polynomial_game_all_AI/output_llama",
    "polynomial/outputs/polynomial_game_human/output",
    "polynomial/outputs/polynomial_game_human/output_claude",
    "polynomial/outputs/polynomial_game_human/output_llama",
]

ADVERSARIAL_ROOTS = {
    "obstructive": "polynomial/outputs/polynomial_game_adversarial/output/obstructive",
    "outcome_targeted": "polynomial/outputs/polynomial_game_adversarial/output/outcome_targeted",
}

METRICS_PATH = ROOT / "viewer" / "metrics_summary.json"
GENERATED_METRICS_PATH = ROOT / "summarys" / "metrics_summary.generated.json"
DYNAMICS_PATH = ROOT / "viewer" / "dynamics_summary.json"
HISTORY_MANIFEST_PATH = ROOT / "viewer" / "history_manifest.json"
ADVERSARIAL_OUTPUTS = {
    "obstructive": ROOT / "summarys" / "metrics_summary.adversarial_obstructive.json",
    "outcome_targeted": ROOT / "summarys" / "metrics_summary.adversarial_outcome_targeted.json",
}


def run_script(script: str, *args: str) -> None:
    cmd = [sys.executable, script, *args]
    print("+", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_history_manifest(metrics_path: Path, manifest_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    runs = metrics.get("runs") or []
    entries = []
    for row in runs:
        path = str(row.get("path", "")).replace("\\", "/")
        entries.append(
            {
                "path": path,
                "category": row.get("category"),
                "variant": row.get("variant"),
                "group": row.get("group"),
                "file": Path(path).name,
            }
        )
    out = {
        "generated_from": metrics.get("generated_from") or ["viewer/metrics_summary.json"],
        "count": len(entries),
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(entries)} manifest entries to {manifest_path.relative_to(ROOT)}")


def main() -> None:
    run_script(
        "scripts/build_metrics_summary.py",
        *BASELINE_ROOTS,
        "--output",
        str(METRICS_PATH.relative_to(ROOT)),
    )
    GENERATED_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(METRICS_PATH, GENERATED_METRICS_PATH)
    print(f"Copied baseline metrics to {GENERATED_METRICS_PATH.relative_to(ROOT)}")

    run_script(
        "scripts/build_dynamics_summary.py",
        "--summary",
        str(METRICS_PATH.relative_to(ROOT)),
        "--out",
        str(DYNAMICS_PATH.relative_to(ROOT)),
    )
    write_history_manifest(METRICS_PATH, HISTORY_MANIFEST_PATH)

    for mode, root in ADVERSARIAL_ROOTS.items():
        run_script(
            "scripts/build_metrics_summary.py",
            root,
            "--output",
            str(ADVERSARIAL_OUTPUTS[mode].relative_to(ROOT)),
        )


if __name__ == "__main__":
    main()
