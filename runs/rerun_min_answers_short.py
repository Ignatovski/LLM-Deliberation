#!/usr/bin/env python3
"""
Rerun runs that violate min_answers (short runs).

Reads summarys/min_answers_short_runs.txt, infers game_dir/config/seed/output_dir,
updates initial_deal.txt per seed, and replays each run with --min_answers 16.

Usage:
  python runs/rerun_min_answers_short.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SHORT_LIST = REPO_ROOT / "summarys" / "min_answers_short_runs.txt"
MIN_ANSWERS = 16


def iter_runs(path: Path) -> Iterable[Path]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield (REPO_ROOT / line).resolve()


def parse_seed(variant: str) -> Optional[str]:
    match = re.search(r"poly_x-?(-?\d+)", variant)
    if not match:
        return None
    return match.group(1)


def find_first(path: Path, pattern: str) -> Optional[Path]:
    for candidate in sorted(path.glob(pattern)):
        if candidate.is_file():
            return candidate
    return None


def pick_config(run_dir: Path, game_dir: Path) -> Optional[Path]:
    cfg = find_first(run_dir, "config*.txt")
    if cfg:
        return cfg
    for name in ("config_origianl.txt", "config_original.txt"):
        candidate = game_dir / name
        if candidate.exists():
            return candidate
    return find_first(game_dir, "config*.txt")


def infer_game_dir(run_dir: Path) -> Optional[Path]:
    parts = run_dir.parts
    if "games_descriptions" in parts:
        idx = parts.index("games_descriptions")
        if idx + 1 < len(parts):
            return Path(*parts[: idx + 2])
    for marker in ("output_mix_all_diff", "output_mix_split"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                game_name = parts[idx + 1]
                return REPO_ROOT / "games_descriptions" / game_name
    return None


def set_initial_deal(game_dir: Path, seed: str) -> Path:
    initial = game_dir / "initial_deal.txt"
    backup = game_dir / ".initial_deal.min_answers.bak"
    if initial.exists() and not backup.exists():
        backup.write_text(initial.read_text(encoding="utf-8"), encoding="utf-8")
    initial.write_text(f"<VALUE>{seed}</VALUE>\n", encoding="utf-8")
    return backup


def restore_initial_deal(game_dir: Path) -> None:
    backup = game_dir / ".initial_deal.min_answers.bak"
    initial = game_dir / "initial_deal.txt"
    if backup.exists():
        initial.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        backup.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun short min_answers runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Path to .env file to pass into main_polynomial.py (default: repo .env).",
    )
    parser.add_argument(
        "--azure",
        action="store_true",
        help="Force Azure mode (adds --azure when invoking main_polynomial.py).",
    )
    args = parser.parse_args()

    if not SHORT_LIST.exists():
        raise SystemExit(f"Missing {SHORT_LIST}")

    env_file = Path(args.env_file)
    env_keys = set()
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key:
                env_keys.add(key)

    use_azure = args.azure or bool(
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or ("AZURE_OPENAI_ENDPOINT" in env_keys)
        or ("AZURE_OPENAI_API_KEY" in env_keys)
    )

    touched = set()
    for run_dir in iter_runs(SHORT_LIST):
        variant = run_dir.parent.name
        seed = parse_seed(variant)
        if seed is None:
            print(f"Skipping {run_dir}: cannot parse seed from {variant}")
            continue
        game_dir = infer_game_dir(run_dir)
        if game_dir is None or not game_dir.exists():
            print(f"Skipping {run_dir}: cannot infer game_dir")
            continue
        cfg = pick_config(run_dir, game_dir)
        if cfg is None:
            print(f"Skipping {run_dir}: missing config*.txt (run or game_dir)")
            continue
        output_parent = run_dir.parent.resolve()
        try:
            output_dir = output_parent.relative_to(game_dir)
        except ValueError:
            output_dir = output_parent

        exp_name = run_dir.name

        set_initial_deal(game_dir, seed)
        touched.add(game_dir)

        cmd = [
            "python",
            "main_polynomial.py",
            "--exp_name",
            exp_name,
            "--game_dir",
            str(game_dir),
            "--config_file",
            str(cfg),
            "--output_dir",
            str(output_dir),
            "--temp",
            "1",
            "--reuse_faiss",
            "--result",
            seed,
            "--min_answers",
            str(MIN_ANSWERS),
            "--env-file",
            str(env_file),
        ]
        if use_azure:
            cmd.append("--azure")

        print(" ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Run failed: {run_dir}")

    for game_dir in touched:
        restore_initial_deal(game_dir)


if __name__ == "__main__":
    main()
