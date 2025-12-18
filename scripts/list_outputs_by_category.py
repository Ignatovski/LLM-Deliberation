#!/usr/bin/env python3
"""
Create a categorized list of output runs grouped by category and model mix.

Example:
  python scripts/list_outputs_by_category.py --output summarys/outputs_by_category.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


OUTPUT_MARKERS = (
    "output_mix_all_diff",
    "output_mix_split",
    "output_xyz_reminder",
    "output_xyz",
    "output_claude",
    "output_llama",
    "output",
)


def iter_output_roots() -> List[Path]:
    roots: List[Path] = []
    for entry in REPO_ROOT.iterdir():
        if entry.is_dir() and entry.name.startswith("output"):
            roots.append(entry)
    games_dir = REPO_ROOT / "games_descriptions"
    if games_dir.exists():
        for game_dir in games_dir.iterdir():
            if not game_dir.is_dir():
                continue
            for out_dir in game_dir.iterdir():
                if out_dir.is_dir() and out_dir.name.startswith("output"):
                    roots.append(out_dir)
    return roots


def iter_run_dirs(root: Path) -> Set[Path]:
    run_dirs: Set[Path] = set()
    for path in root.rglob("history*.json"):
        if path.is_file():
            run_dirs.add(path.parent)
    for path in root.rglob("config*.txt"):
        if path.is_file():
            run_dirs.add(path.parent)
    return run_dirs


def find_first(path: Path, pattern: str) -> Optional[Path]:
    for candidate in sorted(path.glob(pattern)):
        if candidate.is_file():
            return candidate
    return None


def infer_game_dir(run_dir: Path) -> Optional[Path]:
    parts = run_dir.parts
    if "games_descriptions" in parts:
        idx = parts.index("games_descriptions")
        if idx + 1 < len(parts):
            return Path(*parts[: idx + 2])
    for marker in OUTPUT_MARKERS:
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                game = parts[idx + 1]
                candidate = REPO_ROOT / "games_descriptions" / game
                if candidate.exists():
                    return candidate
    return None


def pick_config(run_dir: Path, game_dir: Optional[Path]) -> Optional[Path]:
    cfg = find_first(run_dir, "config*.txt")
    if cfg:
        return cfg
    if game_dir:
        for name in ("config_origianl.txt", "config_original.txt"):
            candidate = game_dir / name
            if candidate.exists():
                return candidate
        cfg = find_first(game_dir, "config*.txt")
        if cfg:
            return cfg
    return None


def parse_model_mix(cfg: Optional[Path]) -> str:
    if cfg is None:
        return "unknown_mix"
    try:
        lines = cfg.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unknown_mix"
    models: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p.strip() for p in stripped.split(",") if p.strip()]
        if not parts:
            continue
        models.append(parts[-1])
    if not models:
        return "unknown_mix"
    counts: Dict[str, int] = {}
    for model in models:
        counts[model] = counts.get(model, 0) + 1
    chunks = []
    for model in sorted(counts):
        count = counts[model]
        chunks.append(f"{model}x{count}" if count > 1 else model)
    return "+".join(chunks)


def category_label(run_dir: Path) -> str:
    path = str(run_dir)
    if "polynomial_game_all_AI" in path:
        return "All AI"
    if "polynomial_game_human" in path:
        return "All Human"
    if "polynomial_game" in path:
        return "Uniform"
    if "output_xyz_reminder" in path:
        return "XYZ Reminder"
    if "output_xyz" in path:
        return "XYZ"
    if "output" in path:
        return "Output"
    return "Other"


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="List outputs grouped by category/model mix.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("summarys/outputs_by_category.txt"),
        help="Output path for the categorized list.",
    )
    args = parser.parse_args()

    output_roots = iter_output_roots()
    by_category: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for root in output_roots:
        for run_dir in iter_run_dirs(root):
            game_dir = infer_game_dir(run_dir)
            cfg = pick_config(run_dir, game_dir)
            mix = parse_model_mix(cfg)
            category = category_label(run_dir)
            variant_dir = run_dir.parent
            variant_key = rel_path(variant_dir)
            by_category[category][mix][variant_key].append(run_dir.name)

    lines: List[str] = []
    for category in sorted(by_category):
        mixes = by_category[category]
        total = sum(
            sum(len(runs) for runs in variants.values()) for variants in mixes.values()
        )
        lines.append(f"Category: {category} (runs: {total})")
        for mix in sorted(mixes):
            variants = mixes[mix]
            mix_total = sum(len(runs) for runs in variants.values())
            lines.append(f"  Model mix: {mix} (runs: {mix_total})")
            for variant in sorted(variants):
                runs = sorted(variants[variant])
                lines.append(
                    f"    - {variant}: {', '.join(runs)}"
                )
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
