#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Batch-generate collective-behavior plots for all adversarial outputs. "
            "Writes one plot per poly_x* folder next to the data."
        )
    )
    ap.add_argument(
        "--output-root",
        default="games_descriptions/polynomial_game_adversarial/output",
        help="Root folder containing adversarial outputs (default: games_descriptions/polynomial_game_adversarial/output)",
    )
    ap.add_argument(
        "--best-x",
        type=int,
        default=2,
        help="Best feasible x reference used for regret (default: 2)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing collective_behavior.png files",
    )
    args = ap.parse_args()

    output_root = Path(args.output_root)
    if not output_root.exists():
        raise SystemExit(f"Missing output root: {output_root}")

    script = Path("scripts/visualize_collective_behavior_across_runs.py")
    if not script.exists():
        raise SystemExit(f"Missing script: {script}")

    # Pattern:
    # games_descriptions/polynomial_game_adversarial/output/<mode>/<game>/<output_*/poly_x*/...>
    poly_dirs = sorted(output_root.glob("*/*/output_*/poly_x*"))
    if not poly_dirs:
        raise SystemExit(f"No poly_x* folders found under {output_root}")

    total = 0
    made = 0
    skipped = 0

    for poly_dir in poly_dirs:
        total += 1
        try:
            mode = poly_dir.parts[poly_dir.parts.index("output") + 1]
            game = poly_dir.parts[poly_dir.parts.index("output") + 2]
        except Exception:
            print(f"[skip] Unexpected path shape: {poly_dir}")
            skipped += 1
            continue

        game_dir = Path("games_descriptions/polynomial_game_adversarial") / mode / game
        if not game_dir.exists():
            print(f"[skip] Missing game_dir for {poly_dir}: {game_dir}")
            skipped += 1
            continue

        out_png = poly_dir / "collective_behavior.png"
        if out_png.exists() and not args.overwrite:
            skipped += 1
            continue

        cmd = [
            "python",
            str(script),
            "--runs-root",
            str(poly_dir),
            "--game-dir",
            str(game_dir),
            "--save",
            str(out_png),
            "--best-x",
            str(args.best_x),
        ]
        try:
            subprocess.run(cmd, check=True)
            made += 1
        except subprocess.CalledProcessError:
            print(f"[error] Failed: {poly_dir}")
            skipped += 1

    print(f"Done. total={total} made={made} skipped={skipped}")


if __name__ == "__main__":
    main()

