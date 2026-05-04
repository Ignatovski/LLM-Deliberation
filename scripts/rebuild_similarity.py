#!/usr/bin/env python3
"""
Rebuild FAISS similarity index from saved histories and regenerate similarity outputs.

Example:
  python scripts/rebuild_similarity.py \
    --output-root polynomial/outputs/polynomial_game/output/Uniformed/poly_x-7 \
    --runs 3.1 4.1 6.1 10.1 14.1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from faiss_utility import AnswerComparator
from evaluation.plot_similarity import compute_agent_matrices, load_embeddings, plot_matrices


def iter_run_dirs(output_root: Path) -> Iterable[Path]:
    for candidate in sorted(output_root.iterdir()):
        if not candidate.is_dir():
            continue
        if any(candidate.glob("history*.json")):
            yield candidate


def find_history(run_dir: Path) -> Optional[Path]:
    histories = sorted(run_dir.glob("history*.json"))
    return histories[0] if histories else None


def infer_index_dir(output_root: Path) -> Path:
    direct = output_root / "faiss_index"
    if direct.exists():
        return direct
    parts = output_root.parts
    if "output" in parts:
        idx = parts.index("output")
        if idx + 1 < len(parts):
            rel = Path(*parts[idx + 1 :])
            return REPO_ROOT / "output" / rel / "faiss_index"
    return output_root / "faiss_index"


def parse_rounds(history_path: Path) -> Tuple[List[Dict], List[str]]:
    data = json.loads(history_path.read_text(encoding="utf-8"))
    rounds = data.get("rounds") or []
    agents = []
    for entry in rounds:
        agent = entry.get("agent")
        if agent and agent not in agents:
            agents.append(agent)
    return rounds, agents


def pick_existing_similarity(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("similarity_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return None


def resolve_run(output_root: Path, token: str) -> Path:
    candidate = Path(token)
    if candidate.is_absolute():
        return candidate
    if token.startswith(str(output_root)):
        return Path(token)
    return output_root / token


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild similarity index and reports.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("polynomial/outputs/polynomial_game/output/Uniformed/poly_x-7"),
        help="Directory containing run folders (default: consolidated Uniformed poly_x-7 root).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="FAISS index directory (default: inferred from output-root).",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run folder names or paths to regenerate similarity for (default: all).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not back up the existing faiss_index folder before rebuilding.",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if not output_root.exists():
        raise SystemExit(f"Output root not found: {output_root}")

    index_dir = (args.index_dir or infer_index_dir(output_root)).expanduser().resolve()
    if index_dir.exists() and not args.no_backup:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = index_dir.with_name(f"{index_dir.name}.backup_{stamp}")
        shutil.move(str(index_dir), str(backup_dir))
        print(f"Moved existing index to {backup_dir}")

    comparator = AnswerComparator(
        str(index_dir),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=None,
        reuse_existing=True,
    )

    run_dirs = list(iter_run_dirs(output_root))
    if not run_dirs:
        raise SystemExit(f"No runs found under {output_root}")

    run_ids: Dict[Path, str] = {}
    for run_dir in run_dirs:
        history_path = find_history(run_dir)
        if history_path is None:
            continue
        rounds, _ = parse_rounds(history_path)
        existing = pick_existing_similarity(run_dir)
        if existing:
            run_id = existing.stem.split("similarity_", 1)[-1]
        else:
            run_id = str(uuid.uuid4())
        run_ids[run_dir] = run_id
        for idx, entry in enumerate(rounds):
            agent = entry.get("agent")
            if not agent:
                continue
            answer = entry.get("public_answer")
            if not isinstance(answer, str):
                answer = "" if answer is None else str(answer)
            comparator.add_answer(
                answer=answer,
                agent_name=agent,
                round_num=idx,
                run_id=run_id,
            )

    emb_path = Path(comparator.embeddings_file)
    embeddings, metadata = load_embeddings(emb_path)
    agent_mats = compute_agent_matrices(embeddings, metadata)
    if not agent_mats:
        raise SystemExit("No agents found in embeddings; cannot plot similarity.")

    if args.runs:
        target_runs = [resolve_run(output_root, token) for token in args.runs]
    else:
        target_runs = run_dirs

    for run_dir in target_runs:
        if run_dir not in run_ids:
            print(f"Skipping {run_dir}: no history found")
            continue
        run_id = run_ids[run_dir]
        history_path = find_history(run_dir)
        if history_path is None:
            print(f"Skipping {run_dir}: missing history*.json")
            continue
        rounds, agents = parse_rounds(history_path)
        similarity_report = {
            "run_id": run_id,
            "timestamp": int(time.time()),
            "output_root": str(run_dir),
            "agents": [],
        }
        for agent in agents:
            similar = comparator.compare_agent_answers(
                agent_name=agent,
                round_num=0,
                run_id=run_id,
            )
            similarity_report["agents"].append(
                {"agent_name": agent, "results": similar}
            )

        existing = pick_existing_similarity(run_dir)
        if existing:
            sim_path = existing
        else:
            sim_path = run_dir / f"similarity_{run_id}.json"
        sim_path.write_text(json.dumps(similarity_report, indent=2), encoding="utf-8")
        print(f"Wrote {sim_path}")

        images_dir = run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        plot_matrices(agent_mats, images_dir / "similarity.png")


if __name__ == "__main__":
    main()
