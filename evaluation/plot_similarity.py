"""
Plot cross-run similarity of agent public answers using the stored FAISS embeddings/metadata.

Usage:
    python evaluation/plot_similarity.py --embeddings output/cooperative/faiss_index/embeddings.npy --save similarity.png

It loads the JSON-style embeddings file written by AnswerComparator, groups answers by agent and run_id,
and computes average cosine similarity for matching rounds across runs. The output is a heatmap per agent
showing how similar their answers are between runs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_embeddings(path: Path) -> Tuple[np.ndarray, List[Dict]]:
    with path.open("r") as f:
        data = json.load(f)
    embeddings = np.array(data.get("embeddings", []), dtype="float32")
    metadata = data.get("metadata", [])
    return embeddings, metadata


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def drop_runs(
    embeddings: np.ndarray, metadata: List[Dict], exclude_runs: List[str]
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Remove entries that belong to run_ids in exclude_runs to avoid plotting broken runs.
    """
    if not exclude_runs:
        return embeddings, metadata
    keep_mask = [meta.get("run_id") not in exclude_runs for meta in metadata]
    if not any(keep_mask):
        return np.empty((0, embeddings.shape[1] if embeddings.size else 0), dtype="float32"), []
    filtered_embeddings = embeddings[np.array(keep_mask, dtype=bool)]
    filtered_metadata = [m for m, keep in zip(metadata, keep_mask) if keep]
    removed = len(metadata) - len(filtered_metadata)
    print(f"Filtered out {removed} entries from runs: {', '.join(exclude_runs)}")
    return filtered_embeddings, filtered_metadata


def compute_agent_matrices(
    embeddings: np.ndarray, metadata: List[Dict]
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """
    For each agent, build a run_id x run_id matrix of average cosine similarity across matching rounds.
    """
    agents_runs: Dict[str, Dict[str, Dict[int, int]]] = {}
    for idx, meta in enumerate(metadata):
        agent = meta["agent_name"]
        run = meta["run_id"]
        round_num = meta.get("round_num", -1)
        agents_runs.setdefault(agent, {}).setdefault(run, {})[round_num] = idx

    matrices: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for agent, runs in agents_runs.items():
        run_ids = sorted(runs.keys())
        n = len(run_ids)
        mat = np.zeros((n, n), dtype=np.float32)
        for i, run_i in enumerate(run_ids):
            for j, run_j in enumerate(run_ids):
                if i == j:
                    mat[i, j] = 1.0
                    continue
                # average over matching round numbers
                rounds = set(runs[run_i].keys()) & set(runs[run_j].keys())
                if not rounds:
                    mat[i, j] = 0.0
                    continue
                sims = []
                for r in rounds:
                    emb_i = embeddings[runs[run_i][r]]
                    emb_j = embeddings[runs[run_j][r]]
                    sims.append(cosine(emb_i, emb_j))
                mat[i, j] = float(np.mean(sims)) if sims else 0.0
        matrices[agent] = (run_ids, mat)
    return matrices


def plot_matrices(agent_mats: Dict[str, Tuple[List[str], np.ndarray]], save: Path | None):
    # Build a consistent friendly label per run across all agents (Run 1, Run 2, ...)
    all_run_ids = []
    for _, (runs, _) in agent_mats.items():
        all_run_ids.extend(runs)
    friendly = {rid: f"Run {idx+1}" for idx, rid in enumerate(sorted(set(all_run_ids)))}

    num_agents = len(agent_mats)
    fig, axes = plt.subplots(
        num_agents, 1, figsize=(10, 3 * num_agents), constrained_layout=True
    )
    if num_agents == 1:
        axes = [axes]
    for ax, (agent, (run_ids, mat)) in zip(axes, agent_mats.items()):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{agent} similarity (avg cosine by round)")
        ax.set_xticks(range(len(run_ids)))
        ax.set_yticks(range(len(run_ids)))
        ax.set_xticklabels([friendly[r] for r in run_ids], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([friendly[r] for r in run_ids], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200)
        print(f"Saved similarity heatmaps to {save}")
    else:
        plt.show()


def plot_per_agent(agent_mats: Dict[str, Tuple[List[str], np.ndarray]], save_dir: Path):
    """
    Save one clean heatmap per agent to keep each role separate.
    """
    all_run_ids = []
    for _, (runs, _) in agent_mats.items():
        all_run_ids.extend(runs)
    friendly = {rid: f"Run {idx+1}" for idx, rid in enumerate(sorted(set(all_run_ids)))}

    save_dir.mkdir(parents=True, exist_ok=True)
    for agent, (run_ids, mat) in agent_mats.items():
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{agent} similarity")
        ax.set_xticks(range(len(run_ids)))
        ax.set_yticks(range(len(run_ids)))
        ax.set_xticklabels([friendly[r] for r in run_ids], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([friendly[r] for r in run_ids], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out_path = save_dir / f"{agent.replace(' ', '_').lower()}_similarity.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot cross-run answer similarity per agent.")
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to embeddings file (e.g., output/.../faiss_index/embeddings.npy JSON).",
    )
    parser.add_argument("--save", default=None, help="Optional path to save a combined PNG; show interactively if omitted.")
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Optional directory to save one heatmap per agent (e.g., output/cooperative/plots).",
    )
    parser.add_argument(
        "--exclude_runs",
        nargs="*",
        default=[],
        help="List of run_id values to drop before plotting (useful to ignore failed runs).",
    )
    args = parser.parse_args()

    emb_path = Path(args.embeddings).expanduser().resolve()
    embeddings, metadata = load_embeddings(emb_path)
    embeddings, metadata = drop_runs(embeddings, metadata, args.exclude_runs)
    agent_mats = compute_agent_matrices(embeddings, metadata)
    if not agent_mats:
        raise SystemExit("No agents found in embeddings file.")
    if args.save_dir:
        plot_per_agent(agent_mats, Path(args.save_dir).expanduser().resolve())
    else:
        save_path = Path(args.save).expanduser().resolve() if args.save else None
        plot_matrices(agent_mats, save_path)


if __name__ == "__main__":
    main()
