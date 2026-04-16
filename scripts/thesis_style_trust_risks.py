from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "viewer" / "plots" / "thesis" / "cross-Evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLY_METRICS_PATH = ROOT / "viewer" / "metrics_summary.json"
ADV_OBSTRUCTIVE_PATH = ROOT / "summarys" / "metrics_summary.adversarial_obstructive.json"
ADV_TARGETED_PATH = ROOT / "summarys" / "metrics_summary.adversarial_outcome_targeted.json"
CYBER_ROOT = ROOT / "cyber_negotiation_v1" / "games_descriptions" / "cyber_game" / "output"

# Times New Roman font setup
TNR_PATH = Path(r"C:\Windows\Fonts\times.ttf")
if TNR_PATH.exists():
    fm.fontManager.addfont(str(TNR_PATH))

# THESIS STYLE CONFIGURATION
plt.rcParams.update(
    {
        "font.family": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.color": "#D0D0D0",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,
        "axes.grid.which": "major",
        "axes.grid.axis": "y",  # Only horizontal grid lines
        "ytick.major.size": 4,
        "ytick.minor.size": 2,
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
    }
)

# SEMANTIC COLOR SYSTEM
COLORS = {
    "GPT-5": "#0072B2",
    "Claude": "#D55E00",
    "Llama": "#009E73",
    "Mixed": "#CC79A7",
    "Neutral": "#4D4D4D",
}

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def rate(values: Iterable[Any]) -> float:
    vals = list(values)
    return sum(1 for v in vals if bool(v)) / len(vals) if vals else float("nan")

def mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [float(v) for v in values if v is not None]
    return mean(numeric) if numeric else None

def model_family(model_mix: str) -> str:
    if model_mix == "gpt-5x4":
        return "GPT-5"
    if model_mix == "claude-sonnet-4-5x4":
        return "Claude"
    if model_mix == "Llama-3.3-70B-Instructx4":
        return "Llama"
    return "Mixed"

def agent_outcome_mean(rows: Sequence[Dict[str, Any]], agent: str) -> Optional[float]:
    vals = []
    for row in rows:
        scores = row.get("scores_outcome") or row.get("scores") or {}
        if isinstance(scores, dict) and scores.get(agent) is not None:
            vals.append(float(scores[agent]))
    return mean(vals) if vals else None

def load_runs(path: Path) -> List[Dict[str, Any]]:
    runs = list(load_json(path).get("runs") or [])
    for row in runs:
        row["family"] = model_family(str(row.get("model_mix") or ""))
    return runs

def load_cyber_latest() -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for metrics_path in sorted(CYBER_ROOT.rglob("metrics_*.json")):
        relative_parts = metrics_path.relative_to(CYBER_ROOT).parts
        if not relative_parts or relative_parts[0] == "llm_evaluator":
            continue
        payload = load_json(metrics_path)
        report = dict(payload.get("run_report") or {})
        if not report:
            continue
        category = relative_parts[0]
        scenario_id = str(report.get("scenario_id") or "")
        condition_id = str(report.get("condition_id") or "")
        run_id = str(report.get("run_id") or metrics_path.stem.replace("metrics_", "", 1))
        if not category or not scenario_id or not condition_id:
            continue
        history_path = metrics_path.parent / f"{run_id}.json"
        if not history_path.exists():
            continue
        history = load_json(history_path)
        if history.get("run_status") != "completed":
            continue

        record = {
            "category": category,
            "scenario_id": scenario_id,
            "condition_id": condition_id,
            "run_id": run_id,
            "metrics_path": metrics_path,
            "report": report,
        }
        key = (category, scenario_id, condition_id)
        current = latest.get(key)
        if current is None or metrics_path.stat().st_mtime > current["metrics_path"].stat().st_mtime:
            latest[key] = record
    return sorted(latest.values(), key=lambda r: (r["category"], r["scenario_id"], int(r["condition_id"][1:]) if len(r["condition_id"]) > 1 and r["condition_id"][1:].isdigit() else 999))

def cyber_condition_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cond[record["condition_id"]].append(record)

    out: Dict[str, Dict[str, float]] = {}
    for cond, rows in by_cond.items():
        wrong: List[float] = []
        under: List[float] = []
        for row in rows:
            report = row["report"]
            headline = report.get("headline_metrics") or {}
            derived = report.get("derived_metrics") or {}
            if derived.get("WrongConsensusExact") is not None:
                wrong.append(float(derived.get("WrongConsensusExact")))
            if headline.get("SeverityBias") is not None:
                b = float(headline.get("SeverityBias"))
                under.append(1.0 if b < 0 else 0.0)
        out[cond] = {
            "wrong": mean(wrong),
            "under": mean(under),
        }
    return out

def make_thesis_style_trust_risks() -> None:
    # Load data
    poly_runs = load_runs(POLY_METRICS_PATH)
    obstructive_runs = load_runs(ADV_OBSTRUCTIVE_PATH)
    targeted_runs = load_runs(ADV_TARGETED_PATH)
    cyber_records = load_cyber_latest()
    cyber_stats = cyber_condition_stats(cyber_records)
    
    # Filter for exact model families
    baseline = [r for r in poly_runs if r["family"] in ("GPT-5", "Claude")]
    obstructive = [r for r in obstructive_runs if r["family"] in ("GPT-5", "Claude")]
    targeted = [r for r in targeted_runs if r["family"] in ("GPT-5", "Claude")]
    
    # Calculate metrics
    baseline_success_rate = rate(r.get("any_success") for r in baseline)
    obstructive_success_rate = rate(r.get("any_success") for r in obstructive)
    success_drop = baseline_success_rate - obstructive_success_rate
    
    # Claude utility calculation (agents A, B, C only)
    claude_baseline = [r for r in poly_runs if r["family"] == "Claude"]
    claude_targeted = [r for r in targeted if r["family"] == "Claude"]
    
    coop_agents = ["Analyst A", "Builder B", "Critic C"]
    baseline_agent_means = [agent_outcome_mean(claude_baseline, agent) for agent in coop_agents]
    targeted_agent_means = [agent_outcome_mean(claude_targeted, agent) for agent in coop_agents]
    
    baseline_coop_utility = mean(v for v in baseline_agent_means if v is not None)
    targeted_coop_utility = mean(v for v in targeted_agent_means if v is not None)
    utility_loss = (baseline_coop_utility - targeted_coop_utility) / baseline_coop_utility if baseline_coop_utility else 0
    
    # Cybersecurity metrics
    committee_conds = ["C3", "C4", "C5", "C6", "C7"]
    wrong_consensus = mean(cyber_stats[c]["wrong"] for c in committee_conds)
    under_severity = mean(cyber_stats[c]["under"] for c in committee_conds)
    
    # Create thesis-style figure
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=300, sharey=True)
    
    # Panel A: Polynomial trust risks
    ax = axes[0]
    
    # Data for polynomial
    poly_metrics = [success_drop, utility_loss]
    poly_labels = ["Success rate drop", "Utility loss"]
    
    # Create bars with neutral color
    bars = ax.bar(range(len(poly_metrics)), poly_metrics, 
                  color=COLORS["Neutral"], edgecolor="#333333", linewidth=0.6)
    
    # Labels and formatting
    ax.set_xticks(range(len(poly_metrics)))
    ax.set_xticklabels(poly_labels)
    ax.set_ylabel("Effect size")
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, poly_metrics)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    
    # Panel B: Cybersecurity trust risks
    ax = axes[1]
    
    # Data for cybersecurity
    cyber_metrics = [wrong_consensus, under_severity]
    cyber_labels = ["Wrong consensus", "Under-severity"]
    
    # Create bars with neutral color
    bars = ax.bar(range(len(cyber_metrics)), cyber_metrics,
                  color=COLORS["Neutral"], edgecolor="#333333", linewidth=0.6)
    
    # Labels and formatting
    ax.set_xticks(range(len(cyber_metrics)))
    ax.set_xticklabels(cyber_labels)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, cyber_metrics)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    
    # Panel labels (as per thesis style - no figure title, use panel labels)
    axes[0].text(0.02, 0.98, "A", transform=axes[0].transAxes, 
                 fontsize=12, fontweight="bold", va="top")
    axes[1].text(0.02, 0.98, "B", transform=axes[1].transAxes,
                 fontsize=12, fontweight="bold", va="top")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save in thesis style
    path = OUT_DIR / "thesis_style_trust_risks.png"
    pdf_path = OUT_DIR / "thesis_style_trust_risks.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    print(f"Wrote {path}")
    print(f"Wrote {pdf_path}")
    
    # Print data summary for methods section
    print(f"\nTHESIS STYLE TRUST RISKS SUMMARY:")
    print(f"Panel A - Polynomial:")
    print(f"  Success rate drop: {success_drop:.3f} (GPT-5 + Claude, baseline vs obstructive)")
    print(f"  Utility loss: {utility_loss:.3f} (Claude agents A,B,C, baseline vs targeted)")
    print(f"  Data: {len(baseline)} baseline, {len(obstructive)} obstructive, {len(claude_targeted)} targeted runs")
    print(f"Panel B - Cybersecurity:")
    print(f"  Wrong consensus: {wrong_consensus:.3f} (conditions C3-C7)")
    print(f"  Under-severity: {under_severity:.3f} (conditions C3-C7)")
    print(f"  Data: {len([r for r in cyber_records if r['condition_id'] in committee_conds])} runs")

if __name__ == "__main__":
    make_thesis_style_trust_risks()
