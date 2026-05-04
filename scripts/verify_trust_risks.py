import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
POLY_METRICS_PATH = ROOT / "viewer" / "metrics_summary.json"
ADV_OBSTRUCTIVE_PATH = ROOT / "summarys" / "metrics_summary.adversarial_obstructive.json"
ADV_TARGETED_PATH = ROOT / "summarys" / "metrics_summary.adversarial_outcome_targeted.json"

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def rate(values: Sequence[Any]) -> float:
    vals = list(values)
    return sum(1 for v in vals if bool(v)) / len(vals) if vals else float("nan")

def mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
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

def verify_calculations():
    print("=== VERIFYING TRUST RISKS CALCULATIONS ===\n")
    
    # Load data
    poly_runs = load_runs(POLY_METRICS_PATH)
    obstructive_runs = load_runs(ADV_OBSTRUCTIVE_PATH)
    targeted_runs = load_runs(ADV_TARGETED_PATH)
    
    print(f"Polynomial runs total: {len(poly_runs)}")
    print(f"Obstructive runs total: {len(obstructive_runs)}")
    print(f"Targeted runs total: {len(targeted_runs)}\n")
    
    # Filter for GPT-5 and Claude only (as in the code)
    baseline = [r for r in poly_runs if r["family"] in ("GPT-5", "Claude")]
    obstructive = [r for r in obstructive_runs if r["family"] in ("GPT-5", "Claude")]
    targeted = [r for r in targeted_runs if r["family"] in ("GPT-5", "Claude")]
    
    print(f"Baseline runs (GPT-5 + Claude): {len(baseline)}")
    print(f"Obstructive runs (GPT-5 + Claude): {len(obstructive)}")
    print(f"Targeted runs (GPT-5 + Claude): {len(targeted)}\n")
    
    # Calculate obstructive success drop
    baseline_any = rate(r.get("any_success") for r in baseline)
    obstructive_any = rate(r.get("any_success") for r in obstructive)
    obstructive_drop = baseline_any - obstructive_any
    
    print("=== OBSTRUCTIVE SUCCESS DROP ===")
    print(f"Baseline any-success rate: {baseline_any:.4f} ({baseline_any*100:.2f}%)")
    print(f"Obstructive any-success rate: {obstructive_any:.4f} ({obstructive_any*100:.2f}%)")
    print(f"Drop: {obstructive_drop:.4f} ({obstructive_drop*100:.2f}%)")
    
    # Calculate targeted utility loss
    coop_agents = ["Analyst A", "Builder B", "Critic C"]
    claude_baseline = [r for r in poly_runs if r["family"] == "Claude"]
    claude_targeted = [r for r in targeted if r["family"] == "Claude"]
    
    print(f"\n=== TARGETED UTILITY LOSS ===")
    print(f"Claude baseline runs: {len(claude_baseline)}")
    print(f"Claude targeted runs: {len(claude_targeted)}")
    
    baseline_agent_means = [agent_outcome_mean(claude_baseline, agent) for agent in coop_agents]
    targeted_agent_means = [agent_outcome_mean(claude_targeted, agent) for agent in coop_agents]
    
    print(f"Baseline agent means: {[f'{v:.3f}' if v is not None else 'None' for v in baseline_agent_means]}")
    print(f"Targeted agent means: {[f'{v:.3f}' if v is not None else 'None' for v in targeted_agent_means]}")
    
    baseline_coop_utility = mean(v for v in baseline_agent_means if v is not None)
    targeted_coop_utility = mean(v for v in targeted_agent_means if v is not None)
    
    print(f"Baseline cooperative utility: {baseline_coop_utility:.4f}")
    print(f"Targeted cooperative utility: {targeted_coop_utility:.4f}")
    
    if baseline_coop_utility and baseline_coop_utility > 0:
        targeted_utility_loss = (baseline_coop_utility - targeted_coop_utility) / baseline_coop_utility
        print(f"Utility loss: {targeted_utility_loss:.4f} ({targeted_utility_loss*100:.2f}%)")
    else:
        print("Cannot calculate utility loss - baseline utility is zero or None")
    
    print(f"\n=== FINAL VALUES FOR PLOT ===")
    print(f"Obstructive drop: {obstructive_drop:.4f} ({obstructive_drop*100:.2f}%)")
    if baseline_coop_utility and baseline_coop_utility > 0:
        targeted_utility_loss = (baseline_coop_utility - targeted_coop_utility) / baseline_coop_utility
        print(f"Targeted utility loss: {targeted_utility_loss:.4f} ({targeted_utility_loss*100:.2f}%)")

if __name__ == "__main__":
    verify_calculations()
