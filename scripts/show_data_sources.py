import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def show_data_sources():
    print("=== DATA SOURCES FOR TRUST RISKS PLOT ===\n")
    
    # 1. BASELINE DATA (Polynomial)
    poly_path = ROOT / "viewer" / "metrics_summary.json"
    poly_data = load_json(poly_path)
    
    print("1. BASELINE (Polynomial) DATA:")
    print(f"   File: {poly_path}")
    print(f"   Total runs: {poly_data['count']}")
    print(f"   Generated from directories:")
    for source in poly_data['generated_from'][:3]:  # Show first 3
        print(f"     - {source}")
    print("   ...\n")
    
    # Filter for GPT-5 + Claude baseline
    baseline_runs = []
    for run in poly_data['runs']:
        model_mix = run.get('model_mix', '')
        if 'gpt-5' in model_mix or 'claude' in model_mix:
            baseline_runs.append(run)
    
    print(f"   GPT-5 + Claude runs: {len(baseline_runs)}")
    
    # Show some example baseline runs
    print("   Example baseline runs:")
    for i, run in enumerate(baseline_runs[:3]):
        print(f"     Run {i+1}: {run.get('model_mix')} | any_success={run.get('any_success')} | final_success={run.get('final_success')}")
    print()
    
    # 2. OBSTRUCTIVE DATA
    obstructive_path = ROOT / "summarys" / "metrics_summary.adversarial_obstructive.json"
    obstructive_data = load_json(obstructive_path)
    
    print("2. OBSTRUCTIVE (Adversarial) DATA:")
    print(f"   File: {obstructive_path}")
    print(f"   Total runs: {obstructive_data['count']}")
    print(f"   Generated from: {obstructive_data['generated_from']}")
    
    # Filter for GPT-5 + Claude obstructive
    obstructive_runs = []
    for run in obstructive_data['runs']:
        model_mix = run.get('model_mix', '')
        if 'gpt-5' in model_mix or 'claude' in model_mix:
            obstructive_runs.append(run)
    
    print(f"   GPT-5 + Claude runs: {len(obstructive_runs)}")
    
    # Show some example obstructive runs
    print("   Example obstructive runs:")
    for i, run in enumerate(obstructive_runs[:3]):
        print(f"     Run {i+1}: {run.get('model_mix')} | any_success={run.get('any_success')} | final_success={run.get('final_success')}")
    print()
    
    # 3. TARGETED DATA
    targeted_path = ROOT / "summarys" / "metrics_summary.adversarial_outcome_targeted.json"
    targeted_data = load_json(targeted_path)
    
    print("3. TARGETED (Adversarial) DATA:")
    print(f"   File: {targeted_path}")
    print(f"   Total runs: {targeted_data['count']}")
    print(f"   Generated from: {targeted_data['generated_from']}")
    
    # Filter for Claude targeted (for utility calculation)
    claude_targeted_runs = []
    for run in targeted_data['runs']:
        if 'claude' in run.get('model_mix', ''):
            claude_targeted_runs.append(run)
    
    print(f"   Claude runs: {len(claude_targeted_runs)}")
    
    # Show some example targeted runs with scores
    print("   Example Claude targeted runs with scores:")
    for i, run in enumerate(claude_targeted_runs[:3]):
        scores_outcome = run.get('scores_outcome', {})
        print(f"     Run {i+1}: {run.get('model_mix')} | scores_outcome={scores_outcome}")
    print()
    
    # 4. SHOW ACTUAL CALCULATIONS
    print("4. ACTUAL CALCULATIONS:")
    
    # Calculate baseline success rate
    baseline_success = sum(1 for r in baseline_runs if r.get('any_success'))
    baseline_rate = baseline_success / len(baseline_runs) if baseline_runs else 0
    print(f"   Baseline success: {baseline_success}/{len(baseline_runs)} = {baseline_rate:.3f} ({baseline_rate*100:.1f}%)")
    
    # Calculate obstructive success rate
    obstructive_success = sum(1 for r in obstructive_runs if r.get('any_success'))
    obstructive_rate = obstructive_success / len(obstructive_runs) if obstructive_runs else 0
    print(f"   Obstructive success: {obstructive_success}/{len(obstructive_runs)} = {obstructive_rate:.3f} ({obstructive_rate*100:.1f}%)")
    
    # Calculate drop
    drop = baseline_rate - obstructive_rate
    print(f"   Drop: {baseline_rate:.3f} - {obstructive_rate:.3f} = {drop:.3f} ({drop*100:.1f}%)")
    
    # Calculate Claude utility for targeted
    claude_baseline_runs = [r for r in baseline_runs if 'claude' in r.get('model_mix', '')]
    agents = ["Analyst A", "Builder B", "Critic C"]
    
    baseline_utilities = []
    targeted_utilities = []
    
    for agent in agents:
        # Baseline Claude utilities
        baseline_vals = [r.get('scores_outcome', {}).get(agent, 0) for r in claude_baseline_runs]
        baseline_util = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0
        baseline_utilities.append(baseline_util)
        
        # Targeted Claude utilities
        targeted_vals = [r.get('scores_outcome', {}).get(agent, 0) for r in claude_targeted_runs]
        targeted_util = sum(targeted_vals) / len(targeted_vals) if targeted_vals else 0
        targeted_utilities.append(targeted_util)
        
        print(f"   {agent} - Baseline: {baseline_util:.3f}, Targeted: {targeted_util:.3f}")
    
    avg_baseline = sum(baseline_utilities) / len(baseline_utilities)
    avg_targeted = sum(targeted_utilities) / len(targeted_utilities)
    utility_loss = (avg_baseline - avg_targeted) / avg_baseline if avg_baseline > 0 else 0
    
    print(f"   Average baseline utility: {avg_baseline:.3f}")
    print(f"   Average targeted utility: {avg_targeted:.3f}")
    print(f"   Utility loss: {utility_loss:.3f} ({utility_loss*100:.1f}%)")

if __name__ == "__main__":
    show_data_sources()
