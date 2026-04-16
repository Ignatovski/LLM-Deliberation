import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

ROOT = Path(__file__).resolve().parent

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def show_cybersecurity_sources():
    print("=== CYBERSECURITY DATA SOURCES ===\n")
    
    # 1. Load cyber records
    cyber_root = ROOT / "cyber_negotiation_v1" / "games_descriptions" / "cyber_game" / "output"
    
    # Find all metrics files
    metrics_files = list(cyber_root.rglob("metrics_*.json"))
    print(f"1. CYBERSECURITY RAW DATA:")
    print(f"   Root directory: {cyber_root}")
    print(f"   Total metrics files found: {len(metrics_files)}")
    
    # Show some example file paths
    for i, file in enumerate(metrics_files[:5]):
        print(f"     Example {i+1}: {file.relative_to(cyber_root)}")
    print("   ...\n")
    
    # 2. Load and process cyber data (same as in the plotting script)
    latest = {}
    for metrics_path in sorted(metrics_files):
        relative_parts = metrics_path.relative_to(cyber_root).parts
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
        
        key = (category, scenario_id, condition_id)
        current = latest.get(key)
        if current is None or metrics_path.stat().st_mtime > current["metrics_path"].stat().st_mtime:
            latest[key] = {
                "category": category,
                "scenario_id": scenario_id,
                "condition_id": condition_id,
                "run_id": run_id,
                "metrics_path": metrics_path,
                "report": report,
            }
    
    records = sorted(latest.values(), key=lambda r: (r["category"], r["scenario_id"], int(r["condition_id"][1:]) if len(r["condition_id"]) > 1 and r["condition_id"][1:].isdigit() else 999))
    
    print(f"2. PROCESSED CYBERSECURITY RECORDS:")
    print(f"   Total unique records: {len(records)}")
    
    # Group by condition
    by_cond = defaultdict(list)
    for record in records:
        by_cond[record["condition_id"]].append(record)
    
    print(f"   Records by condition:")
    for cond in sorted(by_cond.keys(), key=lambda x: int(x[1:]) if len(x) > 1 and x[1:].isdigit() else 999):
        print(f"     {cond}: {len(by_cond[cond])} records")
    print()
    
    # 3. Show the actual calculations for the three cybersecurity metrics
    print("3. CYBERSECURITY METRIC CALCULATIONS:")
    
    committee_conds = ["C3", "C4", "C5", "C6", "C7"]
    
    # Calculate wrong consensus rate
    wrong_rates = []
    for cond in committee_conds:
        rows = by_cond[cond]
        wrong_count = 0
        total_count = 0
        for row in rows:
            report = row["report"]
            headline = report.get("headline_metrics") or {}
            derived = report.get("derived_metrics") or {}
            if derived.get("WrongConsensusExact") is not None:
                wrong_count += float(derived.get("WrongConsensusExact"))
                total_count += 1
        rate = wrong_count / total_count if total_count > 0 else 0
        wrong_rates.append(rate)
        print(f"   {cond} - Wrong consensus: {wrong_count}/{total_count} = {rate:.3f} ({rate*100:.1f}%)")
    
    wrong_avg = mean(wrong_rates)
    print(f"   Average wrong consensus: {wrong_avg:.3f} ({wrong_avg*100:.1f}%)")
    print()
    
    # Calculate under-severity rate
    under_rates = []
    for cond in committee_conds:
        rows = by_cond[cond]
        under_count = 0
        total_count = 0
        for row in rows:
            report = row["report"]
            headline = report.get("headline_metrics") or {}
            if headline.get("SeverityBias") is not None:
                b = float(headline.get("SeverityBias"))
                if b < 0:
                    under_count += 1
                total_count += 1
        rate = under_count / total_count if total_count > 0 else 0
        under_rates.append(rate)
        print(f"   {cond} - Under-severity: {under_count}/{total_count} = {rate:.3f} ({rate*100:.1f}%)")
    
    under_avg = mean(under_rates)
    print(f"   Average under-severity: {under_avg:.3f} ({under_avg*100:.1f}%)")
    print()
    
    # Calculate cookie scenario exact failure rate
    cookie_records = [r for r in records if r["category"] == "cookies"]
    print(f"   Cookie scenario records: {len(cookie_records)}")
    
    exact_count = 0
    total_cookie = 0
    for record in cookie_records:
        report = record["report"]
        headline = report.get("headline_metrics") or {}
        if headline.get("FinalCorrectExact") is not None:
            exact_count += float(headline.get("FinalCorrectExact"))
            total_cookie += 1
    
    cookie_exact_rate = exact_count / total_cookie if total_cookie > 0 else 0
    cookie_failure_rate = 1.0 - cookie_exact_rate
    print(f"   Cookie exact success: {exact_count}/{total_cookie} = {cookie_exact_rate:.3f} ({cookie_exact_rate*100:.1f}%)")
    print(f"   Cookie exact failure: {cookie_failure_rate:.3f} ({cookie_failure_rate*100:.1f}%)")
    print()
    
    # 4. Show final values
    print("4. FINAL CYBERSECURITY VALUES FOR PLOT:")
    print(f"   Wrong consensus: {wrong_avg:.3f} ({wrong_avg*100:.1f}%)")
    print(f"   Under-severity: {under_avg:.3f} ({under_avg*100:.1f}%)")
    print(f"   Cookie exact failure: {cookie_failure_rate:.3f} ({cookie_failure_rate*100:.1f}%)")
    
    # 5. Show example data structure
    print("\n5. EXAMPLE DATA STRUCTURE:")
    if records:
        example = records[0]
        print(f"   Example record structure:")
        print(f"     category: {example['category']}")
        print(f"     scenario_id: {example['scenario_id']}")
        print(f"     condition_id: {example['condition_id']}")
        report = example['report']
        print(f"     headline_metrics keys: {list(report.get('headline_metrics', {}).keys())}")
        print(f"     derived_metrics keys: {list(report.get('derived_metrics', {}).keys())}")

if __name__ == "__main__":
    show_cybersecurity_sources()
