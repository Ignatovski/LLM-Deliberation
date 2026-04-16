import csv, json
from pathlib import Path

FILES = {
    'Baseline': Path('summarys/metrics_summary.generated.json'),
    'Obstructive': Path('summarys/metrics_summary.adversarial_obstructive.json'),
    'Outcome targeted': Path('summarys/metrics_summary.adversarial_outcome_targeted.json'),
}

MODELS = {
    'GPT-5': 'gpt-5x4',
    'Claude': 'claude-sonnet-4-5x4',
}

rows = []
for condition, path in FILES.items():
    data = json.loads(path.read_text())
    runs = data['runs']
    for model_label, mix in MODELS.items():
        sel = [r for r in runs if r.get('model_mix') == mix]
        any_sel = [r for r in sel if r.get('any_success')]
        total = len(sel)
        any_total = len(any_sel)
        count_all = sum(1 for r in sel if r.get('final_x') == -2)
        count_any = sum(1 for r in any_sel if r.get('final_x') == -2)
        rows.append({
            'Condition': condition,
            'Model': model_label,
            'Final x = -2 (n)': count_all,
            'Total runs (n)': total,
            'Final x = -2 (%)': f"{(count_all/total*100):.1f}%" if total else '0.0%',
            'Any success (n)': any_total,
            'Final x = -2 | Any success (n)': count_any,
            'Final x = -2 | Any success (%)': f"{(count_any/any_total*100):.1f}%" if any_total else '0.0%'
        })

out_path = Path('viewer/plots/thesis/adversarial_final_x_minus2_table_conditional.csv')
with out_path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print('Wrote', out_path)
