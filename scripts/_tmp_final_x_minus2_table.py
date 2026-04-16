import csv, json
from pathlib import Path

FILES = {
    'Baseline': Path('summarys/metrics_summary.generated.json'),
    'Obstructive': Path('summarys/metrics_summary.adversarial_obstructive.json'),
    'Outcome targeted': Path('summarys/metrics_summary.adversarial_outcome_targeted.json'),
}

# model mixes to keep consistent with other tables
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
        total = len(sel)
        count = sum(1 for r in sel if r.get('final_x') == -2)
        rows.append({
            'Condition': condition,
            'Model': model_label,
            'Final x = -2 (n)': count,
            'Total runs (n)': total,
            'Final x = -2 (%)': f"{(count/total*100):.1f}%" if total else '0.0%'
        })

out_path = Path('viewer/plots/thesis/adversarial_final_x_minus2_table.csv')
with out_path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print('Wrote', out_path)
