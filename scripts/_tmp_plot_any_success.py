import json
from pathlib import Path
import matplotlib.pyplot as plt

from thesis_plot_style import apply_thesis_style

BASELINE = Path('summarys/metrics_summary.generated.json')
OBSTRUCT = Path('summarys/metrics_summary.adversarial_obstructive.json')

MODELS = {
    'GPT-5': 'gpt-5x4',
    'Claude': 'claude-sonnet-4-5x4',
}


def load_any_success_rate(path, model_mix):
    data = json.loads(path.read_text())
    runs = data['runs']
    sel = [r for r in runs if r.get('model_mix') == model_mix]
    if not sel:
        raise RuntimeError(f'No runs for {model_mix} in {path}')
    rate = sum(1 for r in sel if r.get('any_success')) / len(sel)
    return rate, len(sel)

x_labels = ['Baseline', 'Obstructive']

series = {}
counts = {}
for label, mix in MODELS.items():
    base_rate, base_n = load_any_success_rate(BASELINE, mix)
    obs_rate, obs_n = load_any_success_rate(OBSTRUCT, mix)
    series[label] = [base_rate, obs_rate]
    counts[label] = [base_n, obs_n]

# Plot
apply_thesis_style(font_size=12, y_grid=True)
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
xs = list(range(len(x_labels)))

for label, ys in series.items():
    ax.plot(xs, ys, marker='s', linewidth=2, markersize=7, label=label)

ax.set_xticks(xs)
ax.set_xticklabels(x_labels)
ax.set_ylabel('Any success rate')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', color='#d9d9e3', linewidth=1, alpha=0.8)

ax.legend(title='Model', frameon=False, loc='upper right')

out_path = Path('viewer/plots/thesis/adversarial_any_success_rates_baseline_obstructive.png')
fig.tight_layout()
fig.savefig(out_path)
print('Wrote', out_path)
print('Counts', counts)
