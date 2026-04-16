import re
from html.parser import HTMLParser
from html import unescape
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HTML_PATH = Path('cyber_negotiation_v1/games_descriptions/cyber_game/output/research_overview_by_category.html')
OUT_DIR = Path('viewer/plots/thesis/cybersecurity')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style (force Times New Roman from Windows fonts)
_TNR_PATH = r"C:\Windows\Fonts\times.ttf"
fm.fontManager.addfont(_TNR_PATH)
plt.rcParams.update({
    'font.family': ['Times New Roman'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#222222',
    'axes.labelcolor': '#111111',
    'xtick.color': '#111111',
    'ytick.color': '#111111',
})

COLORS = {
    'GPT-5': '#0072B2',
    'Claude': '#D55E00',
    'Llama': '#009E73',
    'Mixed': '#CC79A7',
    'Neutral': '#4D4D4D',
    'Warm': '#D55E00',
    'Cool': '#0072B2',
}

COND_INFO = {
    'C1': {'label': 'Single GPT-5', 'model': 'GPT-5', 'setup': 'single', 'prior': 'none'},
    'C2': {'label': 'Single Claude', 'model': 'Claude', 'setup': 'single', 'prior': 'none'},
    'C3': {'label': '3x GPT-5', 'model': 'GPT-5', 'setup': 'committee', 'prior': 'none'},
    'C4': {'label': '3x Claude', 'model': 'Claude', 'setup': 'committee', 'prior': 'none'},
    'C5': {'label': 'Mixed Committee', 'model': 'Mixed', 'setup': 'mixed', 'prior': 'none'},
    'C6': {'label': '3x GPT-5 + LLM Prior', 'model': 'GPT-5', 'setup': 'committee', 'prior': 'llm'},
    'C7': {'label': '3x Claude + Human Prior', 'model': 'Claude', 'setup': 'committee', 'prior': 'human'},
}

MARKERS = {
    'single': 'o',
    'committee': 's',
    'mixed': 'D',
}

LINESTYLES = {
    'none': '-',
    'human': '--',
    'llm': ':',
}

class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._cell_data = []
        self._row = []
        self._rows = []
        self._row_is_header = False
        self._in_thead = False

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            if self._in_table:
                # nested table should not happen after preprocessing
                return
            self._in_table = True
            self._rows = []
        elif tag == 'thead':
            self._in_thead = True
        elif tag == 'tr' and self._in_table:
            self._in_row = True
            self._row = []
            self._row_is_header = False
        elif tag in ('th', 'td') and self._in_row:
            self._in_cell = True
            self._cell_data = []
            if tag == 'th' or self._in_thead:
                self._row_is_header = True
        elif tag == 'br' and self._in_cell:
            self._cell_data.append('\n')

    def handle_data(self, data):
        if self._in_cell:
            self._cell_data.append(data)

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self._in_cell:
            text = unescape(''.join(self._cell_data)).strip()
            text = re.sub(r'\s+', ' ', text)
            self._row.append(text)
            self._cell_data = []
            self._in_cell = False
        elif tag == 'tr' and self._in_row:
            # include empty cells if row has headers or data
            if self._row:
                self._rows.append((self._row, self._row_is_header))
            self._row = []
            self._in_row = False
        elif tag == 'thead':
            self._in_thead = False
        elif tag == 'table' and self._in_table:
            self.tables.append(self._rows)
            self._rows = []
            self._in_table = False


def preprocess_html(text: str) -> str:
    # Replace mini-stat-table with its Mean value
    text = re.sub(
        r"<table class='mini-stat-table'.*?Mean</th>\s*<td>(.*?)</td>.*?</table>",
        lambda m: m.group(1),
        text,
        flags=re.DOTALL
    )
    return text


def parse_tables():
    raw = HTML_PATH.read_text(encoding='utf-8')
    raw = preprocess_html(raw)
    parser = TableParser()
    parser.feed(raw)
    return parser.tables


def table_to_df(table_rows):
    header = None
    data_rows = []
    for row, is_header in table_rows:
        if is_header and header is None:
            header = row
        else:
            data_rows.append(row)
    if header is None:
        return None
    # normalize row length
    data_rows = [r + [''] * (len(header) - len(r)) for r in data_rows]
    return header, data_rows


def find_tables(headers_required):
    tables = parse_tables()
    matches = []
    for t in tables:
        converted = table_to_df(t)
        if not converted:
            continue
        header, rows = converted
        if all(h in header for h in headers_required):
            matches.append((header, rows))
    return matches


def find_table(headers_required):
    matches = find_tables(headers_required)
    return matches[0] if matches else None


def parse_pct(s):
    if s is None:
        return np.nan
    s = str(s).strip().replace('%','')
    if s == '' or s.lower() == 'nan':
        return np.nan
    try:
        return float(s)/100.0
    except ValueError:
        return np.nan


def parse_float(s):
    if s is None:
        return np.nan
    s = str(s).strip().replace('+','')
    if s == '' or s.lower() == 'nan':
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# Extract condition stats
cond_table = find_table(['Condition', 'Exact Correct', 'Type Correct', 'Severity Correct', 'Wrong Consensus', 'Severity Bias', 'Type Transitions', 'Severity Transitions'])
if not cond_table:
    raise SystemExit('Condition table not found')

cond_header, cond_rows = cond_table

cond_idx = {name: cond_header.index(name) for name in cond_header}

cond_data = []
for r in cond_rows:
    cond_cell = r[cond_idx['Condition']]
    m = re.search(r'C\d', cond_cell)
    if not m:
        continue
    cond = m.group(0)
    info = COND_INFO[cond]
    cond_data.append({
        'cond': cond,
        'model': info['model'],
        'setup': info['setup'],
        'prior': info['prior'],
        'exact': parse_pct(r[cond_idx['Exact Correct']]),
        'type': parse_pct(r[cond_idx['Type Correct']]),
        'severity': parse_pct(r[cond_idx['Severity Correct']]),
        'wrong': parse_pct(r[cond_idx['Wrong Consensus']]),
        'severity_bias': parse_float(r[cond_idx['Severity Bias']]),
        'type_trans': parse_float(r[cond_idx['Type Transitions']]),
        'sev_trans': parse_float(r[cond_idx['Severity Transitions']]),
    })

# Scenario outcome table
scen_table = find_table(['Category', 'Scenario', 'Runs', 'Exact Correct', 'Type Correct', 'Severity Correct', 'Wrong Consensus'])
if not scen_table:
    raise SystemExit('Scenario table not found')

scen_header, scen_rows = scen_table
scen_idx = {name: scen_header.index(name) for name in scen_header}

scenario_rows = []
for r in scen_rows:
    scenario_rows.append({
        'category': r[scen_idx['Category']],
        'scenario': r[scen_idx['Scenario']],
        'exact': parse_pct(r[scen_idx['Exact Correct']]),
        'type': parse_pct(r[scen_idx['Type Correct']]),
        'severity': parse_pct(r[scen_idx['Severity Correct']]),
        'wrong': parse_pct(r[scen_idx['Wrong Consensus']]),
    })

# Run-level details tables (aggregate across all sections)
run_tables = find_tables(['Scenario', 'Condition', 'Final Severity', 'GT Severity', 'Exact Correct', 'Type Correct', 'Severity Correct', 'Wrong Consensus'])
if not run_tables:
    raise SystemExit('Run-level table not found')

# Normalize and combine rows
run_header = run_tables[0][0]
run_rows = []
for header, rows in run_tables:
    # map this table's columns to the main header
    idx_map = {name: header.index(name) for name in header}
    for r in rows:
        norm = [''] * len(run_header)
        for i, h in enumerate(run_header):
            if h in idx_map:
                norm[i] = r[idx_map[h]]
        run_rows.append(norm)

run_idx = {name: run_header.index(name) for name in run_header}

SEV_MAP = {'Low': 1, 'Medium': 2, 'High': 3}

run_data = []
for r in run_rows:
    cond = r[run_idx['Condition']].strip()
    scenario = r[run_idx['Scenario']].strip()
    final_sev = r[run_idx['Final Severity']].strip()
    gt_sev = r[run_idx['GT Severity']].strip()
    final_num = SEV_MAP.get(final_sev, None)
    gt_num = SEV_MAP.get(gt_sev, None)
    under = None
    over = None
    if final_num is not None and gt_num is not None:
        under = 1.0 if final_num < gt_num else 0.0
        over = 1.0 if final_num > gt_num else 0.0
    run_data.append({
        'condition': cond,
        'scenario': scenario,
        'exact': parse_pct(r[run_idx['Exact Correct']]),
        'type': parse_pct(r[run_idx['Type Correct']]),
        'severity': parse_pct(r[run_idx['Severity Correct']]),
        'wrong': parse_pct(r[run_idx['Wrong Consensus']]),
        'under': under,
        'over': over,
    })

# Sort conditions C1..C7
cond_data = sorted(cond_data, key=lambda d: int(d['cond'][1:]))

def mean_rate(items):
    vals = [v for v in items if v is not None and not np.isnan(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals))

# Attach under/over-severity rates per condition from run-level data
cond_under = {}
cond_over = {}
for c in COND_INFO.keys():
    rows = [r for r in run_data if r['condition'] == c]
    cond_under[c] = mean_rate([r['under'] for r in rows])
    cond_over[c] = mean_rate([r['over'] for r in rows])

for d in cond_data:
    d['under'] = cond_under.get(d['cond'], np.nan)
    d['over'] = cond_over.get(d['cond'], np.nan)

# Plot helpers

def setup_ax(ax, ylim=(0,1)):
    ax.set_ylim(*ylim)
    ax.grid(axis='y', color='#D0D0D0', linewidth=0.8)
    ax.set_axisbelow(True)

# 1) Final exact correctness by condition
fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=300)
xs = np.arange(len(cond_data))
vals = [d['exact'] for d in cond_data]
ax.bar(xs, vals, color=COLORS['Neutral'], edgecolor='#333333', linewidth=0.6)
ax.set_xticks(xs)
ax.set_xticklabels([d['cond'] for d in cond_data])
ax.set_ylabel('Final exact correctness')
setup_ax(ax, (0,1))
# value labels
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_exact_by_condition.png', dpi=300)
plt.close(fig)

# 2) Type vs severity by condition (grouped bars)
fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=300)
xs = np.arange(len(cond_data))
width = 0.35

type_vals = [d['type'] for d in cond_data]
sev_vals = [d['severity'] for d in cond_data]
colors = [COLORS[d['model']] for d in cond_data]

bars_type = ax.bar(xs - width/2, type_vals, width=width, color=colors, edgecolor='#333333', linewidth=0.6, label='Type')
bars_sev = ax.bar(xs + width/2, sev_vals, width=width, color=colors, edgecolor='#333333', linewidth=0.6, hatch='//', label='Severity')

ax.set_xticks(xs)
ax.set_xticklabels([d['cond'] for d in cond_data])
ax.set_ylabel('Correctness')
setup_ax(ax, (0,1))

# Legends: model colors + metric hatch
legend_models = [
    Patch(facecolor=COLORS['GPT-5'], edgecolor='#333333', label='GPT-5'),
    Patch(facecolor=COLORS['Claude'], edgecolor='#333333', label='Claude'),
    Patch(facecolor=COLORS['Mixed'], edgecolor='#333333', label='Mixed'),
]
legend_metrics = [
    Patch(facecolor='white', edgecolor='#333333', label='Type'),
    Patch(facecolor='white', edgecolor='#333333', hatch='//', label='Severity'),
]
leg1 = ax.legend(handles=legend_models, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
ax.add_artist(leg1)
ax.legend(handles=legend_metrics, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 0.60))

fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_type_vs_severity_by_condition.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 3) Single vs no-prior homogeneous committee (grouped bars, GPT-5 vs Claude panels)
metrics = ['exact', 'type', 'severity', 'wrong']
metric_labels = ['Exact', 'Type', 'Severity', 'Wrong']
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=300, sharey=True)

pairs = [
    ('GPT-5', 'C1', 'C3'),
    ('Claude', 'C2', 'C4'),
]

for ax, (model, c_single, c_comm) in zip(axes, pairs):
    d_single = next(d for d in cond_data if d['cond'] == c_single)
    d_comm = next(d for d in cond_data if d['cond'] == c_comm)
    color = COLORS[model]
    xs = np.arange(len(metrics))
    width = 0.35
    y_single = [d_single[m] for m in metrics]
    y_comm = [d_comm[m] for m in metrics]
    ax.bar(xs - width/2, y_single, width=width, color=color, edgecolor='#333333', linewidth=0.6, label='Single')
    ax.bar(xs + width/2, y_comm, width=width, color=color, edgecolor='#333333', linewidth=0.6, hatch='//', label='3-agent no-prior')
    ax.set_xticks(xs)
    ax.set_xticklabels(metric_labels)
    ax.set_title(model, fontsize=10, pad=4)
    setup_ax(ax, (0,1))

axes[0].set_ylabel('Rate')

legend_setup = [
    Patch(facecolor='white', edgecolor='#333333', label='Single'),
    Patch(facecolor='white', edgecolor='#333333', hatch='//', label='3-agent no-prior'),
]
axes[1].legend(handles=legend_setup, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))

fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_single_vs_committee.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 4) Severity bias by condition
fig, ax = plt.subplots(figsize=(6.6, 3.3), dpi=300)
xs = np.arange(len(cond_data))
vals = [d['severity_bias'] for d in cond_data]
colors = [COLORS['Warm'] if v > 0 else COLORS['Cool'] for v in vals]
ax.bar(xs, vals, color=colors, edgecolor='#333333', linewidth=0.6)
ax.axhline(0, color='#222222', linewidth=1.6)
ax.set_xticks(xs)
ax.set_xticklabels([d['cond'] for d in cond_data])
ax.set_ylabel('Severity bias')
ax.grid(axis='y', color='#D0D0D0', linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_severity_bias_by_condition.png', dpi=300)
plt.close(fig)

# 5) Type vs severity transitions by condition (grouped bars)
fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=300)
xs = np.arange(len(cond_data))
width = 0.35

type_vals = [d['type_trans'] for d in cond_data]
sev_vals = [d['sev_trans'] for d in cond_data]
colors = [COLORS[d['model']] for d in cond_data]

ax.bar(xs - width/2, type_vals, width=width, color=colors, edgecolor='#333333', linewidth=0.6, label='Type')
ax.bar(xs + width/2, sev_vals, width=width, color=colors, edgecolor='#333333', linewidth=0.6, hatch='//', label='Severity')

ax.set_xticks(xs)
ax.set_xticklabels([d['cond'] for d in cond_data])
ax.set_ylabel('Mean transitions')
ax.grid(axis='y', color='#D0D0D0', linewidth=0.8)
ax.set_axisbelow(True)

# value labels
for i, v in enumerate(type_vals):
    ax.text(i - width/2, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
for i, v in enumerate(sev_vals):
    ax.text(i + width/2, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

# legends
legend_models = [
    Patch(facecolor=COLORS['GPT-5'], edgecolor='#333333', label='GPT-5'),
    Patch(facecolor=COLORS['Claude'], edgecolor='#333333', label='Claude'),
    Patch(facecolor=COLORS['Mixed'], edgecolor='#333333', label='Mixed'),
]
legend_metrics = [
    Patch(facecolor='white', edgecolor='#333333', label='Type'),
    Patch(facecolor='white', edgecolor='#333333', hatch='//', label='Severity'),
]
leg1 = ax.legend(handles=legend_models, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
ax.add_artist(leg1)
ax.legend(handles=legend_metrics, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 0.60))

fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_type_vs_severity_transitions.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 6) Scenario heatmap (Exact, Type, Severity, Wrong)
metrics = ['exact', 'type', 'severity', 'wrong']
metric_labels = ['Exact', 'Type', 'Severity', 'Wrong']
scenarios = [f"{r['category']} | {r['scenario']}" for r in scenario_rows]
values = np.array([[r[m] for m in metrics] for r in scenario_rows], dtype=float)

fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.45*len(scenarios))), dpi=300)
img = ax.imshow(values, aspect='auto', cmap='cividis', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(metric_labels)))
ax.set_xticklabels(metric_labels)
ax.set_yticks(np.arange(len(scenarios)))
ax.set_yticklabels(scenarios, fontsize=8)

# annotate
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        ax.text(j, i, f"{values[i,j]*100:.1f}%", ha='center', va='center', color='white' if values[i,j] < 0.5 else 'black', fontsize=8)

cbar = fig.colorbar(img, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Rate', fontsize=8)
for tick in cbar.ax.yaxis.get_ticklabels():
    tick.set_fontsize(8)
fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_scenario_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 7) Mixed vs homogeneous committees (rates)
metrics = ['exact', 'severity', 'wrong', 'under']
metric_labels = ['Exact', 'Severity', 'Wrong', 'Under']
hom = {m: np.mean([d[m] for d in cond_data if d['cond'] in ('C3', 'C4')]) for m in metrics}
mix = {m: next(d[m] for d in cond_data if d['cond'] == 'C5') for m in metrics}

fig, ax = plt.subplots(figsize=(6.8, 3.3), dpi=300)
xs = np.arange(len(metrics))
width = 0.35
bars_h = ax.bar(xs - width/2, [hom[m] for m in metrics], width=width, color=COLORS['Neutral'], edgecolor='#333333', linewidth=0.6, label='Homogeneous (C3, C4)')
bars_m = ax.bar(xs + width/2, [mix[m] for m in metrics], width=width, color=COLORS['Mixed'], edgecolor='#333333', linewidth=0.6, label='Mixed (C5)')
ax.set_xticks(xs)
ax.set_xticklabels(metric_labels)
ax.set_ylabel('Rate')
ax.grid(axis='y', color='#D0D0D0', linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
# value labels
for b in list(bars_h) + list(bars_m):
    y = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, y + 0.02, f"{y*100:.1f}%", ha='center', va='bottom', fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_mixed_vs_homogeneous.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 7b) Mixed vs homogeneous committees (severity bias)
hom_bias = np.mean([d['severity_bias'] for d in cond_data if d['cond'] in ('C3', 'C4')])
mix_bias = next(d['severity_bias'] for d in cond_data if d['cond'] == 'C5')

fig, ax = plt.subplots(figsize=(4.2, 3.3), dpi=300)
xs = np.arange(2)
ax.bar(xs[0], hom_bias, color=COLORS['Neutral'], edgecolor='#333333', linewidth=0.6, label='Homogeneous (C3, C4)')
ax.bar(xs[1], mix_bias, color=COLORS['Mixed'], edgecolor='#333333', linewidth=0.6, label='Mixed (C5)')
ax.axhline(0, color='#222222', linewidth=1.6)
ax.set_xticks(xs)
ax.set_xticklabels(['Homogeneous (C3, C4)', 'Mixed (C5)'])
ax.set_ylabel('Severity bias')
ax.grid(axis='y', color='#D0D0D0', linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_mixed_vs_homogeneous_bias.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 7c) Mixed committee vs prior committees (rates)
metrics = ['exact', 'type', 'severity', 'wrong']
metric_labels = ['Exact', 'Type', 'Severity', 'Wrong']
prior_compare = [
    ('Mixed (C5)', 'C5', COLORS['Mixed'], ''),
    ('LLM prior (C6)', 'C6', COLORS['GPT-5'], '//'),
    ('Human prior (C7)', 'C7', COLORS['Claude'], '\\\\'),
]

fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=300)
xs = np.arange(len(metrics))
width = 0.24
offsets = np.linspace(-width, width, len(prior_compare))
bars_all = []
for offset, (label, cond, color, hatch) in zip(offsets, prior_compare):
    data = next(d for d in cond_data if d['cond'] == cond)
    bars = ax.bar(
        xs + offset,
        [data[m] for m in metrics],
        width=width,
        color=color,
        edgecolor='#333333',
        linewidth=0.6,
        hatch=hatch,
        label=label,
    )
    bars_all.extend(bars)

ax.set_xticks(xs)
ax.set_xticklabels(metric_labels)
ax.set_ylabel('Rate')
setup_ax(ax, (0,1))
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
for bar in bars_all:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + 0.02, f"{y*100:.1f}%", ha='center', va='bottom', fontsize=7)
fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_mixed_vs_priors.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 8) Agreement vs wrong consensus (baseline vs negotiation)
# Use condition-level aggregates for wrong consensus to stay consistent with other figures
baseline_wrong = np.mean([d['wrong'] for d in cond_data if d['cond'] in ('C1', 'C2')])
neg_wrong = np.mean([d['wrong'] for d in cond_data if d['cond'] in ('C3', 'C4', 'C5', 'C6', 'C7')])
baseline_agree = 1.0
neg_agree = 1.0

fig, ax = plt.subplots(figsize=(5.8, 3.3), dpi=300)
xs = np.arange(2)
ax.scatter(xs - 0.08, [baseline_agree, baseline_wrong], color=COLORS['Neutral'], marker=MARKERS['single'], s=48, edgecolor='#222222', linewidth=0.4, label='Baseline')
ax.scatter(xs + 0.08, [neg_agree, neg_wrong], color=COLORS['Neutral'], marker=MARKERS['committee'], s=48, edgecolor='#222222', linewidth=0.4, label='Negotiation')
ax.set_xticks(xs)
ax.set_xticklabels(['Agreement', 'Wrong consensus'])
ax.set_ylabel('Rate')
setup_ax(ax, (0,1))
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_agreement_vs_wrong_consensus.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 9) Prior effects (rates) — C3 vs C6, C4 vs C7
metrics = ['exact', 'type', 'severity', 'wrong']
metric_labels = ['Exact', 'Type', 'Severity', 'Wrong']
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=300, sharey=True)

prior_pairs = [
    ('GPT-5', 'C3', 'C6', 'LLM prior'),
    ('Claude', 'C4', 'C7', 'Human prior'),
]

for ax, (model, c_base, c_prior, prior_label) in zip(axes, prior_pairs):
    d_base = next(d for d in cond_data if d['cond'] == c_base)
    d_prior = next(d for d in cond_data if d['cond'] == c_prior)
    color = COLORS[model]
    xs = np.arange(len(metrics))
    width = 0.35
    y_base = [d_base[m] for m in metrics]
    y_prior = [d_prior[m] for m in metrics]
    ax.bar(xs - width/2, y_base, width=width, color=color, edgecolor='#333333', linewidth=0.6, label='No prior')
    ax.bar(xs + width/2, y_prior, width=width, color=color, edgecolor='#333333', linewidth=0.6, hatch='//', label=prior_label)
    ax.set_xticks(xs)
    ax.set_xticklabels(metric_labels)
    ax.set_title(f"{model} ({prior_label})", fontsize=10, pad=4)
    setup_ax(ax, (0,1))

axes[0].set_ylabel('Rate')
legend_setup = [
    Patch(facecolor='white', edgecolor='#333333', label='No prior'),
    Patch(facecolor='white', edgecolor='#333333', hatch='//', label='With prior'),
]
axes[1].legend(handles=legend_setup, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))

fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_prior_effects.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 9b) Prior effects (severity bias)
fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.3), dpi=300, sharey=True)
all_bias_vals = []
for ax, (model, c_base, c_prior, prior_label) in zip(axes, prior_pairs):
    d_base = next(d for d in cond_data if d['cond'] == c_base)
    d_prior = next(d for d in cond_data if d['cond'] == c_prior)
    color = COLORS[model]
    xs = np.arange(2)
    bars = ax.bar(xs, [d_base['severity_bias'], d_prior['severity_bias']], color=color, edgecolor='#333333', linewidth=0.6)
    bars[1].set_hatch('//')
    ax.axhline(0, color='#222222', linewidth=1.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(['No prior', 'With prior'])
    ax.set_title(f"{model} ({prior_label})", fontsize=10, pad=4)
    ax.grid(axis='y', color='#D0D0D0', linewidth=0.8)
    ax.set_axisbelow(True)
    # value labels
    for b in bars:
        y = b.get_height()
        offset = 0.05 if y >= 0 else -0.08
        ax.text(b.get_x() + b.get_width()/2, y + offset, f"{y:.2f}", ha='center', va='bottom' if y >= 0 else 'top', fontsize=8)
    all_bias_vals.extend([d_base['severity_bias'], d_prior['severity_bias']])

axes[0].set_ylabel('Severity bias')
min_v = min(all_bias_vals)
max_v = max(all_bias_vals)
pad = 0.2
for ax in axes:
    ax.set_ylim(min_v - pad, max_v + pad)

fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_prior_effects_bias.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 10) Risk summary by scenario removed from main set

# 11) Cookie sub-scenario breakdown heatmap
cookie_rows = [r for r in scenario_rows if r['category'] == 'Cookies']
cookie_scenarios = [r['scenario'] for r in cookie_rows]
cookie_metrics = ['exact', 'type', 'severity', 'wrong']
cookie_labels = ['Exact', 'Type', 'Severity', 'Wrong']
cookie_vals = np.array([[r[m] for m in cookie_metrics] for r in cookie_rows], dtype=float)
fig, ax = plt.subplots(figsize=(6.6, 2.6), dpi=300)
img = ax.imshow(cookie_vals, aspect='auto', cmap='cividis', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(cookie_labels)))
ax.set_xticklabels(cookie_labels)
ax.set_yticks(np.arange(len(cookie_scenarios)))
ax.set_yticklabels(cookie_scenarios)
for i in range(cookie_vals.shape[0]):
    for j in range(cookie_vals.shape[1]):
        ax.text(j, i, f"{cookie_vals[i,j]*100:.1f}%", ha='center', va='center', color='white' if cookie_vals[i,j] < 0.5 else 'black', fontsize=8)
cbar = fig.colorbar(img, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Rate')
fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_cookie_breakdown.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 12) Ceiling-case vs ambiguity-case comparison (grouped bars)
ceiling = ['CSRF', 'Command Injection', 'Info Findings']
ambiguity = ['Cookies', 'Path Disclosure']

def group_mean(categories):
    rows = [r for r in scenario_rows if r['category'] in categories]
    return {
        'exact': mean_rate([r['exact'] for r in rows]),
        'type': mean_rate([r['type'] for r in rows]),
        'severity': mean_rate([r['severity'] for r in rows]),
        'wrong': mean_rate([r['wrong'] for r in rows]),
    }

ceil_vals = group_mean(ceiling)
amb_vals = group_mean(ambiguity)

metrics2 = ['exact', 'type', 'severity', 'wrong']
metric_labels2 = ['Exact', 'Type', 'Severity', 'Wrong']
fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=300)
xs = np.arange(len(metrics2))
width = 0.35
ax.bar(xs - width/2, [ceil_vals[m] for m in metrics2], width=width, color=COLORS['Neutral'], edgecolor='#333333', linewidth=0.6, label='Ceiling')
ax.bar(xs + width/2, [amb_vals[m] for m in metrics2], width=width, color='#999999', edgecolor='#333333', linewidth=0.6, hatch='//', label='Ambiguity')
ax.set_xticks(xs)
ax.set_xticklabels(metric_labels2)
ax.set_ylabel('Rate')
setup_ax(ax, (0,1))
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()
fig.savefig(OUT_DIR/'cyber_ceiling_vs_ambiguity.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print('Wrote plots to', OUT_DIR)
