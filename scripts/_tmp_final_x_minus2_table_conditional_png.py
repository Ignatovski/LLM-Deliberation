import csv
from pathlib import Path
import matplotlib.pyplot as plt

csv_path = Path('viewer/plots/thesis/adversarial_final_x_minus2_table_conditional.csv')
rows = list(csv.DictReader(csv_path.open()))

# Keep a focused view: condition, model, conditional count and percent
columns = ['Condition', 'Model', 'Final x = -2 | Any success (n)', 'Any success (n)', 'Final x = -2 | Any success (%)']
cell_text = [[r[c] for c in columns] for r in rows]

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=300)
ax.axis('off')

_tbl = ax.table(
    cellText=cell_text,
    colLabels=columns,
    cellLoc='center',
    loc='center'
)

for (row, col), cell in _tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#e9eef6')
        cell.set_text_props(weight='bold')
    cell.set_edgecolor('#c9ced6')

for i, r in enumerate(rows, start=1):
    if r['Condition'] == 'Outcome targeted':
        for c in range(len(columns)):
            _tbl[(i, c)].set_facecolor('#fff2cc')
        pct_col = columns.index('Final x = -2 | Any success (%)')
        _tbl[(i, pct_col)].set_text_props(weight='bold', color='#7a4b00')

ax.set_title('Final x = -2 among any-success runs', pad=10)
_tbl.scale(1.0, 1.3)

out_path = Path('viewer/plots/thesis/adversarial_final_x_minus2_table_conditional.png')
fig.tight_layout()
fig.savefig(out_path, bbox_inches='tight')
print('Wrote', out_path)
