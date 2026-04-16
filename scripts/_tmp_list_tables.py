import pandas as pd
from pathlib import Path
html_path = Path('cyber_negotiation_v1/games_descriptions/cyber_game/output/research_overview_by_category.html')

tables = pd.read_html(html_path)
print('tables', len(tables))
for i,t in enumerate(tables[:20]):
    cols = [str(c) for c in t.columns]
    print(i, cols[:6], 'rows', len(t))
