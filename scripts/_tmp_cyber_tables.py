import re
from html.parser import HTMLParser
from html import unescape
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from thesis_plot_style import apply_thesis_style

HTML_PATH = Path('cyber_negotiation_v1/games_descriptions/cyber_game/output/research_overview_by_category.html')
OUT_DIR = Path('viewer/plots/thesis/cybersecurity')
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_thesis_style(font_size=9, y_grid=False)

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


def parse_tables(text: str | None = None):
    raw = HTML_PATH.read_text(encoding='utf-8') if text is None else text
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
    data_rows = [r + [''] * (len(header) - len(r)) for r in data_rows]
    return header, data_rows


def find_table(headers_required, text: str | None = None):
    tables = parse_tables(text)
    for t in tables:
        converted = table_to_df(t)
        if not converted:
            continue
        header, rows = converted
        if all(h in header for h in headers_required):
            return header, rows
    return None


def render_table(headers, rows, out_base, col_widths=None, row_height=0.35, header_height=0.45):
    nrows = len(rows) + 1
    ncols = len(headers)

    if col_widths is None:
        col_widths = [1] * ncols

    fig_w = sum(col_widths) * 0.55
    fig_h = nrows * row_height + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis('off')

    cell_text = [headers] + rows
    tbl = ax.table(cellText=cell_text, cellLoc='center', loc='center')

    # style
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#D0D0D0')
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor('#F2F5F8')
            cell.set_text_props(weight='bold')
            cell.set_height(header_height / nrows)
        else:
            cell.set_facecolor('white')
            cell.set_height(row_height / nrows)

    # set column widths
    total = sum(col_widths)
    for c in range(ncols):
        w = col_widths[c] / total
        for r in range(nrows):
            tbl[(r, c)].set_width(w)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_base}.png", bbox_inches='tight')
    fig.savefig(OUT_DIR / f"{out_base}.pdf", bbox_inches='tight')
    plt.close(fig)

# Condition statistics table
cond_table = find_table(['Condition','Exact Correct','Type Correct','Severity Correct','Wrong Consensus','Severity Bias','Abs Severity Bias','Public Turns','Type Transitions','Severity Transitions'])
if not cond_table:
    raise SystemExit('Condition table not found')
cond_header, cond_rows = cond_table

# Clean condition cell
cond_idx = cond_header.index('Condition')
for r in cond_rows:
    r[cond_idx] = r[cond_idx].replace('Single ', '').replace('3x ', '3x ').replace('\n', ' ').strip()

render_table(
    headers=cond_header,
    rows=cond_rows,
    out_base='cyber_condition_statistics_table',
    col_widths=[1.0,1.0,1.0,1.0,1.0,0.8,0.9,0.8,0.9,0.9]
)

# Run-level table from the path_disclosure section
html_raw = HTML_PATH.read_text(encoding='utf-8')
section_match = re.search(r"<section class='category-section' id='path_disclosure'>.*?</section>", html_raw, flags=re.DOTALL)
section_html = section_match.group(0) if section_match else None
if section_html is None:
    raise SystemExit('path_disclosure section not found')

run_table = find_table(['Scenario','Condition','Setup','Final Label','Final Severity','GT Label','GT Severity','Exact Correct','Type Correct','Severity Correct','Wrong Consensus','Severity Bias','Abs Severity Bias','Public Turns','Type Transitions','Severity Transitions','GPT-5 Trust Hygiene'], text=section_html)
if not run_table:
    raise SystemExit('Run-level table not found')
run_header, run_rows = run_table

# Drop Scenario column (same for all), and drop Report/History if present
keep_cols = [c for c in run_header if c not in ('Scenario','Report','History')]
keep_idx = [run_header.index(c) for c in keep_cols]
filtered_rows = [[r[i] for i in keep_idx] for r in run_rows]

render_table(
    headers=keep_cols,
    rows=filtered_rows,
    out_base='cyber_run_level_path_disclosure_table',
    col_widths=[0.6,1.0,1.0,0.9,1.0,0.9,0.9,0.9,0.9,0.9,0.7,0.9,0.9,0.9,0.9,0.9]
)

print('Wrote table images to', OUT_DIR)
