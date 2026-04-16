import json, re
from pathlib import Path
from html.parser import HTMLParser
from html import unescape
import numpy as np

HTML_PATH = Path('cyber_negotiation_v1/games_descriptions/cyber_game/output/research_overview_by_category.html')

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
        elif tag in ('th','td') and self._in_row:
            self._in_cell = True
            self._cell_data = []
            if tag == 'th' or self._in_thead:
                self._row_is_header = True
        elif tag == 'br' and self._in_cell:
            self._cell_data.append('\n')
    def handle_data(self,data):
        if self._in_cell:
            self._cell_data.append(data)
    def handle_endtag(self, tag):
        if tag in ('th','td') and self._in_cell:
            text = unescape(''.join(self._cell_data)).strip()
            text = re.sub(r'\s+',' ',text)
            self._row.append(text)
            self._cell_data=[]
            self._in_cell=False
        elif tag=='tr' and self._in_row:
            if self._row:
                self._rows.append((self._row,self._row_is_header))
            self._row=[]
            self._in_row=False
        elif tag=='thead':
            self._in_thead=False
        elif tag=='table' and self._in_table:
            self.tables.append(self._rows)
            self._rows=[]
            self._in_table=False


def preprocess_html(text: str) -> str:
    text = re.sub(r"<table class='mini-stat-table'.*?Mean</th>\s*<td>(.*?)</td>.*?</table>", lambda m: m.group(1), text, flags=re.DOTALL)
    return text


def parse_tables():
    raw = HTML_PATH.read_text(encoding='utf-8')
    raw = preprocess_html(raw)
    parser = TableParser()
    parser.feed(raw)
    return parser.tables


def table_to_df(table_rows):
    header=None
    data_rows=[]
    for row,is_header in table_rows:
        if is_header and header is None:
            header=row
        else:
            data_rows.append(row)
    if header is None:
        return None
    data_rows=[r+['']*(len(header)-len(r)) for r in data_rows]
    return header,data_rows


def find_table(headers_required):
    for t in parse_tables():
        converted=table_to_df(t)
        if not converted:
            continue
        header,rows=converted
        if all(h in header for h in headers_required):
            return header,rows
    return None

run_table = find_table(['Scenario','Condition','Setup','Final Label','Final Severity','GT Label','GT Severity','Exact Correct','Type Correct','Severity Correct','Wrong Consensus','Severity Bias','Abs Severity Bias','Public Turns','Type Transitions','Severity Transitions','GPT-5 Trust Hygiene'])
run_header, run_rows = run_table
idx = {h: run_header.index(h) for h in run_header}

SEV_MAP = {'Low':1,'Medium':2,'High':3}

cond_under = {c: [] for c in ['C1','C2','C3','C4','C5','C6','C7']}

for r in run_rows:
    cond = r[idx['Condition']].strip()
    final = r[idx['Final Severity']].strip()
    gt = r[idx['GT Severity']].strip()
    if cond not in cond_under:
        continue
    fn = SEV_MAP.get(final)
    gn = SEV_MAP.get(gt)
    if fn is None or gn is None:
        continue
    cond_under[cond].append(1.0 if fn < gn else 0.0)

for c in cond_under:
    vals = cond_under[c]
    if vals:
        print(c, len(vals), sum(vals)/len(vals))
    else:
        print(c, 'no vals')
