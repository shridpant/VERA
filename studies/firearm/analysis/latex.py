"""CSV and LaTeX (booktabs) writers for the tables, and the number macros the report cites."""
import csv
import json
import math
from pathlib import Path

ESCAPE = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
          '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def esc(text):
    return ''.join(ESCAPE.get(ch, ch) for ch in str(text))


def fmt(value, column=''):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '--'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        if 'pct' in column or '_pp' in column:
            return f'{value:.1f}'
        if any(k in column for k in ('hours', 'gib', 'duration', '_s')):
            return f'{value:.2f}'
        return f'{value:.3f}'
    return esc(value)


def write_csv(t, directory):
    with (Path(directory) / f"{t['id']}.csv").open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=t['columns'], extrasaction='ignore')
        writer.writeheader()
        for row in t['rows']:
            writer.writerow({k: row.get(k) for k in t['columns']})


def write_tex(t, directory, columns=None, longtable=False, size=r'\scriptsize'):
    columns = columns or t['columns']
    head = ' & '.join(esc(c).replace(r'\_', r'\_\allowbreak{}') for c in columns) + r' \\'
    body = '\n'.join(' & '.join(fmt(r.get(c), c) for c in columns) + r' \\' for r in t['rows'])
    notes = ' '.join(esc(n) for n in t['notes'])
    spec = 'l' * min(2, len(columns)) + 'r' * max(0, len(columns) - 2)
    if longtable:
        text = (f'{{{size}\\setlength{{\\tabcolsep}}{{2pt}}\n\\begin{{longtable}}{{{spec}}}\n'
                f'\\caption{{{esc(t["title"])}. {notes}}}\\label{{tab:{t["id"]}}}\\\\\n\\toprule\n{head}\n\\midrule\n\\endfirsthead\n'
                f'\\toprule\n{head}\n\\midrule\n\\endhead\n{body}\n\\bottomrule\n\\end{{longtable}}}}\n')
    else:
        text = (f'\\begin{{table}}[htbp]\n\\centering\n{size}\n\\caption{{{esc(t["title"])}}}\\label{{tab:{t["id"]}}}\n'
                f'\\resizebox{{\\textwidth}}{{!}}{{%\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n{body}\n'
                f'\\bottomrule\n\\end{{tabular}}}}\n\\par\\smallskip\\parbox{{\\textwidth}}{{\\tiny {notes}}}\n\\end{{table}}\n')
    (Path(directory) / f"{t['id']}.tex").write_text(text)


def write_numbers(numbers, directory):
    """numbers.json and numbers.tex; the report reads values as \\val{key}."""
    Path(directory, 'numbers.json').write_text(json.dumps(numbers, indent=1, sort_keys=True) + '\n')
    lines = [r'\makeatletter', r'\newcommand{\val}[1]{\@ifundefined{val@#1}{\textbf{??\detokenize{#1}}}{\csname val@#1\endcsname}}']
    for key, value in sorted(numbers.items()):
        lines.append(r'\expandafter\def\csname val@' + key + r'\endcsname{' + fmt(value, key) + '}')
    lines.append(r'\makeatother')
    Path(directory, 'numbers.tex').write_text('\n'.join(lines) + '\n')
