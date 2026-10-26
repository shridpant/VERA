"""Comparison of Training 1 runs (faithful seeds and declared variants) from their saved logs only.

Tables C1-C6 and figures G1-G3. Test intervals resample recording groups within category (2,000
draws, seed 20261026), as in run.py. Every run tests the same seed questions with greedy decoding,
so their seed-question test answers must agree; C1 reports any disagreement as a determinism check.
Validation scores of several methods on the same 62 videos are optimistic; the test split decides.
"""
import json
from pathlib import Path
import numpy as np
from vera_runner.ledger import read_jsonl
from .latex import esc, fmt, write_csv, write_numbers, write_tex
from .run import FIELDS, acc, correct, interval, table
from .stats import cluster_bootstrap, pct, share

SEED, B = 20261026, 2000
NAMES = {'vera': 'VERA (faithful)', 'vera_fixed': 'VERA, fixed', 'bank': 'Question bank', 'bank_random': 'Bank, random pairs',
         'bank_noscore': 'Bank, no scoring'}
COLORS = ['#2a78d6', '#6250d6', '#008300', '#eb6834', '#1baf7a', '#e87ba4', '#eda100', '#e34948']


def load_run(run_dir):
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / 'spec.json').read_text())
    tr = spec['training']
    r = {'dir': run_dir, 'id': spec['experiment_id'], 'method': tr.get('method', 'vera'), 'seed': tr['seed'], 'spec': spec,
         'validations': read_jsonl(run_dir / 'validations.jsonl')[0], 'tests': {t['name']: t for t in read_jsonl(run_dir / 'test.jsonl')[0]},
         'steps': [s for s in read_jsonl(run_dir / 'steps.jsonl')[0] if s['status'] == 'complete'],
         'rounds': [s for s in read_jsonl(run_dir / 'rounds.jsonl')[0] if s['status'] == 'complete'],
         'passes': read_jsonl(run_dir / 'passes.jsonl')[0],
         'sessions': [json.loads(p.read_text()) for p in sorted((run_dir / 'sessions').glob('*/session.json'))]}
    learned = [v for v in r['validations'] if not v['reference_only']]
    r['selected'] = max(learned, key=lambda v: (v['metrics']['category_accuracy'], v['metrics']['group_accuracy'], -v['step'])) if learned else None
    r['label'] = f"{NAMES.get(r['method'], r['method'])} (order seed {r['seed']})"
    return r


def hours(r):
    secs = 0.0
    for s in r['steps']:
        secs += s.get('learner_seconds', 0) + s.get('optimizer_seconds', 0) + s.get('check_seconds', 0)
    for s in r['rounds']:
        secs += s.get('proposer_seconds', 0) + s.get('score_seconds', 0)
    secs += sum(p.get('seconds', 0) for p in r['passes'])
    secs += sum(p['batch_seconds'] for v in r['validations'] for p in v['predictions'][::2])
    secs += sum(p['batch_seconds'] for t in r['tests'].values() for p in t['predictions'][::2])
    return secs / 3600


def by_video(preds):
    return {p['video']: p for p in preds}


def paired(a, b, k, name):
    """b minus a accuracy on field k over the videos both cover, recording-group bootstrap."""
    va, vb = by_video(a), by_video(b)
    rows = [{**va[v], 'other': vb[v], 'group_key': va[v]['recording_group']} for v in sorted(set(va) & set(vb))]
    stat = lambda u: share(correct(x['other'], k) for x in u) - share(correct(x, k) for x in u)
    return cluster_bootstrap(rows, 'group_key', 'category', stat, B, SEED, name)


def c1(runs):
    rows = []
    seed_reference = None
    for r in runs:
        t = r['tests'].get('learned')
        s = r['tests'].get('seed')
        if s and seed_reference is None:
            seed_reference = by_video(s['predictions'])
        disagree = None if not s else sum(by_video(s['predictions'])[v]['response'] != seed_reference[v]['response'] for v in seed_reference)
        row = {'run': r['label'], 'method': r['method'], 'order_seed': r['seed'],
               'units': len(r['steps']) or len(r['rounds']), 'validations': len([v for v in r['validations'] if not v['reference_only']]),
               'selected_step': r['selected']['step'] if r['selected'] else None,
               'val_category_acc': r['selected']['metrics']['category_accuracy'] if r['selected'] else None,
               'seed_test_disagreements': disagree, 'gpu_hours': hours(r)}
        if t:
            p = t['predictions']
            units = [{**x, 'group_key': x['recording_group']} for x in p]
            for k, f in enumerate(FIELDS):
                est, lo, hi, _ = cluster_bootstrap(units, 'group_key', 'category', lambda u, k=k: acc(u, k), B, SEED, f"C1 {r['id']} {f}")
                row.update({f'test_{f}_acc': est, f'test_{f}_low': lo, f'test_{f}_high': hi})
            row['test_invalid'] = sum(x['pred']['category'] is None for x in p)
            row['test_at_cap'] = sum(x['generated_tokens'] >= 1024 for x in p)
        rows.append(row)
    return table('C1', 'All Training 1 runs: selected questions on the test split', list(rows[0]) if rows else ['run'], rows, [
        'Selection on validation category accuracy, then action-group accuracy, then the earliest step. seed_test_disagreements: test videos '
        'whose seed-question answer differs from the first run\'s (greedy decoding should make it 0).'])


def c2(runs, reference):
    rows = []
    for r in runs:
        if 'learned' not in r['tests'] or 'seed' not in r['tests']:
            continue
        row = {'run': r['label']}
        for k, f in enumerate(FIELDS[:2]):
            est, lo, hi, _ = paired(r['tests']['seed']['predictions'], r['tests']['learned']['predictions'], k, f"C2 seed {r['id']} {f}")
            row.update({f'vs_seed_{f}': est, f'vs_seed_{f}_low': lo, f'vs_seed_{f}_high': hi})
            if reference is not None and r is not reference and 'learned' in reference['tests']:
                est, lo, hi, _ = paired(reference['tests']['learned']['predictions'], r['tests']['learned']['predictions'], k,
                                        f"C2 ref {r['id']} {f}")
                row.update({f'vs_faithful_{f}': est, f'vs_faithful_{f}_low': lo, f'vs_faithful_{f}_high': hi})
        rows.append(row)
    cols = ['run'] + [f'vs_{w}_{f}{s}' for w in ('seed', 'faithful') for f in FIELDS[:2] for s in ('', '_low', '_high')]
    return table('C2', 'Paired test differences: learned questions minus seed questions, and minus the faithful run', cols, rows, [
        'Same 96 test videos; recording-group bootstrap. The faithful reference is the first faithful run (order seed 20261026).'])


def c3(runs):
    faithful = [r for r in runs if r['method'] == 'vera' and 'learned' in r['tests']]
    rows = []
    for key, get in (('validation category acc. (selected)', lambda r: r['selected']['metrics']['category_accuracy']),
                     ('test category acc.', lambda r: acc(r['tests']['learned']['predictions'], 0)),
                     ('test action-group acc.', lambda r: acc(r['tests']['learned']['predictions'], 1))):
        values = [get(r) for r in faithful]
        if values:
            rows.append({'measure': key, 'runs': len(values), 'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                         'min': min(values), 'max': max(values), 'values': ', '.join(f'{v:.3f}' for v in values)})
    return table('C3', 'Faithful VERA across training-order seeds', ['measure', 'runs', 'mean', 'sd', 'min', 'max', 'values'], rows, [
        'Same split, prompts, seed questions and model; only the shuffled training order differs.'])


def c4(runs):
    rows = []
    for r in runs:
        if r['method'] == 'vera_fixed':
            s = r['steps']
            rows.append({'run': r['label'], 'units': len(s), 'proposed_and_checked': sum(x['candidate_check'] is not None for x in s),
                         'rejected_format': sum(x['rejected_because'] not in (None, 'no update') for x in s),
                         'no_update': sum(x['rejected_because'] == 'no update' for x in s), 'accepted': sum(x['accepted'] for x in s),
                         'kept_or_bank_size': sum(x['accepted'] for x in s), 'mean_score': None})
        elif r['method'].startswith('bank'):
            s = r['rounds']
            scores = [x['score'] for x in s if x['score'] is not None]
            rows.append({'run': r['label'], 'units': len(s), 'proposed_and_checked': sum(x['question'] is not None for x in s),
                         'rejected_format': sum(x['question'] is None for x in s), 'no_update': None,
                         'accepted': sum(bool(x['kept']) for x in s), 'kept_or_bank_size': sum(bool(x['kept']) for x in s),
                         'mean_score': share(scores) if scores else None})
        elif r['steps']:
            s = r['steps']
            rows.append({'run': r['label'], 'units': len(s), 'proposed_and_checked': None, 'rejected_format': None,
                         'no_update': sum(x['update'] != 'updated' for x in s), 'accepted': sum(x['update'] == 'updated' for x in s),
                         'kept_or_bank_size': len({x['questions_after'] for x in s}), 'mean_score': None})
    return table('C4', 'What each method did during training', ['run', 'units', 'proposed_and_checked', 'rejected_format', 'no_update',
                                                                 'accepted', 'kept_or_bank_size', 'mean_score'], rows, [
        'Faithful: accepted = rewrites (VERA keeps every rewrite), kept = distinct question sets. Fixed: a rewrite is checked unless its '
        'format is rejected and is accepted when not worse on the check set. Bank: a valid question is scored; kept when balanced '
        'accuracy >= 0.75 (no-scoring ablation keeps all); mean_score = mean balanced accuracy of scored questions.'])


def figures(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e1e0d9',
                         'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 200})
    out.mkdir(parents=True, exist_ok=True)
    made = []
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    seed_val = None
    for r, color in zip(runs, COLORS):
        learned = sorted((v for v in r['validations'] if not v['reference_only']), key=lambda v: v['step'])
        if not learned:
            continue
        x = np.arange(1, len(learned) + 1) / len(learned)
        ax.plot(x, [v['metrics']['category_accuracy'] for v in learned], marker='o', markersize=3, linewidth=1.6, color=color, label=r['label'])
        seed_val = seed_val or next((v['metrics']['category_accuracy'] for v in r['validations'] if v['reference_only']), None)
    if seed_val is not None:
        ax.axhline(seed_val, color='#898781', linewidth=0.9, linestyle=(0, (2, 2)), label='seed questions')
    ax.set_xlabel('fraction of the run (validation checkpoints)'), ax.set_ylabel('validation category accuracy'), ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=6, ncol=2)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'G1_validation_all_runs.{ext}', bbox_inches='tight')
    plt.close(fig)
    made.append('G1_validation_all_runs')
    tested = [r for r in runs if 'learned' in r['tests']]
    if tested:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.45 * len(tested) + 1.2), sharey=True)
        for ax, k, title in ((axes[0], 0, 'category'), (axes[1], 1, 'action group')):
            for i, (r, color) in enumerate(zip(tested, COLORS)):
                p = [{**x, 'group_key': x['recording_group']} for x in r['tests']['learned']['predictions']]
                est, lo, hi, _ = cluster_bootstrap(p, 'group_key', 'category', lambda u, k=k: acc(u, k), B, SEED, f"G2 {r['id']} {k}")
                ax.errorbar(est, i, xerr=[[est - lo], [hi - est]], fmt='o', color=color, capsize=2)
            seed = tested[0]['tests'].get('seed')
            if seed:
                ax.axvline(acc(seed['predictions'], k), color='#898781', linewidth=0.9, linestyle=(0, (2, 2)))
            ax.set_title(f'test {title} accuracy (dashed: seed questions)', fontsize=8), ax.set_xlim(0, 1)
        axes[0].set_yticks(range(len(tested)), [r['label'] for r in tested], fontsize=6)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'G2_test_all_runs.{ext}', bbox_inches='tight')
        plt.close(fig)
        made.append('G2_test_all_runs')
    banks = [r for r in runs if r['rounds'] and any(x['score'] is not None for x in r['rounds'])]
    if banks:
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        for r, color in zip(banks, COLORS[2:]):
            ax.hist([x['score'] for x in r['rounds'] if x['score'] is not None], bins=np.linspace(0.5, 1, 11), alpha=0.6, color=color, label=r['label'])
        ax.axvline(0.75, color='#898781', linewidth=0.9)
        ax.set_xlabel('balanced accuracy of a proposed question on its pair (keep threshold 0.75)'), ax.set_ylabel('questions')
        ax.legend(fontsize=6)
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'G3_bank_scores.{ext}', bbox_inches='tight')
        plt.close(fig)
        made.append('G3_bank_scores')
    return made


def compare(run_dirs, out_dir):
    runs = [load_run(d) for d in run_dirs]
    reference = next((r for r in runs if r['method'] == 'vera'), None)
    out = Path(out_dir)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    tables = {t['id']: t for t in (c1(runs), c2(runs, reference), c3(runs), c4(runs))}
    for t in tables.values():
        write_csv(t, out / 'tables')
    views = {}
    rows = []
    for r in tables['C1']['rows']:
        rows.append({'Run': esc(r['run']), 'Units': fmt(r['units']), 'Selected step': fmt(r['selected_step']),
                     'Val. category \\%': fmt(pct(r['val_category_acc']), 'pct'),
                     'Test category \\% [95\\% CI]': interval(r.get('test_category_acc'), r.get('test_category_low'), r.get('test_category_high')),
                     'Test action group \\%': interval(r.get('test_group_acc'), r.get('test_group_low'), r.get('test_group_high')),
                     'Test code \\%': interval(r.get('test_code_acc'), r.get('test_code_low'), r.get('test_code_high')),
                     'Invalid / at cap': f"{fmt(r.get('test_invalid'))} / {fmt(r.get('test_at_cap'))}", 'GPU h': fmt(r['gpu_hours'], 'hours')})
    views['C1'] = table('C1', tables['C1']['title'], list(rows[0]) if rows else ['Run'], rows, tables['C1']['notes'])
    rows = []
    for r in tables['C2']['rows']:
        rows.append({'Run': esc(r['run']),
                     'vs seed: category (pp)': interval(r.get('vs_seed_category'), r.get('vs_seed_category_low'), r.get('vs_seed_category_high')),
                     'vs seed: group (pp)': interval(r.get('vs_seed_group'), r.get('vs_seed_group_low'), r.get('vs_seed_group_high')),
                     'vs faithful: category (pp)': interval(r.get('vs_faithful_category'), r.get('vs_faithful_category_low'), r.get('vs_faithful_category_high')),
                     'vs faithful: group (pp)': interval(r.get('vs_faithful_group'), r.get('vs_faithful_group_low'), r.get('vs_faithful_group_high'))})
    views['C2'] = table('C2', tables['C2']['title'], list(rows[0]) if rows else ['Run'], rows, tables['C2']['notes'])
    for tid in ('C1', 'C2'):
        write_tex(views[tid], out / 'tables', raw=True)
    for tid in ('C3', 'C4'):
        write_tex(tables[tid], out / 'tables')
    made = figures(runs, out / 'figures')
    questions = ''.join(f"\\subsection*{{{esc(r['label'])}: selected at step {r['selected']['step']}}}\n\\begin{{lstlisting}}\n"
                        f"{r['selected']['questions'].rstrip()}\n\\end{{lstlisting}}\n" for r in runs if r['selected'])
    (out / 'selected_questions.tex').write_text(questions)
    numbers = {}
    for r in tables['C1']['rows']:
        key = f"{r['method']}_{r['order_seed']}"
        for f in FIELDS:
            numbers[f'cmp_{key}_test_{f}_pct'] = pct(r.get(f'test_{f}_acc'))
        numbers[f'cmp_{key}_val_category_pct'] = pct(r['val_category_acc'])
        numbers[f'cmp_{key}_seed_disagreements'] = r['seed_test_disagreements']
    write_numbers(numbers, out)
    (out / 'compare_manifest.json').write_text(json.dumps({'runs': [str(r['dir']) for r in runs], 'tables': sorted(tables), 'figures': made}, indent=1) + '\n')
    return {'runs': [r['label'] for r in runs], 'tables': sorted(tables), 'figures': made}
