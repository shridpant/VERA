"""Training 1 tables (S1-S10), figures (F1-F4), qualitative cases and number macros, from saved logs only.

Test statistics are video-level on the test split. Intervals are 95 % percentile intervals of 2,000
bootstrap draws resampling recording groups within category (repeated takes stay together; seed
20261026). The paired learned-minus-seed difference resamples the same groups for both. An invalid
field counts as wrong; macro-F1 averages the per-category F1 over the three categories.
stats.py and latex.py are copies of studies/firearm/analysis/{stats,latex}.py from study/firearm-a-g-ag.
"""
import json
from pathlib import Path
import numpy as np
from vera_runner.ledger import read_jsonl
from .latex import esc, fmt, write_csv, write_numbers, write_tex
from .stats import cluster_bootstrap, pct, share

CATS = ('Handgun', 'Machine_Gun', 'No_Gun')
GROUPS = ('carry', 'aim', 'move', 'phone', 'bottle', 'heavy_object')
FIELDS = ('category', 'group', 'code')
SEED = 20261026
B = 2000
COND = {'seed': '#2a78d6', 'learned': '#eb6834'}


def table(tid, title, columns, rows, notes):
    return {'id': tid, 'title': title, 'columns': columns, 'rows': rows, 'notes': notes}


def correct(p, k):
    return p['pred'][FIELDS[k]] == p['target'][k]


def acc(preds, k):
    return share(correct(p, k) for p in preds) if preds else None


def macro_f1(preds):
    f1s = []
    for c in CATS:
        tp = sum(p['pred']['category'] == c and p['target'][0] == c for p in preds)
        fp = sum(p['pred']['category'] == c and p['target'][0] != c for p in preds)
        fn = sum(p['pred']['category'] != c and p['target'][0] == c for p in preds)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return share(f1s)


def units(preds):
    return [{**p, 'group_key': p['recording_group']} for p in preds]


def ci(preds, stat, name):
    return cluster_bootstrap(units(preds), 'group_key', 'category', stat, B, SEED, name)


def load(run_dir, data_dir):
    run_dir = Path(run_dir)
    d = {'spec': json.loads((run_dir / 'spec.json').read_text()), 'steps': [s for s in read_jsonl(run_dir / 'steps.jsonl')[0]],
         'validations': read_jsonl(run_dir / 'validations.jsonl')[0], 'tests': {t['name']: t for t in read_jsonl(run_dir / 'test.jsonl')[0]},
         'sessions': [json.loads(p.read_text()) for p in sorted((run_dir / 'sessions').glob('*/session.json'))],
         'splits': json.loads((Path(data_dir) / 'splits.json').read_text())}
    d['complete_steps'] = [s for s in d['steps'] if s['status'] == 'complete']
    learned = [v for v in d['validations'] if not v['reference_only']]
    d['selected'] = max(learned, key=lambda v: (v['metrics']['category_accuracy'], v['metrics']['group_accuracy'], -v['step'])) if learned else None
    return d


def s1(d):
    rows = []
    for split, r in d['splits']['splits'].items():
        rows.append({'split': split, 'videos': r['videos'], 'recording_groups': r['recording_groups'], 'action_codes': r['action_codes'],
                     **{f'cat_{c}': r['category'].get(c, 0) for c in CATS}, **{f'grp_{g}': r['action_group'].get(g, 0) for g in GROUPS},
                     'C1': r['camera'].get('C1', 0), 'C2': r['camera'].get('C2', 0),
                     **{s: r['subject'].get(s, 0) for s in ('V1', 'V2', 'V3', 'V4')}})
    return table('S1', 'Split composition', list(rows[0]), rows, [
        'Recording groups (repeated takes of one action by one subject at one place) never span splits; every action code is in every split.'])


def s2(d):
    rows = []
    for v in d['validations']:
        m = v['metrics']
        rows.append({'step': v['step'], 'epoch': v['epoch'], 'questions': 'seed' if v['reference_only'] else 'learned',
                     'selected': bool(d['selected'] and v['step'] == d['selected']['step'] and not v['reference_only']),
                     'category_accuracy': m['category_accuracy'], 'group_accuracy': m['group_accuracy'], 'code_accuracy': m['code_accuracy'],
                     'category_invalid': m['category_invalid'], 'group_invalid': m['group_invalid'], 'code_invalid': m['code_invalid']})
    return table('S2', 'Validation trajectory', list(rows[0]) if rows else ['step'], rows, [
        'Step 0 validates the seed questions (reference only, not a selection candidate). Selection: highest category accuracy, '
        'then action-group accuracy, then the earliest step.'])


def s4(d):
    rows, diffs = [], []
    for name in ('seed', 'learned'):
        t = d['tests'].get(name)
        if not t:
            continue
        p = t['predictions']
        row = {'questions': name, 'videos': len(p)}
        for k, f in enumerate(FIELDS):
            est, lo, hi, _ = ci(p, lambda u, k=k: acc(u, k), f'S4 {name} {f}')
            row.update({f'{f}_acc': est, f'{f}_low': lo, f'{f}_high': hi, f'{f}_invalid': sum(x['pred'][f] is None for x in p)})
        est, lo, hi, _ = ci(p, macro_f1, f'S4 {name} macro')
        row.update(category_macro_f1=est, macro_low=lo, macro_high=hi, answers_at_token_cap=sum(x['generated_tokens'] >= 1024 for x in p))
        rows.append(row)
    if len(rows) == 2:
        seed, learned = d['tests']['seed']['predictions'], d['tests']['learned']['predictions']
        pair = [{**a, 'learned': b} for a, b in zip(sorted(seed, key=lambda x: x['video']), sorted(learned, key=lambda x: x['video']))]
        assert all(a['video'] == a['learned']['video'] for a in pair)
        row = {'questions': 'learned - seed', 'videos': len(pair)}
        for k, f in enumerate(FIELDS):
            stat = lambda u, k=k: share(correct(x['learned'], k) for x in u) - share(correct(x, k) for x in u)
            est, lo, hi, _ = ci(pair, stat, f'S4 diff {f}')
            row.update({f'{f}_acc': est, f'{f}_low': lo, f'{f}_high': hi})
        est, lo, hi, _ = ci(pair, lambda u: macro_f1([x['learned'] for x in u]) - macro_f1(u), 'S4 diff macro')
        row.update(category_macro_f1=est, macro_low=lo, macro_high=hi)
        rows.append(row)
    cols = ['questions', 'videos'] + [f'{f}_{s}' for f in FIELDS for s in ('acc', 'low', 'high')] + ['category_macro_f1', 'macro_low', 'macro_high'] + \
           [f'{f}_invalid' for f in FIELDS] + ['answers_at_token_cap']
    return table('S4', 'Test results: seed questions vs the selected learned questions', cols, rows, [
        'Accuracy over the 96 test videos (invalid = wrong). 95 % intervals resample recording groups within category (2,000 draws, '
        'seed 20261026); the difference row is paired on the same videos. Chance: category 1/3, action group 1/6, code 1/41.'])


def confusion(preds, k, labels):
    rows = []
    for t in labels:
        sub = [p for p in preds if p['target'][k] == t]
        row = {'target': t, 'videos': len(sub)}
        for l in labels:
            row[l] = sum(p['pred'][FIELDS[k]] == l for p in sub)
        row['invalid'] = sum(p['pred'][FIELDS[k]] is None for p in sub)
        rows.append(row)
    return rows


def s5(d):
    out = []
    for name in ('seed', 'learned'):
        if name in d['tests']:
            out += [{'questions': name, **r} for r in confusion(d['tests'][name]['predictions'], 0, CATS)]
    return table('S5', 'Category confusion on the test split (rows: true category)', ['questions', 'target', 'videos', *CATS, 'invalid'], out,
                 ['Counts of test videos.'])


def s6(d):
    out = []
    for name in ('seed', 'learned'):
        if name in d['tests']:
            out += [{'questions': name, **r} for r in confusion(d['tests'][name]['predictions'], 1, GROUPS)]
    return table('S6', 'Action-group confusion on the test split (rows: true group)', ['questions', 'target', 'videos', *GROUPS, 'invalid'], out,
                 ['carry/aim occur in Handgun and Machine_Gun videos; move/phone/bottle/heavy_object in No_Gun videos.'])


def s7(d):
    rows = []
    for name in ('seed', 'learned'):
        t = d['tests'].get(name)
        if not t:
            continue
        for split_by in ('category', 'camera'):
            for value in sorted({p[split_by] for p in t['predictions']}):
                sub = [p for p in t['predictions'] if p[split_by] == value]
                rows.append({'questions': name, 'stratum': f'{split_by}={value}', 'videos': len(sub), 'category_acc': acc(sub, 0),
                             'group_acc': acc(sub, 1), 'code_acc': acc(sub, 2)})
    return table('S7', 'Test accuracy by category and camera', ['questions', 'stratum', 'videos', 'category_acc', 'group_acc', 'code_acc'], rows,
                 ['Descriptive strata; C1 = IP camera 640x480, C2 = phone 1920x1080.'])


def s8(d):
    rows = []
    for name in ('seed', 'learned'):
        t = d['tests'].get(name)
        if not t:
            continue
        p = t['predictions']
        wrong = [x for x in p if not correct(x, 2)]
        rows.append({'questions': name, 'videos': len(p), 'code_correct': len(p) - len(wrong),
                     'wrong_code_right_group': sum(x['pred']['code'] is not None and correct(x, 1) for x in wrong),
                     'wrong_code_right_category': sum(x['pred']['code'] is not None and correct(x, 0) for x in wrong),
                     'code_invalid': sum(x['pred']['code'] is None for x in p),
                     'code_inconsistent_with_category': sum(x['pred']['code'] is not None and x['pred']['category'] is not None
                                                            and d['book']['codes'][x['pred']['code']]['category'] != x['pred']['category'] for x in p)})
    return table('S8', 'Exact action codes (exploratory)', list(rows[0]) if rows else ['questions'], rows, [
        '41 codes, 2-3 test videos each: per-code accuracy is not estimable and is not reported. wrong_code_right_group: a wrong code '
        'whose action group is correct. code_inconsistent_with_category: the predicted code belongs to another category than the '
        'predicted category.'])


def s9(d):
    steps = d['complete_steps']
    updated = [s for s in steps if s['update'] == 'updated']
    lines = [len([l for l in s['questions_after'].split('\n') if l.strip()]) for s in steps]
    rows = [{'steps': len(steps), 'updated': len(updated), 'kept': len(steps) - len(updated),
             'distinct_question_sets': len({s['questions_after'] for s in steps}),
             'median_nonblank_lines': float(np.median(lines)) if lines else None,
             'optimizer_answers_at_cap': sum(s['optimizer_tokens'] >= 1024 for s in steps),
             'learner_answers_invalid_category': sum(p[0] == 'invalid' for s in steps for p in s['predictions']),
             'learner_answers_at_token_cap': sum(tok >= 1024 for s in steps for tok in s['learner_tokens']),
             'learner_answers': sum(len(s['predictions']) for s in steps)}]
    return table('S9', 'Optimizer behaviour during training', list(rows[0]), rows, [
        'updated: the optimizer answer contained "New Prompt Questions" (VERA then takes the text after it, at most five lines); kept: it did not. '
        'optimizer_answers_at_cap / learner_answers_at_token_cap: answers that reached the 1,024-token limit (greedy repetition loops '
        'are not stopped: VERA decodes greedily without a repetition penalty).'])


def s10(d):
    steps = d['complete_steps']
    rows = []
    for key, label in (('learner_seconds', 'learner (2 videos)'), ('optimizer_seconds', 'optimizer (16 frames)')):
        x = [s[key] for s in steps]
        tok = [sum(s['learner_tokens']) if key == 'learner_seconds' else s['optimizer_tokens'] for s in steps]
        rows.append({'stage': label, 'calls': len(x), 'total_hours': sum(x) / 3600 if x else None, 'median_s': float(np.median(x)) if x else None,
                     'p95_s': float(np.quantile(x, 0.95)) if x else None, 'median_tokens': float(np.median(tok)) if tok else None})
    for name, records in (('validation', d['validations']), ('test', list(d['tests'].values()))):
        batch_s = [p['batch_seconds'] for r in records for p in r['predictions'][::2]]
        rows.append({'stage': name, 'calls': len(batch_s), 'total_hours': sum(batch_s) / 3600 if batch_s else None,
                     'median_s': float(np.median(batch_s)) if batch_s else None, 'p95_s': float(np.quantile(batch_s, 0.95)) if batch_s else None})
    loads = [s.get('model_load_seconds') for s in d['sessions'] if s.get('model_load_seconds')]
    rows.append({'stage': 'model loads', 'calls': len(loads), 'total_hours': sum(loads) / 3600 if loads else None,
                 'median_s': float(np.median(loads)) if loads else None})
    gpus = sorted({s.get('gpu') for s in d['sessions'] if s.get('gpu')})
    return table('S10', 'Computation', ['stage', 'calls', 'total_hours', 'median_s', 'p95_s', 'median_tokens'], rows,
                 [f"GPU: {'; '.join(gpus) or 'not recorded'}. BF16, FlashAttention, greedy, 1,024-token cap."])


def qualitative(d):
    """Per true category, the first correct and the first wrong learned-question answer by video ID."""
    t = d['tests'].get('learned')
    if not t:
        return []
    cases = []
    for c in CATS:
        sub = sorted((p for p in t['predictions'] if p['target'][0] == c), key=lambda p: p['video'])
        for want in (True, False):
            hit = next((p for p in sub if correct(p, 0) == want), None)
            if hit:
                cases.append({'video': hit['video'], 'true': hit['target'], 'pred': [hit['pred'][f] for f in FIELDS],
                              'kind': 'correct category' if want else 'wrong category', 'response': hit['response']})
    return cases


def figures(d, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e1e0d9',
                         'grid.linewidth': 0.5, 'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 200})
    out.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name):
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}', bbox_inches='tight')
        plt.close(fig)
        made.append(name)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    learned = [v for v in d['validations'] if not v['reference_only']]
    seed = next((v for v in d['validations'] if v['reference_only']), None)
    for f, color, label in (('category', '#2a78d6', 'category'), ('group', '#eb6834', 'action group'), ('code', '#1baf7a', 'action code')):
        ax.plot([v['step'] for v in learned], [v['metrics'][f + '_accuracy'] for v in learned], marker='o', markersize=4, color=color, linewidth=2,
                label=label)
        if seed:
            ax.axhline(seed['metrics'][f + '_accuracy'], color=color, linewidth=0.8, linestyle=(0, (2, 2)))
    if d['selected']:
        ax.axvline(d['selected']['step'], color='#898781', linewidth=0.8)
    ax.set_xlabel('training step (batches of 2 videos)'), ax.set_ylabel('validation accuracy'), ax.set_ylim(0, 1)
    ax.legend(loc='upper left', ncol=3)
    ax.set_title('Thin dashed lines: seed questions; grey line: selected step', fontsize=7)
    save(fig, 'F1_validation_trajectory')
    if d['tests']:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for ax, name in zip(axes, ('seed', 'learned')):
            if name not in d['tests']:
                ax.axis('off')
                continue
            rows = confusion(d['tests'][name]['predictions'], 0, CATS)
            m = np.array([[r[c] for c in CATS] + [r['invalid']] for r in rows])
            ax.imshow(m, cmap='Blues', vmin=0, vmax=max(1, m.max()))
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    ax.text(j, i, m[i, j], ha='center', va='center', color='white' if m[i, j] > 0.6 * m.max() else '#0b0b0b')
            ax.set_xticks(range(4), [c.replace('_', ' ') for c in CATS] + ['invalid']), ax.set_yticks(range(3), [c.replace('_', ' ') for c in CATS])
            ax.grid(False)
            ax.set_title(f'{name} questions (rows: true)', fontsize=8)
        fig.tight_layout()
        save(fig, 'F2_category_confusion')
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
        for ax, k, title in ((axes[0], 0, 'category'), (axes[1], 1, 'action group')):
            for j, name in enumerate(('seed', 'learned')):
                if name not in d['tests']:
                    continue
                vals = [acc([p for p in d['tests'][name]['predictions'] if p['target'][0] == c], k) or 0 for c in CATS]
                ax.bar(np.arange(3) * 2.6 + j, vals, width=0.9, color=COND[name], label=name, edgecolor='#fcfcfb')
                for x, v in zip(np.arange(3) * 2.6 + j, vals):
                    ax.text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=6, color='#52514e')
            ax.set_xticks(np.arange(3) * 2.6 + 0.5, [c.replace('_', ' ') for c in CATS]), ax.set_ylim(0, 1.1)
            ax.set_title(f'{title} accuracy by true category', fontsize=8), ax.grid(axis='x', visible=False)
        axes[0].legend(loc='upper left')
        fig.tight_layout()
        save(fig, 'F3_accuracy_by_category')
    steps = d['complete_steps']
    if steps:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
        for ax, key, label in ((axes[0], 'learner_seconds', 'learner seconds per batch'), (axes[1], 'optimizer_seconds', 'optimizer seconds per step')):
            ax.hist([s[key] for s in steps], bins=30, color='#2a78d6', edgecolor='#fcfcfb')
            ax.set_xlabel(label), ax.set_ylabel('steps')
        fig.tight_layout()
        save(fig, 'F4_runtime')
    return made


def numbers(d, tables):
    n = {}
    if d['selected']:
        n['selected_step'] = d['selected']['step']
        n['selected_epoch'] = d['selected']['epoch']
        for f in FIELDS:
            n[f'val_selected_{f}_acc_pct'] = pct(d['selected']['metrics'][f + '_accuracy'])
    seed = next((v for v in d['validations'] if v['reference_only']), None)
    if seed:
        for f in FIELDS:
            n[f'val_seed_{f}_acc_pct'] = pct(seed['metrics'][f + '_accuracy'])
    n['steps_complete'] = len(d['complete_steps'])
    n['validations'] = len([v for v in d['validations'] if not v['reference_only']])
    for r in tables['S4']['rows']:
        key = r['questions'].replace(' - ', 'minus')
        for f in FIELDS:
            n[f'test_{key}_{f}_acc_pct'] = pct(r[f + '_acc'])
            n[f'test_{key}_{f}_low_pct'] = pct(r[f + '_low'])
            n[f'test_{key}_{f}_high_pct'] = pct(r[f + '_high'])
        n[f'test_{key}_macro_f1'] = r['category_macro_f1']
    for r in tables['S9']['rows']:
        n.update({f'opt_{k}': v for k, v in r.items()})
    for r in tables['S10']['rows']:
        n[f"hours_{r['stage'].split()[0]}"] = r['total_hours']
    for s, r in d['splits']['splits'].items():
        n[f'split_{s}_videos'] = r['videos']
        n[f'split_{s}_groups'] = r['recording_groups']
    return n


def interval(est, lo, hi, scale=100.0, digits=1):
    if est is None:
        return '--'
    text = f'{est * scale:.{digits}f}'
    return text if lo is None else f'{text} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]'


def views(tables):
    """Readable LaTeX versions (raw cells) of the wide tables; the CSVs keep every column."""
    out = {}
    s1 = tables['S1']
    keys = [c for c in s1['columns'] if c != 'split']
    label = lambda c: c.replace('cat_', '').replace('grp_', 'group: ').replace('_', ' ')
    out['S1'] = table('S1', s1['title'], ['attribute'] + [r['split'] for r in s1['rows']],
                      [{'attribute': esc(label(k)), **{r['split']: fmt(r[k]) for r in s1['rows']}} for k in keys], s1['notes'])
    s4 = tables['S4']
    rows = []
    for r in s4['rows']:
        rows.append({'Questions': esc(r['questions']), 'Videos': fmt(r['videos']),
                     'Category acc. \\% [95\\% CI]': interval(r['category_acc'], r['category_low'], r['category_high']),
                     'Category macro-F1 [95\\% CI]': interval(r['category_macro_f1'], r['macro_low'], r['macro_high'], 1.0, 3),
                     'Action-group acc. \\%': interval(r['group_acc'], r['group_low'], r['group_high']),
                     'Code acc. \\%': interval(r['code_acc'], r['code_low'], r['code_high']),
                     'Invalid (cat./group/code)': '--' if r['questions'] == 'learned - seed' else
                     f"{r['category_invalid']}/{r['group_invalid']}/{r['code_invalid']}",
                     'At token cap': '--' if r['questions'] == 'learned - seed' else fmt(r['answers_at_token_cap'])})
    out['S4'] = table('S4', s4['title'], list(rows[0]) if rows else ['Questions'], rows, s4['notes'])
    s9 = tables['S9']
    out['S9'] = table('S9', s9['title'], ['measure', 'value'],
                      [{'measure': esc(k.replace('_', ' ')), 'value': fmt(v, k)} for r in s9['rows'] for k, v in r.items()], s9['notes'])
    return out


def analyze(run_dir, data_dir, codebook_path, out_dir):
    d = load(run_dir, data_dir)
    d['book'] = json.loads(Path(codebook_path).read_text())
    out = Path(out_dir)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    tables = {t['id']: t for t in (s1(d), s2(d), s4(d), s5(d), s6(d), s7(d), s8(d), s9(d), s10(d))}
    readable = views(tables)
    for t in tables.values():
        write_csv(t, out / 'tables')
        if t['id'] in readable:
            write_tex(readable[t['id']], out / 'tables', raw=True)
        else:
            write_tex(t, out / 'tables')
    made = figures(d, out / 'figures')
    write_numbers(numbers(d, tables), out)
    cases = qualitative(d)
    (out / 'qualitative.json').write_text(json.dumps(cases, indent=1, ensure_ascii=False) + '\n')
    (out / 'qualitative.tex').write_text(''.join(
        f"\\paragraph{{{esc(c['video'])} ({esc(c['kind'])})}} True: {esc(', '.join(c['true']))}; predicted: "
        f"{esc(', '.join(x or 'invalid' for x in c['pred']))}.\\\\\n\\textit{{Answer:}} {esc(c['response'])}\n\n" for c in cases) or 'No cases.\n')
    q = {'seed': d['spec']['training']['seed_questions'], 'selected': d['selected']['questions'] if d['selected'] else ''}
    for name, text in q.items():
        (out / f'questions_{name}.txt').write_text(text)
    for name in ('learner_instruct', 'optimizer_instruct'):
        (out / f'{name}.txt').write_text(d['spec']['prompts'][name])
    manifest = {'tables': sorted(tables), 'figures': made, 'selected_step': d['selected']['step'] if d['selected'] else None,
                'spec_sha256': d['spec']['spec_sha256'], 'tests': sorted(d['tests'])}
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest
