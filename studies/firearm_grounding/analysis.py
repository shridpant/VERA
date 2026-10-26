"""Grounding study: tables R1-R7, figures F1-F4 and number macros, from saved artifacts only.

H1 (primary): per grounding-evaluable test video, mean over seeds of reliance(G) - reliance(P) (seeds where
both exist); mean over videos; 95 % CI from 2,000 draws resampling test recording groups within category.
H2 (co-primary): per test video, mean over seeds of correct(G) - correct(T1); lower 95 % bound > -0.05.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from vera_runner.ledger import read_jsonl
from studies.firearm_training.analysis.latex import esc, fmt, write_csv, write_numbers, write_tex
from studies.firearm_training.analysis.stats import cluster_bootstrap, pct, share

SEED, B = 20261026, 2000
ARMS = ('T1', 'G', 'P')
CATS = ('Handgun', 'Machine_Gun', 'No_Gun')
COLORS = {'T1': '#2a78d6', 'G': '#eb6834', 'P': '#1baf7a', 'seed questions': '#898781'}


def table(tid, title, columns, rows, notes):
    return {'id': tid, 'title': title, 'columns': columns, 'rows': rows, 'notes': notes}


def ci(units, stat, name):
    return cluster_bootstrap(units, 'recording_group', 'category', stat, B, SEED, name)


def correct(r):
    return r['pred']['category'] == r['category']


def rfwr(r):
    return r.get('reliance') is not None and correct(r) and r['reliance'] <= 0


def flipped(r, mask):
    """The unmasked answer is right and the answer regenerated under the mask is not."""
    return correct(r) and r['occlusion'][mask]['pred']['category'] != r['category']


def far_controls(r):
    """Every control patch at least one token cell (62.5 of 1000) from the firearm boxes."""
    gaps = (r.get('control_gaps') or {}).values()
    return bool(gaps) and min(gaps) >= 62.5


def size_bin(r, boxes, edges):
    own = boxes.get(r['video'], {}).get('boxes', {})
    areas = [(b[2] - b[0]) * (b[3] - b[1]) / 1e6 for k in r['boxed_slots'] for b in own.get(str(r['frame_indices'][k]), [])]
    if not areas:
        return None
    a = float(np.mean(areas))
    return 'small' if a < edges[0] else ('medium' if a < edges[1] else 'large')


def load(root):
    root = Path(root)
    d = {'phase0': json.loads((root / 'phase0/phase0.json').read_text()) if (root / 'phase0/phase0.json').exists() else None,
         'gate': json.loads((root / 'gate.json').read_text()) if (root / 'gate.json').exists() else None,
         'phase1': read_jsonl(root / 'phase1_seed_measurements.jsonl')[0] if (root / 'phase1_seed_measurements.jsonl').exists() else [],
         'runs': {}, 'root': str(root)}
    for run in sorted((root / 'arms').glob('*_s*')) if (root / 'arms').exists() else []:
        arm, seed = run.name.split('_s')
        entry = {'arm': arm, 'seed': int(seed), 'dir': run}
        if (run / 'selected.json').exists():
            entry['selected'] = json.loads((run / 'selected.json').read_text())
        src = Path(entry.get('selected', {}).get('adopted_from', run))
        entry['validations'] = read_jsonl(src / 'validations.jsonl')[0] if (src / 'validations.jsonl').exists() else []
        entry['steps'] = read_jsonl(src / 'steps.jsonl')[0] if (src / 'steps.jsonl').exists() else []
        entry['test'] = {r['video']: r for r in read_jsonl(run / 'test_measurements.jsonl')[0]} if (run / 'test_measurements.jsonl').exists() else {}
        d['runs'][(arm, int(seed))] = entry
    d['seed_test'] = {r['video']: r for r in d['phase1'] if r.get('split') == 'test'}
    return d


def r1(d):
    def detail(v):
        return '; '.join(f'{x} {y:.4g}' if isinstance(y, float) else f'{x} {y}' for x, y in v.items() if x != 'passed')
    rows = [{'check': k, 'passed': v['passed'], 'detail': detail(v)} for k, v in (d['phase0'] or {}).get('checks', {}).items()]
    return table('R1', 'Phase 0 measurement checks', ['check', 'passed', 'detail'], rows, [
        'b2/b3 failed their strict thresholds; the diagnostic (DEVIATIONS.md, D1) found the same two near-tie mismatches with a '
        'FlashAttention replay, so attention is read with eager attention and every probability with FlashAttention.'])


def transposed(t, key, title=None):
    """Report view of a wide table: one row per measure, one column per value of `key`."""
    cols = [str(r[key]) for r in t['rows']]
    rows = [{'measure': c, **{str(r[key]): r.get(c) for r in t['rows']}} for c in t['columns'] if c != key]
    return {**t, 'columns': ['measure'] + cols, 'rows': rows, 'title': title or t['title']}


def lift_summary(records, label):
    ev = [r for r in records if r.get('lift') is not None]
    row = {'set': label, 'videos': len(ev)}
    if not ev:
        return row
    row.update(median_lift=float(np.median([r['lift'] for r in ev])), mean_lift=float(np.mean([r['lift'] for r in ev])),
               share_above_chance=share(r['lift'] > 1 for r in ev),
               median_lift_shuffled=float(np.median([r['lift_shuffled'] for r in ev if r.get('lift_shuffled') is not None] or [np.nan])),
               median_sink_excluded=float(np.median([r['lift_sink_excluded'] for r in ev])),
               **{f'median_{k}': float(np.median([r['lift_layers'][k] for r in ev])) for k in ('early', 'middle', 'late')},
               mean_reliance=float(np.mean([r['reliance'] for r in ev if r.get('reliance') is not None] or [np.nan])))
    return row


def r2(d, boxes=None, edges=None):
    g = d['gate'] or {}
    rows = [{'measure': 'V1 lift true - shuffled, same frames (mean)', 'estimate': g.get('V1', {}).get('mean_difference'),
             'ci_low': (g.get('V1', {}).get('ci') or [None, None])[0], 'ci_high': (g.get('V1', {}).get('ci') or [None, None])[1],
             'videos': g.get('V1', {}).get('videos'), 'passed': g.get('V1', {}).get('passed')},
            {'measure': 'V2 Spearman rho(lift, reliance)', 'estimate': g.get('V2', {}).get('rho'),
             'ci_low': (g.get('V2', {}).get('ci') or [None, None])[0], 'ci_high': (g.get('V2', {}).get('ci') or [None, None])[1],
             'videos': g.get('V2', {}).get('videos'), 'passed': g.get('V2', {}).get('passed')}]
    for name, v in (g.get('V3') or {}).items():
        rows.append({'measure': f'V3 {name} category (mean lift)', 'estimate': v.get('mean_lift'), 'mean_reliance': v.get('mean_reliance'),
                     'videos': v.get('videos')})
    for name, keep in (('correct', True), ('incorrect', False)):
        sub = [r for r in d['phase1'] if r.get('lift') is not None and r.get('reliance') is not None
               and (r['pred']['category'] == r['category']) == keep]
        rows.append({'measure': f'Spearman rho within {name} category (not pre-registered)', 'videos': len(sub),
                     'estimate': float(spearmanr([r['lift'] for r in sub], [r['reliance'] for r in sub]).correlation) if len(sub) >= 5 else None})
    for size in ('small', 'medium', 'large') if boxes else ():
        sub = [r for r in d['phase1'] if r.get('lift') is not None and r.get('reliance') is not None and size_bin(r, boxes, edges) == size]
        rows.append({'measure': f'Spearman rho within {size} boxes (not pre-registered)', 'videos': len(sub),
                     'estimate': float(spearmanr([r['lift'] for r in sub], [r['reliance'] for r in sub]).correlation) if len(sub) >= 5 else None})
    gate_table = table('R2', 'Phase 1 validity gate (seed questions, all splits)',
                       ['measure', 'estimate', 'ci_low', 'ci_high', 'mean_reliance', 'videos', 'passed'], rows,
                       [f"Gate {'passed' if g.get('gate_passed') else 'not passed'}: V1 and V2 lower bounds above 0 (pre-registered). "
                        'V3 (descriptive): mean lift and mean reliance for answers with the correct and an incorrect category. '
                        'Rows marked not pre-registered were added after the independent review, before the gate was read.'])
    sens = [lift_summary(d['phase1'], 'seed questions, all splits'), lift_summary(d['seed_test'].values(), 'seed questions, test')]
    sens += [lift_summary(arm_records(d, arm), f'{arm}, test') for arm in ARMS if arm_records(d, arm)]
    sens_table = table('R2s', 'Lift by layer group and sink exclusion', list(sens[0]), sens,
                       ['Medians over grounding-evaluable videos; 1.0 = chance. early = layers 0-10, middle 11-21, late 22-31. '
                        'Arms pool their seeds. Sink excluded: each frame\'s most-attended token removed before renormalising.'])
    return gate_table, sens_table


def r3(d):
    rows = []
    for (arm, seed), e in sorted(d['runs'].items()):
        sel = e.get('selected') or {}
        m = sel.get('metrics', {})
        rows.append({'arm': arm, 'seed': seed, 'steps': sum(s['status'] == 'complete' for s in e['steps']),
                     'updates': sum(s.get('update') == 'updated' for s in e['steps']), 'selected_step': sel.get('step'),
                     'val_category_acc': m.get('category_accuracy'), 'val_group_acc': m.get('group_accuracy'),
                     'val_pooled_lift': m.get('pooled_lift'), 'adopted': 'adopted_from' in sel})
    return table('R3', 'Training summary per arm and seed', ['arm', 'seed', 'steps', 'updates', 'selected_step', 'val_category_acc',
                                                            'val_group_acc', 'val_pooled_lift', 'adopted'], rows,
                 ['T1 seed 20261026 is Training 1 itself. G and P select by pooled validation lift among steps within 5 points of '
                  'their best validation category accuracy; T1 by validation accuracy (VERA).'])


def per_video(d, arm, field):
    """{video: {seed: value}} for one arm."""
    out = {}
    for (a, seed), e in d['runs'].items():
        if a != arm:
            continue
        for v, r in e['test'].items():
            out.setdefault(v, {})[seed] = r.get(field) if field != 'correct' else float(correct(r))
    return out


def paired_units(d, a, b, field):
    va, vb = per_video(d, a, field), per_video(d, b, field)
    units = []
    for v in sorted(set(va) & set(vb)):
        seeds = [s for s in va[v] if s in vb[v] and va[v][s] is not None and vb[v][s] is not None]
        if not seeds:
            continue
        rec = next(e['test'][v] for e in d['runs'].values() if v in e['test'])
        units.append({'video': v, 'category': rec['category'], 'recording_group': rec['recording_group'],
                      'd': float(np.mean([va[v][s] - vb[v][s] for s in seeds])), 'seeds': len(seeds)})
    return units


def contrast(d, a, b, field, name):
    units = paired_units(d, a, b, field)
    est, lo, hi, _ = ci(units, lambda u: share(x['d'] for x in u), name)
    return {'contrast': f'{a} - {b}', 'measure': field, 'videos': len(units), 'estimate': est, 'ci_low': lo, 'ci_high': hi}


def r4(d):
    h1 = contrast(d, 'G', 'P', 'reliance', 'H1')
    h1['hypothesis'] = 'H1 primary: reliance G > P'
    h1['supported'] = h1['ci_low'] is not None and h1['ci_low'] > 0
    h2 = contrast(d, 'G', 'T1', 'correct', 'H2')
    h2['hypothesis'] = 'H2 co-primary: accuracy G >= T1 - 5 pp'
    h2['supported'] = h2['ci_low'] is not None and h2['ci_low'] > -0.05
    rows = [h1, h2]
    for seed in sorted({s for (_, s) in d['runs']}):
        for a, b, field in (('G', 'P', 'reliance'), ('G', 'T1', 'correct')):
            sub = {'runs': {k: v for k, v in d['runs'].items() if k[1] == seed}}
            c = contrast(sub, a, b, field, f'R4 seed {seed} {field}')
            rows.append({**c, 'hypothesis': f'per seed {seed}', 'supported': None})
    return table('R4', 'Primary and co-primary outcomes on the test split', ['hypothesis', 'contrast', 'measure', 'videos', 'estimate', 'ci_low',
                                                                             'ci_high', 'supported'], rows, [
        'Per test video, the mean over seeds of the arm difference; reliance in nats (log p_true with the control patch masked minus with '
        'the firearm masked), correct as 0/1. 95 % CI: 2,000 bootstrap draws over test recording groups within category (seed 20261026). '
        'Per-seed rows show the spread across training orders.'])


def arm_records(d, arm):
    if arm == 'seed questions':
        return list(d['seed_test'].values())
    return [r for (a, _), e in d['runs'].items() if a == arm for r in e['test'].values()]


def record_sets(d):
    """(label, records): Phase 1 on all splits and by true category, the seed questions on test, then each arm."""
    sets = [('seed, all splits', d['phase1'])] + [(f'seed, {c}', [r for r in d['phase1'] if r['category'] == c]) for c in CATS]
    return [(label, recs) for label, recs in sets + [(arm, arm_records(d, arm)) for arm in ('seed questions',) + ARMS] if recs]


def r5(d):
    rows = []
    for arm, recs in record_sets(d):
        ev = [r for r in recs if r.get('lift') is not None]
        firearm = [r for r in recs if r['category'] != 'No_Gun']
        nogun = [r for r in recs if r['category'] == 'No_Gun']
        rows.append({'arm': arm, 'answers': len(recs), 'category_acc': share(correct(r) for r in recs),
                     'group_acc': share(r['pred']['group'] == r['target'][1] for r in recs),
                     'no_decision': sum(r.get('decision') is None for r in recs), 'evaluable': len(ev),
                     'missing_reliance': sum(r.get('reliance') is None for r in ev),
                     'mean_lift': share(r['lift'] for r in ev), 'median_lift': float(np.median([r['lift'] for r in ev])) if ev else None,
                     'mean_reliance': share(r['reliance'] for r in ev if r.get('reliance') is not None),
                     'right_for_wrong_reasons': share(rfwr(r) for r in ev if r.get('reliance') is not None),
                     'flip_box_mask': share(flipped(r, 'box') for r in ev if r.get('occlusion')),
                     'flip_control_mask': share(flipped(r, 'control') for r in ev if r.get('occlusion')),
                     'weapon_mentioned_firearm': share(r['weapon_mentioned'] for r in firearm),
                     'weapon_mentioned_no_gun': share(r['weapon_mentioned'] for r in nogun)})
    return table('R5', 'Behaviour, grounding and reliance by question set (seed rows: Phase 1; others: test split, pooled over seeds)', list(rows[0]) if rows else ['arm'], rows, [
        'seed, ...: Phase 1 (seed questions) on all 398 videos, then by true category. seed questions: the same questions on the '
        'test split only, the reference for the arms. Right for the wrong reasons: correct category with reliance <= 0. '
        'Flip: the unmasked answer has the true category and the answer regenerated under the mask does not (share of '
        'videos with occlusion). Missing reliance: grounding-evaluable videos without a reliance value (no usable control '
        'or an excluded decision under a mask). '
        'Weapon mentioned: the explanation (text before Category:) names a weapon; on No_Gun videos this is a false mention.'])


def r6(d, boxes, edges):
    rows = []
    for arm, all_recs in [(a, r) for a, r in record_sets(d) if a == 'seed, all splits' or not a.startswith('seed,')]:
        recs = [r for r in all_recs if r.get('lift') is not None]
        for size in ('small', 'medium', 'large'):
            sub = [r for r in recs if size_bin(r, boxes, edges) == size]
            rows.append({'arm': arm, 'size': size, 'videos': len(sub), 'mean_lift': share(r['lift'] for r in sub),
                         'mean_reliance': share(r['reliance'] for r in sub if r.get('reliance') is not None)})
    return table('R6', 'Lift and reliance by firearm size (Phase 1 on all splits; question sets on the test split)', ['arm', 'size', 'videos', 'mean_lift', 'mean_reliance'], rows,
                 ['Size: mean annotated box area of the video\'s boxed sampled frames, binned at the GT-area tertiles of the annotation audit.'])


def r7(d):
    far = {v for e in d['runs'].values() for v, r in e['test'].items() if far_controls(r)}
    sub = {'runs': {k: {**e, 'test': {v: r for v, r in e['test'].items() if v in far}} for k, e in d['runs'].items()}}
    rows = [{**contrast(sub, 'G', 'P', 'reliance', 'R7 H1 far controls'), 'contrast': 'G - P (controls >= 1 cell away)'}]
    for a, b in (('G', 'T1'), ('P', 'T1'), ('G', 'P')):
        for field in ('lift', 'reliance', 'correct'):
            rows.append(contrast(d, a, b, field, f'R7 {a}-{b} {field}'))
    return table('R7', 'Secondary contrasts (test split, paired by video, mean over seeds)', ['contrast', 'measure', 'videos', 'estimate',
                                                                                            'ci_low', 'ci_high'], rows, [
        'Same bootstrap as R4. Not corrected for multiplicity; read as descriptive. First row: H1 on the videos whose '
        'control patches all lie at least one token cell from the firearm (not pre-registered; a control next to the gun '
        'can cover part of the hand or gun and pull reliance toward 0).'])


def figures(d, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e1e0d9',
                         'grid.linewidth': 0.5, 'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 200})
    out.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name):
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}', bbox_inches='tight')
        plt.close(fig)
        made.append(name)
    ev = [r for r in d['phase1'] if r.get('lift') is not None and r.get('lift_shuffled') is not None]
    if ev:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        bins = np.logspace(-1.5, 1.5, 31)
        axes[0].hist([r['lift'] for r in ev], bins=bins, alpha=0.7, color='#eb6834', label='true boxes')
        axes[0].hist([r['lift_shuffled'] for r in ev], bins=bins, alpha=0.6, color='#2a78d6', label='shuffled boxes')
        axes[0].axvline(1, color='#52514e', linewidth=0.8)
        axes[0].set_xscale('log'), axes[0].set_xlabel('attention lift (1 = chance)'), axes[0].set_ylabel('videos'), axes[0].legend()
        rel = [r for r in ev if r.get('reliance') is not None]
        for keep, label, color in ((True, 'correct category', '#1baf7a'), (False, 'wrong category', '#eb6834')):
            sub = [r for r in rel if correct(r) == keep]
            axes[1].scatter([r['lift'] for r in sub], [r['reliance'] for r in sub], s=8, color=color, alpha=0.7, label=label)
        axes[1].legend(loc='center left')
        axes[1].set_xscale('log'), axes[1].axhline(0, color='#52514e', linewidth=0.8), axes[1].axvline(1, color='#52514e', linewidth=0.8)
        axes[1].set_xticks([0.25, 0.5, 1, 2, 4, 8], ['0.25', '0.5', '1', '2', '4', '8']), axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[1].set_xlabel('attention lift (1 = chance)'), axes[1].set_ylabel('occlusion reliance (nats)')
        fig.tight_layout()
        save(fig, 'F1_phase1_lift_and_reliance')
    if d['runs']:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
        for ax, field, label in ((axes[0], 'reliance', 'mean reliance (nats)'), (axes[1], 'lift', 'mean attention lift'), (axes[2], 'correct', 'category accuracy')):
            for i, arm in enumerate(ARMS):
                vals = []
                for (a, s), e in sorted(d['runs'].items()):
                    if a != arm or not e['test']:
                        continue
                    recs = list(e['test'].values())
                    x = [float(correct(r)) for r in recs] if field == 'correct' else [r[field] for r in recs if r.get(field) is not None]
                    if x:
                        vals.append(float(np.mean(x)))
                if vals:
                    ax.scatter([i] * len(vals), vals, color=COLORS[arm], s=18, zorder=3)
                    ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=COLORS[arm], linewidth=2)
            seed_recs = list(d['seed_test'].values())
            sx = [float(correct(r)) for r in seed_recs] if field == 'correct' else [r[field] for r in seed_recs if r.get(field) is not None]
            if sx:
                ax.axhline(np.mean(sx), color=COLORS['seed questions'], linewidth=0.8, linestyle=(0, (2, 2)))
            ax.set_xticks(range(3), ARMS), ax.set_title(label, fontsize=8), ax.grid(axis='x', visible=False)
        fig.suptitle('Dots: seeds; bars: mean over seeds; dashed: seed questions', fontsize=7)
        fig.tight_layout()
        save(fig, 'F2_arms')
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for (arm, seed), e in sorted(d['runs'].items()):
            vals = [v for v in e['validations'] if not v['reference_only']]
            if not vals:
                continue
            style = {'color': COLORS[arm], 'linewidth': 1, 'alpha': 0.8}
            axes[0].plot([v['step'] for v in vals], [v['metrics']['category_accuracy'] for v in vals], **style)
            lifts = [(v['step'], v['metrics'].get('pooled_lift')) for v in vals if v['metrics'].get('pooled_lift') is not None]
            if lifts:
                axes[1].plot(*zip(*lifts), **style)
        for arm in ARMS:
            axes[0].plot([], [], color=COLORS[arm], label=arm)
        axes[0].legend(loc='lower right'), axes[0].set_xlabel('step'), axes[0].set_ylabel('validation category accuracy')
        axes[1].set_xlabel('step'), axes[1].set_ylabel('validation pooled lift (G, P)')
        fig.tight_layout()
        save(fig, 'F3_validation_trajectories')
    return made


def numbers(tables, d):
    n = {}
    g = d['gate'] or {}
    for k in ('V1', 'V2'):
        v = g.get(k, {})
        n[f'gate_{k}_estimate'] = v.get('mean_difference', v.get('rho'))
        n[f'gate_{k}_low'], n[f'gate_{k}_high'] = (v.get('ci') or [None, None])
        n[f'gate_{k}_videos'] = v.get('videos')
    n['gate_passed'] = g.get('gate_passed')
    for r in tables['R4']['rows'][:2]:
        key = 'H1' if r['hypothesis'].startswith('H1') else 'H2'
        n.update({f'{key}_estimate': r['estimate'], f'{key}_low': r['ci_low'], f'{key}_high': r['ci_high'], f'{key}_videos': r['videos'],
                  f'{key}_supported': r['supported']})
    for r in tables['R5']['rows']:
        key = r['arm'].replace(' ', '').replace(',', '_')
        for c in ('answers', 'evaluable', 'category_acc', 'group_acc', 'mean_lift', 'median_lift', 'mean_reliance', 'right_for_wrong_reasons',
                  'flip_box_mask', 'flip_control_mask', 'weapon_mentioned_firearm', 'weapon_mentioned_no_gun'):
            n[f'R5_{key}_{c}'] = r.get(c)
    for r in tables['R2s']['rows']:
        n[f"sens_{r['set'].replace(' ', '').replace(',', '_')}_median_lift"] = r.get('median_lift')
    n['phase1_videos'] = len(d['phase1'])
    n['phase1_evaluable'] = sum(r.get('lift') is not None for r in d['phase1'])
    n['phase1_with_reliance'] = sum(r.get('reliance') is not None for r in d['phase1'])
    n['phase1_decision_excluded'] = sum(r.get('decision') is None for r in d['phase1'])
    mass = [r['decision']['candidate_mass'] for r in d['phase1'] if r.get('decision') and 'candidate_mass' in r['decision']]
    n['phase1_min_candidate_mass'] = min(mass) if mass else None
    gaps = [g for r in d['phase1'] for g in (r.get('control_gaps') or {}).values()]
    n['phase1_controls_within_one_cell_pct'] = pct(share(g < 62.5 for g in gaps)) if gaps else None
    for r in tables['R2']['rows']:
        if 'within' in r['measure']:
            key = r['measure'].split('within ')[1].split(' (')[0].replace(' ', '_')
            n[f'rho_within_{key}'], n[f'rho_within_{key}_videos'] = r.get('estimate'), r.get('videos')
    n['runs'] = len(d['runs'])
    return n


def appendix(d, out):
    """questions.tex (seed and every run's selected questions) and grounding_example.txt (the first G step's block)."""
    def listing(title, text):
        return f'\\subsection*{{{esc(title)}}}\n\\begin{{lstlisting}}\n{text.strip()}\n\\end{{lstlisting}}\n'
    seed = json.loads((Path(d['root']) / 'phase1_seed_measurements.session.json').read_text()).get('questions', '') \
        if (Path(d['root']) / 'phase1_seed_measurements.session.json').exists() else ''
    parts = [listing('Seed questions', seed)] if seed else []
    for (arm, seed_), e in sorted(d['runs'].items()):
        sel = e.get('selected')
        if sel:
            parts.append(listing(f"{arm}, seed {seed_} (step {sel['step']})", sel['questions']))
    (out / 'questions.tex').write_text('\n'.join(parts) or 'No run has selected questions.\n')
    example = next((s for (arm, _), e in sorted(d['runs'].items()) if arm == 'G' for s in e['steps'] if s.get('grounding_block')), None)
    (out / 'grounding_example.txt').write_text(f"step {example['step']}\n{example['grounding_block']}\n" if example else 'none\n')
    from .arms import grounded_template
    t1 = (Path(__file__).resolve().parents[1] / 'firearm_training/prompts/optimizer_instruct.txt').read_text()
    (out / 'optimizer_grounded.txt').write_text(grounded_template(t1))


def analyze(root, boxes_dir, policy_path, out_dir):
    d = load(root)
    boxes = {}
    for s in ('train', 'val', 'test'):
        p = Path(boxes_dir) / f'boxes_{s}.json'
        if p.exists():
            boxes.update(json.loads(p.read_text()))
    edges = json.loads(Path(policy_path).read_text())['size_bins_area_fraction']
    out = Path(out_dir)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    r2a, r2b = r2(d, boxes, edges)
    tables = {t['id']: t for t in (r1(d), r2a, r2b, r3(d), r4(d), r5(d), r6(d, boxes, edges), r7(d))}
    views = {'R2s': 'set', 'R5': 'arm'}
    for t in tables.values():
        write_csv(t, out / 'tables')
        write_tex(transposed(t, views[t['id']]) if t['id'] in views and t['rows'] else t, out / 'tables')
    made = figures(d, out / 'figures')
    appendix(d, out)
    write_numbers(numbers(tables, d), out)
    manifest = {'tables': sorted(tables), 'figures': made, 'runs': sorted(f'{a}_s{s}' for a, s in d['runs'])}
    (out / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest
