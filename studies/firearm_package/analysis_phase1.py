"""Phase 1 analysis: E1 (detector arms) and E2 (recognition arms), and the gates G1/G2 as declared in
docs/superpowers/specs/2026-10-26-firearm-package-phase1-rules.md. Reads saved records only.

Intervals resample recording groups within category (2,000 draws, seed 20261026), as in Training 1.
"""
import json
from pathlib import Path
import numpy as np
from vera_runner.ledger import read_jsonl
from studies.firearm_training.analysis.stats import cluster_bootstrap

B, SEED = 2000, 20261026
GUN = ('Handgun', 'Machine_Gun')
EXPECTED_LABEL = {'Handgun': 'handgun', 'Machine_Gun': 'long gun'}


def ci(units, stat, name):
    est, lo, hi, _ = cluster_bootstrap(units, 'recording_group', 'category', stat, B, SEED, name)
    return {'est': est, 'low': lo, 'high': hi}


def ratio(num, den):
    return lambda u: (sum(num(x) for x in u) / d) if (d := sum(den(x) for x in u)) else None


def e1_units(manifest, records):
    by_id = {f['frame_id']: f for f in manifest}
    out = []
    for r in records:
        if r['status'] == 'failed':
            continue
        f = by_id[r['frame_id']]
        areas = [(g[2] - g[0]) * (g[3] - g[1]) / 1e6 for g in f['gt']]
        out.append({**r, 'recording_group': f['recording_group'], 'gt_area': max(areas) if areas else None,
                    'subset300': f['subset300'], 'subset100': f['subset100']})
    return out


def e1_metrics(units, name, missed_videos=()):
    gun = [u for u in units if u['category'] in GUN]
    neg = [u for u in units if u['category'] == 'No_Gun']
    s = lambda u: u['score']
    m = {'frames': len(units), 'firearm_frames': len(gun), 'nogun_frames': len(neg),
         'recall50': ci(gun, ratio(lambda u: s(u)['tp50'], lambda u: s(u)['n_gt']), f'{name} r50'),
         'recall75': ci(gun, ratio(lambda u: s(u)['tp75'], lambda u: s(u)['n_gt']), f'{name} r75'),
         'precision50': ci(gun, ratio(lambda u: s(u)['tp50'], lambda u: s(u)['n_pred']), f'{name} p50'),
         'nogun_fp': ci(neg, ratio(lambda u: s(u)['n_pred'] > 0, lambda u: 1), f'{name} fp') if neg else None,
         'whole_image_rate': (sum(s(u)['n_whole_image'] for u in units) / n) if (n := sum(s(u)['n_pred'] for u in units)) else None,
         'empty_firearm': sum(s(u)['n_pred'] == 0 for u in gun) / len(gun) if gun else None,
         'status': {k: sum(u['status'] == k for u in units) for k in sorted({u['status'] for u in units})},
         'seconds_median': float(np.median([u['seconds'] for u in units])) if units else None,
         'seconds_p95': float(np.percentile([u['seconds'] for u in units], 95)) if units else None,
         'tokens_median': float(np.median([u['tokens'] for u in units])) if units else None,
         'truncated': sum(bool(u.get('truncated')) for u in units)}
    for cat in GUN:
        part = [u for u in gun if u['category'] == cat]
        m[f'recall50_{cat}'] = ratio(lambda u: s(u)['tp50'], lambda u: s(u)['n_gt'])(part)
        hit = [u for u in part if s(u)['tp50'] > 0]
        m[f'label_right_{cat}'] = (sum(any(f['label'] == EXPECTED_LABEL[cat] for f in u['found']) for u in hit) / len(hit)) if hit else None
    areas = sorted(u['gt_area'] for u in gun if u['gt_area'] is not None)
    if areas:
        cuts = [areas[len(areas) // 3], areas[2 * len(areas) // 3]]
        for size, keep in (('small', lambda a: a < cuts[0]), ('medium', lambda a: cuts[0] <= a < cuts[1]), ('large', lambda a: a >= cuts[1])):
            m[f'recall50_{size}'] = ratio(lambda u: s(u)['tp50'], lambda u: s(u)['n_gt'])([u for u in gun if u['gt_area'] is not None and keep(u['gt_area'])])
    missed = [u for u in gun if u['category'] == 'Handgun' and u['video'] in missed_videos]
    m['recall50_handgun_missed_by_seed'] = ratio(lambda u: s(u)['tp50'], lambda u: s(u)['n_gt'])(missed)
    m['handgun_frames_missed_by_seed'] = len(missed)
    return m


def usable(m, rules):
    return (m['recall50']['est'] is not None and m['recall50']['est'] >= rules['min_recall50'] and
            (m['whole_image_rate'] or 0) <= rules['max_whole_image'] and (m['nogun_fp'] or {}).get('est', 0) <= rules['max_nogun_fp'])


def analyze_e1(out_root, manifest_path, seed_measurements, arms, rules):
    manifest = json.loads(Path(manifest_path).read_text())
    missed = {r['video'] for r in read_jsonl(seed_measurements)[0] if r['category'] == 'Handgun' and r['pred']['category'] != 'Handgun'}
    result = {}
    for name in arms:
        recs = read_jsonl(Path(out_root) / 'e1' / name / 'records.jsonl')[0]
        if not recs:
            continue
        units = e1_units(manifest, recs)
        result[name] = {'own': e1_metrics(units, name, missed),
                        'subset300': e1_metrics([u for u in units if u['subset300']], name + ' s300', missed)}
        if name.endswith('_r2'):
            base = {u['frame_id']: u for u in e1_units(manifest, read_jsonl(Path(out_root) / 'e1' / name[:-3] / 'records.jsonl')[0])}
            pairs = [(base[u['frame_id']], u) for u in units if u['frame_id'] in base and u['category'] in GUN]
            result[name]['repeat_agreement_tp50'] = (sum((a['score']['tp50'] > 0) == (b['score']['tp50'] > 0) for a, b in pairs) / len(pairs)) if pairs else None
    ranked = sorted((n for n, r in result.items() if not n.endswith('_r2') and usable(r['subset300'], rules)),
                    key=lambda n: (-result[n]['subset300']['recall50']['est'], (result[n]['subset300']['nogun_fp'] or {}).get('est', 0),
                                   result[n]['subset300']['seconds_median']))
    gate = {'usable': ranked, 'chosen': ranked[0] if ranked else None, 'rules': rules,
            'note': 'ranked on subset300, the frames every main arm ran; _r2 arms measure sampling variance only'}
    return result, gate


def e2_metrics(recs, name):
    units = [r for r in recs if r['status'] == 'complete']
    acc = lambda k: (lambda u: sum(x['pred'][k] == x['target'][i] for x in u) / len(u))
    m = {'videos': len(units), 'invalid': sum(u['pred']['category'] is None for u in units),
         'truncated': sum(bool(u.get('truncated')) for u in units),
         'seconds_median': float(np.median([u['seconds'] for u in units])) if units else None,
         'tokens_median': float(np.median([u['tokens'] for u in units])) if units else None}
    for i, k in enumerate(('category', 'group', 'code')):
        m[k] = ci(units, lambda u, i=i, k=k: sum(x['pred'][k] == x['target'][i] for x in u) / len(u), f'{name} {k}')
    for cat in ('Handgun', 'Machine_Gun', 'No_Gun'):
        part = [u for u in units if u['category'] == cat]
        m[f'category_{cat}'] = acc('category')(part) if part else None
    return m


def paired(a, b, k, i, name):
    """b minus a on field k (target index i), same videos."""
    ia = {r['video']: r for r in a if r['status'] == 'complete'}
    rows = [{**ia[r['video']], 'other': r} for r in b if r['status'] == 'complete' and r['video'] in ia]
    stat = lambda u: (sum(x['other']['pred'][k] == x['target'][i] for x in u) - sum(x['pred'][k] == x['target'][i] for x in u)) / len(u)
    return ci(rows, stat, name)


def analyze_e2(out_root, arms, reference='iv2_packed'):
    recs = {n: read_jsonl(Path(out_root) / 'e2' / n / 'records.jsonl')[0] for n in arms}
    recs = {n: r for n, r in recs.items() if r}
    result = {n: e2_metrics(r, n) for n, r in recs.items()}
    for n in recs:
        if n != reference and reference in recs:
            result[n]['vs_reference_category'] = paired(recs[reference], recs[n], 'category', 0, f'E2 {n} cat')
            result[n]['vs_reference_group'] = paired(recs[reference], recs[n], 'group', 1, f'E2 {n} grp')
    def better(a, b):  # b beats a on category with an interval above zero
        if a in recs and b in recs:
            d = paired(recs[a], recs[b], 'category', 0, f'gate {a} {b}')
            return d, d['low'] is not None and d['low'] > 0
        return None, False
    gate = {}
    d, ok = better('iv2_packed', 'iv35_packed')
    gate['backbone'] = {'compare': 'iv35_packed - iv2_packed', 'diff': d, 'chosen': 'internvl35_8b' if ok else 'internvl2_8b'}
    bb = 'iv35' if ok else 'iv2'
    options = [n for n in (f'{bb}_frames', f'{bb}_tiles') if n in recs]
    best = max(options, key=lambda n: result[n]['category']['est'], default=None)
    d, ok2 = better(f'{bb}_packed', best) if best else (None, False)
    gate['input'] = {'best_non_oracle': best, 'diff_vs_packed': d, 'chosen': best.split('_', 1)[1] if ok2 else 'packed'}
    d, ok3 = better('iv35_frames', 'iv35_think_frames') if bb == 'iv35' else (None, False)
    gate['learner_thinking'] = {'diff': d, 'chosen': ok3}
    gate['oracle_crops'] = {n: result[n]['vs_reference_category'] for n in ('iv2_crops', 'iv35_crops') if n in result and 'vs_reference_category' in result[n]}
    return result, gate
