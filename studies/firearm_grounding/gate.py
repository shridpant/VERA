"""Phase 1 gate of the grounding pre-registration, from a measurement JSONL (seed questions, all splits).

V1 spatial specificity: mean(lift with true boxes - lift with shuffled boxes) over grounding-evaluable
   videos that have both; 95 % CI from 2,000 draws resampling recording groups within category.
V2 agreement: Spearman rho between video lift and occlusion reliance; same bootstrap.
V3 behaviour (descriptive): lift and reliance for correct vs incorrect categories.
Gate: V1 CI above 0 and V2 CI above 0. Otherwise Phase 2 does not run.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from vera_runner.ledger import read_jsonl
from studies.firearm_training.analysis.stats import cluster_bootstrap

SEED, B = 20261026, 2000


def evaluable(records):
    return [r for r in records if r.get('lift') is not None]


def v1(records):
    rows = [r for r in evaluable(records) if r.get('lift_shuffled') is not None]
    units = [{'g': r['recording_group'], 'category': r['category'], 'd': r['lift'] - r['lift_shuffled']} for r in rows]
    est, lo, hi, n = cluster_bootstrap(units, 'g', 'category', lambda u: float(np.mean([x['d'] for x in u])) if u else None, B, SEED, 'V1')
    return {'videos': len(rows), 'mean_lift_true': float(np.mean([r['lift'] for r in rows])) if rows else None,
            'mean_lift_shuffled': float(np.mean([r['lift_shuffled'] for r in rows])) if rows else None,
            'median_lift_true': float(np.median([r['lift'] for r in rows])) if rows else None,
            'mean_difference': est, 'ci': [lo, hi], 'draws': n, 'passed': lo is not None and lo > 0}


def v2(records):
    rows = [r for r in evaluable(records) if r.get('reliance') is not None]
    units = [{'g': r['recording_group'], 'category': r['category'], 'lift': r['lift'], 'rel': r['reliance']} for r in rows]

    def rho(u):
        if len(u) < 5 or len({x['lift'] for x in u}) < 2 or len({x['rel'] for x in u}) < 2:
            return None
        return float(spearmanr([x['lift'] for x in u], [x['rel'] for x in u]).correlation)
    est, lo, hi, n = cluster_bootstrap(units, 'g', 'category', rho, B, SEED, 'V2')
    return {'videos': len(rows), 'rho': est, 'ci': [lo, hi], 'draws': n, 'passed': lo is not None and lo > 0,
            'mean_reliance': float(np.mean([r['reliance'] for r in rows])) if rows else None}


def v3(records):
    out = {}
    for name, keep in (('correct', True), ('incorrect', False)):
        rows = [r for r in evaluable(records) if (r['pred']['category'] == r['category']) == keep]
        out[name] = {'videos': len(rows), 'mean_lift': float(np.mean([r['lift'] for r in rows])) if rows else None,
                     'mean_reliance': float(np.mean([r['reliance'] for r in rows if r.get('reliance') is not None]))
                     if any(r.get('reliance') is not None for r in rows) else None}
    return out


def gate(path, out_path=None):
    records = read_jsonl(path)[0]
    result = {'records': len(records), 'grounding_evaluable': len(evaluable(records)),
              'no_decision': sum(r.get('decision') is None for r in records), 'V1': v1(records), 'V2': v2(records), 'V3': v3(records)}
    result['gate_passed'] = result['V1']['passed'] and result['V2']['passed']
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=1) + '\n')
    return result
