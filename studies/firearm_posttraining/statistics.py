"""Paired recording-group bootstrap; frames/seeds are not independent videos."""
import numpy as np
from .data import CATEGORIES


def _bacc(correct, categories, weights):
    return sum(np.average(correct[categories == c], weights=weights[categories == c]) for c in CATEGORIES) / 3


def paired_interval(rows, a_correct, b_correct, a_false_positive, b_false_positive, repeats=5000, seed=20261026):
    """Inputs [seeds, videos], paired on both axes; seed-average per-video outcomes.

    This interval resamples recording groups, not seeds. Report across-seed spread
    separately. Never interpret three seeds as three independent data samples.
    """
    n = len(rows)
    arrays = [np.asarray(x, dtype=float) for x in (a_correct, b_correct, a_false_positive, b_false_positive)]
    if any(x.ndim != 2 or x.shape[1] != n for x in arrays):
        raise ValueError('expected [seed,video] paired arrays')
    if arrays[0].shape != arrays[1].shape or arrays[2].shape != arrays[3].shape:
        raise ValueError('seed pairing differs')
    a, b, afp, bfp = [x.mean(axis=0) for x in arrays]
    cats = np.array([r['category'] for r in rows])
    groups = np.array([r['recording_group'] for r in rows])
    unique = sorted(set(groups))
    if set(cats) != set(CATEGORIES):
        raise ValueError('all categories required')
    negative = cats == 'No_Gun'
    group_index = np.array([unique.index(g) for g in groups])
    rng = np.random.default_rng(seed)
    category, false_positive = [], []
    for _ in range(repeats):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[group_index]
        if any(weights[cats == c].sum() == 0 for c in CATEGORIES):
            continue
        category.append(_bacc(a, cats, weights) - _bacc(b, cats, weights))
        false_positive.append(np.average(afp[negative] - bfp[negative], weights=weights[negative]))
    if len(category) < repeats * .95:
        raise ValueError('insufficient class coverage in bootstrap samples')
    # Zero-event bootstrap [0,0] cannot establish a 5pp noninferiority margin.
    from scipy.stats import binomtest
    delta = arrays[2] - arrays[3]
    negative_groups = sorted(set(groups[negative]))
    sizes, harmful, uniformly_beneficial = [], [], []
    for group in negative_groups:
        mask = negative & (groups == group)
        sizes.append(int(mask.sum()))
        harmful.append(bool((delta[:, mask] > 0).any()))
        uniformly_beneficial.append(bool((delta[:, mask] < 0).all()))
    g = len(negative_groups)
    harm_upper = binomtest(sum(harmful), g).proportion_ci(.95, method='exact').high
    benefit_lower = binomtest(sum(uniformly_beneficial), g).proportion_ci(.95, method='exact').low
    # Each endpoint has 2.5% error. The observed size bounds conservatively
    # weight group harm and uniformly beneficial events for this support screen.
    mean_size = float(np.mean(sizes))
    support_upper = min(1., max(-1., max(sizes) / mean_size * harm_upper - min(sizes) / mean_size * benefit_lower))
    raw_fp_interval = np.quantile(false_positive, [.025, .975]).tolist()
    return {'balanced_category_accuracy_difference': _bacc(a, cats, np.ones(n)) - _bacc(b, cats, np.ones(n)),
            'balanced_category_accuracy_interval': np.quantile(category, [.025, .975]).tolist(),
            'no_gun_false_positive_difference': float((afp[negative] - bfp[negative]).mean()),
            'no_gun_false_positive_interval': raw_fp_interval,
            'no_gun_noninferiority_upper_screen': max(raw_fp_interval[1], support_upper),
            'no_gun_group_support': {'groups': g, 'harmful_groups': sum(harmful),
                 'uniformly_beneficial_groups': sum(uniformly_beneficial), 'support_upper': support_upper,
                 'assumptions': 'independent recording groups; observed group-size bounds; conservative support screen, not the bootstrap interval'},
            'recording_groups': len(unique), 'bootstrap_replicates': len(category),
            'interval_unit': 'recording groups, paired across methods and fixed seeds'}


def transitions(rows, baseline_correct, method_correct):
    if len(rows) != len(baseline_correct) or len(rows) != len(method_correct):
        raise ValueError('paired video counts differ')
    def count(indices):
        recovered = sum(not baseline_correct[i] and method_correct[i] for i in indices)
        broken = sum(baseline_correct[i] and not method_correct[i] for i in indices)
        return {'recovered': recovered, 'broken': broken, 'net': recovered - broken}
    return {'all': count(range(len(rows))),
            'by_category': {c: count([i for i, r in enumerate(rows) if r['category'] == c]) for c in CATEGORIES}}
