"""Paired recording-group inference for seed-averaged pooled box F1."""
import numpy as np


def grounding_test(groups, rg, sg, bootstraps=5000, permutations=100000,
                   exact_max_groups=16, seed=20261027):
    """Counts [fixed seeds, videos, (TP, predicted boxes, GT boxes)].

    Exchange methods jointly for every video and seed within a recording group.
    This tests groupwise method-label exchangeability, not randomized treatment
    assignment. The bootstrap interval estimates an RG-minus-SG F1 difference.
    """
    a, b = np.asarray(rg, dtype=float), np.asarray(sg, dtype=float)
    if (a.shape != b.shape or a.ndim != 3 or a.shape[1] != len(groups) or
        a.shape[2] != 3 or a.shape[0] == 0 or not len(groups) or
        not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('matching finite [seeds,videos,3] counts required')
    for x in (a, b):
        if (x < 0).any() or (x != np.floor(x)).any() or (x[:, :, 0] > x[:, :, 1:3].min(axis=2)).any():
            raise ValueError('invalid box counts')
    if not np.array_equal(a[:, :, 2], b[:, :, 2]) or not np.all(a[:, :, 2] == a[0, :, 2]):
        raise ValueError('ground truth must match across methods and seeds')
    unique = sorted(set(groups))
    index = {g: i for i, g in enumerate(unique)}
    ac = np.zeros((len(a), len(unique), 3))
    bc = np.zeros_like(ac)
    for j, group in enumerate(groups):
        ac[:, index[group]] += a[:, j]
        bc[:, index[group]] += b[:, j]
    support = int((ac[0, :, 2] > 0).sum())
    result = {'contrast': 'RG-SG', 'endpoint': 'mean_across_seeds_pooled_F1_IoU50',
              'recording_groups': len(unique), 'positive_gt_recording_groups': support,
              'seeds': len(a), 'videos': len(groups), 'bootstrap_requested': bootstraps,
              'rng_seed': seed, 'permutation_rng_seed': seed + 1, 'bootstrap_unit': 'recording_group',
              'permutation_unit': 'one joint method swap per recording group across all videos and seeds',
              'null': 'joint groupwise exchangeability of RG and SG count vectors',
              'assumptions': 'independent representative recording groups; method exchangeability for permutation null; fixed training seeds'}
    if len(unique) < 2 or support < 2:
        return {**result, 'status': 'inconclusive', 'reason': 'fewer than two independent groups with positive ground truth'}
    def difference(aa, bb):
        fa = 2 * aa[..., 0] / (aa[..., 1] + aa[..., 2])
        fb = 2 * bb[..., 0] / (bb[..., 1] + bb[..., 2])
        return (fa - fb).mean(axis=-1)
    at, bt = ac.sum(axis=1), bc.sum(axis=1)
    observed = float(difference(at, bt))
    result.update(difference=observed,
                  rg_F1_by_seed=(2 * at[:, 0] / (at[:, 1] + at[:, 2])).tolist(),
                  sg_F1_by_seed=(2 * bt[:, 0] / (bt[:, 1] + bt[:, 2])).tolist())
    rng = np.random.default_rng(seed)
    boot = []
    # Recompute pooled ratios, never average frame F1 or resample seeds/frames.
    for _ in range(bootstraps):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        aa, bb = np.einsum('g,sgc->sc', weights, ac), np.einsum('g,sgc->sc', weights, bc)
        if (aa[:, 2] > 0).all():
            boot.append(float(difference(aa, bb)))
    result['bootstrap_valid'] = len(boot)
    if len(boot) < .95 * bootstraps:
        return {**result, 'status': 'inconclusive', 'reason': 'too many resamples without positive GT support'}
    result['RG_F1_seed_sd'] = float(np.std(result['rg_F1_by_seed'], ddof=1)) if len(a) > 1 else None
    result['SG_F1_seed_sd'] = float(np.std(result['sg_F1_by_seed'], ddof=1)) if len(a) > 1 else None
    result['paired_F1_difference_by_seed'] = (2 * at[:, 0] / (at[:, 1] + at[:, 2]) - 2 * bt[:, 0] / (bt[:, 1] + bt[:, 2])).tolist()
    result['confidence_interval_95'] = np.quantile(boot, [.025, .975]).tolist()
    exact = len(unique) <= exact_max_groups
    total = 2 ** len(unique) if exact else permutations
    rng = np.random.default_rng(seed + 1)
    extreme = 0
    for start in range(0, total, 1024):
        size = min(1024, total - start)
        if exact:
            values = np.arange(start, start + size, dtype=np.uint64)
            swaps = ((values[:, None] >> np.arange(len(unique), dtype=np.uint64)) & 1).astype(float)
        else:
            swaps = rng.integers(0, 2, (size, len(unique))).astype(float)
        changes = np.einsum('bg,sgc->bsc', swaps, bc - ac)
        simulated = difference(at[None] + changes, bt[None] - changes)
        extreme += int((np.abs(simulated) >= abs(observed) - 1e-14).sum())
    p = extreme / total if exact else (extreme + 1) / (total + 1)
    result.update(status='complete', p_value_two_sided=p,
                  permutation_mode='exact' if exact else 'Monte_Carlo_plus_one',
                  permutations=total, permutations_as_or_more_extreme=extreme,
                  nominal_positive_evidence=bool(observed > 0 and p <= .05 and result['confidence_interval_95'][0] > 0))
    return result


def claim(result, split, recognition_passed):
    """Fixed sequence: existing test recognition rule, then test grounding F1."""
    result = dict(result)
    result['role'] = 'key_secondary_fixed_sequence' if split == 'test' else 'exploratory_validation'
    result['recognition_gate_passed'] = bool(recognition_passed)
    result['confirmatory_grounding_improvement'] = bool(
        split == 'test' and recognition_passed and result.get('nominal_positive_evidence', False))
    return result
