"""Paired cluster bootstrap and small summaries.

Windows overlap and share source frames, so they are not independent. Uncertainty comes from
resampling whole clusters (source videos by default; recording groups as a sensitivity) with
replacement, within each category (stratified), and recomputing the statistic on the pooled units
of the drawn clusters. Each statistic gets its own generator seeded from (seed, crc32(name)), so a
result does not depend on the order in which tables are built.
"""
import zlib
import numpy as np


def rng_for(seed, name):
    return np.random.default_rng([seed, zlib.crc32(name.encode('utf-8'))])


def cluster_bootstrap(units, cluster_key, strata_key, statistic, resamples, seed, name, level=0.95):
    """units: list of dicts; statistic(list_of_units) -> float or None. Returns (estimate, low, high, valid_draws)."""
    estimate = statistic(units)
    if estimate is None:
        return None, None, None, 0
    strata = {}
    for u in units:
        strata.setdefault(u[strata_key], {}).setdefault(u[cluster_key], []).append(u)
    groups = {s: list(clusters.values()) for s, clusters in strata.items()}
    rng = rng_for(seed, name)
    draws = []
    for _ in range(resamples):
        sample = []
        for clusters in groups.values():
            for i in rng.integers(0, len(clusters), len(clusters)):
                sample.extend(clusters[i])
        value = statistic(sample)
        if value is not None:
            draws.append(value)
    if not draws:
        return estimate, None, None, 0
    alpha = (1 - level) / 2
    return estimate, float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha)), len(draws)


def share(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def quantiles(values, qs=(0.25, 0.5, 0.75)):
    values = [v for v in values if v is not None]
    return [float(np.quantile(values, q)) for q in qs] if values else [None] * len(qs)


def pct(x):
    return None if x is None else 100.0 * x


def cluster_bootstrap_sums(units, cluster_key, strata_key, values, statistic, resamples, seed, name, level=0.95):
    """cluster_bootstrap for statistics of summed quantities, vectorised.

    values(unit) -> k numbers; statistic(totals) maps an (n, k) array of column sums to n values
    (nan where undefined). Each cluster's values are summed once; a draw resamples clusters with
    replacement within each stratum and adds their sums. Same estimator as cluster_bootstrap,
    without rebuilding every draw in Python."""
    sums, strata = {}, {}
    for u in units:
        key = (u[strata_key], u[cluster_key])
        v = np.asarray(values(u), dtype=float)
        sums[key] = sums[key] + v if key in sums else v
    if not sums:
        return None, None, None, 0
    keys = list(sums)
    matrix = np.stack([sums[k] for k in keys])
    with np.errstate(divide='ignore', invalid='ignore'):
        estimate = float(statistic(matrix.sum(axis=0)[None, :])[0])
    if not np.isfinite(estimate):
        return None, None, None, 0
    for i, k in enumerate(keys):
        strata.setdefault(k[0], []).append(i)
    rng = rng_for(seed, name)
    counts = np.zeros((resamples, len(keys)))
    rows = np.arange(resamples)[:, None]
    for members in strata.values():
        members = np.asarray(members)
        np.add.at(counts, (rows, members[rng.integers(0, len(members), (resamples, len(members)))]), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        draws = statistic(counts @ matrix)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return estimate, None, None, 0
    alpha = (1 - level) / 2
    return estimate, float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha)), len(draws)
