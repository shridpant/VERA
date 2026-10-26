import unittest
import numpy as np
from studies.firearm.analysis.stats import cluster_bootstrap, cluster_bootstrap_sums


class FastBootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.units = [{'video': f'v{v}', 'category': c, 'a': int(rng.random() < p), 'b': int(rng.random() < p + 0.1)}
                      for c, p in (('X', 0.3), ('Y', 0.6)) for v in range(40) for _ in range(rng.integers(3, 12))]
        for u in self.units:
            u['video'] = u['category'] + u['video']

    def test_same_estimate_and_interval_as_the_per_draw_version(self):
        slow = cluster_bootstrap(self.units, 'video', 'category', lambda s: sum(u['b'] - u['a'] for u in s) / len(s), 3000, 7, 'slow')
        fast = cluster_bootstrap_sums(self.units, 'video', 'category', lambda u: (u['b'] - u['a'], 1), lambda t: t[:, 0] / t[:, 1], 3000, 7, 'fast')
        self.assertAlmostEqual(slow[0], fast[0], places=12)
        self.assertLess(abs(slow[1] - fast[1]), 0.015)
        self.assertLess(abs(slow[2] - fast[2]), 0.015)

    def test_reproducible_and_undefined_statistics(self):
        args = (self.units, 'video', 'category', lambda u: (u['a'], 1), lambda t: t[:, 0] / t[:, 1], 500, 11, 'r')
        self.assertEqual(cluster_bootstrap_sums(*args), cluster_bootstrap_sums(*args))
        none = cluster_bootstrap_sums(self.units, 'video', 'category', lambda u: (0, 0), lambda t: t[:, 0] / t[:, 1], 100, 1, 'z')
        self.assertEqual(none, (None, None, None, 0))
