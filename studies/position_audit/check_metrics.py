"""Independent tests against explicit duplication/sklearn, including score ties."""
import numpy as np
from sklearn.metrics import roc_auc_score
from metrics import CachedAUC, gate
from reconstruct import labels

rng = np.random.default_rng(13)
for _ in range(30):
    video = np.repeat(np.arange(4), [7, 11, 5, 9])
    y = rng.integers(0, 2, len(video)); s = rng.integers(0, 5, len(video))/4
    counts = rng.integers(1, 4, 4)
    cached = CachedAUC(y, s, video).auc(counts)
    ix = np.repeat(np.arange(len(y)), counts[video])
    expected = roc_auc_score(y[ix], s[ix])
    assert abs(cached-expected) < 1e-12, (cached, expected)
    assert CachedAUC(y, np.ones(len(y)), video).auc(counts) == .5
    assert abs(CachedAUC(np.roll(y, 3), np.roll(s, 3), np.roll(video, 3)).auc(counts)-cached) < 1e-12
assert labels({'length': 5, 'temporal_label': [2, 7, -1, -1]}).tolist() == [0, 0, 1, 1, 1]
assert gate([.89, .84], [.88, .88], .001)['pass']
assert not gate([.9, .89], [.9, .9], .001)['pass']
print('30 explicit-bootstrap/sklearn/tie/shift cases and boundary/gate checks passed')
