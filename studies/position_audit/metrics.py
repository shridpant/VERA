"""Tie-aware pooled AUC with cached sort order and whole-video weights."""
import numpy as np


class CachedAUC:
    def __init__(self, y, scores, video):
        order = np.argsort(scores)
        s = scores[order]
        self.y = y[order].astype(bool)
        self.video = video[order]
        self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1]

    def auc(self, multiplicity):
        w = multiplicity[self.video]
        p = np.add.reduceat(w*self.y, self.starts).astype(np.float64)
        n = np.add.reduceat(w*~self.y, self.starts).astype(np.float64)
        den = p.sum()*n.sum()
        return float(np.sum(p*(np.cumsum(n)-0.5*n))/den) if den else float('nan')


def gate(original, no_position, ci_lower):
    spread = float(max(original)-min(original))
    benefit = float(min(no_position)-min(original))
    cost = float(original[0]-no_position[0])
    checks = {'spread_at_least_3pp': spread >= .03, 'worst_benefit_at_least_3pp': benefit >= .03,
              'paired_ci_above_zero': ci_lower > 0, 'unshifted_cost_at_most_1p5pp': cost <= .015}
    return {'spread': spread, 'worst_position_benefit': benefit, 'unshifted_cost': cost, 'checks': checks, 'pass': all(checks.values())}
