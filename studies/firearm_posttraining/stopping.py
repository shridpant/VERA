"""Serializable primary-metric patience; best checkpoint and anchor are distinct."""
from dataclasses import dataclass, asdict


@dataclass
class StopState:
    anchor: float = -1.
    best: tuple = (-1., -1., -1.)
    stale: int = 0
    checks: int = 0
    best_checkpoint: str = ''
    reason: str = ''

    def observe(self, metrics, checkpoint, passes):
        key = tuple(metrics[k] for k in ('balanced_category_accuracy', 'action_group_accuracy', 'action_code_accuracy'))
        improved = key > tuple(self.best)
        if improved:
            self.best, self.best_checkpoint = key, checkpoint
        if key[0] >= self.anchor + .005:
            self.anchor, self.stale = key[0], 0
        else:
            self.stale += 1
        # Baseline (passes==0) is a candidate but does not use a post-update query.
        if passes > 0:
            self.checks += 1
        if passes >= 1 and self.stale >= 3:
            self.reason = 'patience'
        elif self.checks >= 10:
            self.reason = 'maximum_validation_checks'
        return improved

    def snapshot(self):
        return asdict(self)
