# CPU-only position audit: completed, practical gate failed

No GPU, model inference, weight training, verbalized learning or new human annotation was used.

## Primary result

All 290 released feature arrays passed CRC, shape, finite-value and anchor checks. Independent reconstruction exactly matched all 1,111,808 scores and labels from unmodified upstream code.

| Agreed criterion | Measured | Pass |
|---|---:|---|
| Original AUC spread >=3 pp | 3.728 pp | Yes |
| Worst-position benefit >=3 pp | 2.721 pp | **No** |
| Paired 95% CI lower bound >0 | [1.307, 4.371] pp | Yes |
| Unshifted AUC cost <=1.5 pp | 1.007 pp | Yes |

Original weighting gives 86.607% AUC at zero shift and 82.879% at its worst fixed shift. No-position scores give 85.600% at every shift. This demonstrates positional dependence in the final multiplier, not end-to-end failure on edited videos. No-position invariance is mathematically expected.

All four retrieval/endpoint settings fail the same practical-benefit requirement: improvements range from 2.699 to 2.758 pp. The positive confidence interval does not override the predeclared 3 pp effect threshold. The paper's 10% retrieval sensitivity reproduces 86.548% AUC, which rounds to its published 86.55%.

The secondary AP tradeoff matters: original-order AP is 32.745% with weighting and 28.717% without it. The no-position method is not uniformly better.

## Reproducibility and fixes

Protocol819ba57, initial implementation/input freeze3da8543. Pre-performance relative-path launch failure preserved and corrected by absolute CLI paths (bc7e03f). Upstream completed but independent replay exposed short-video convolution truncation; amendment3cce2c8 matches upstream's first-N loop. No thresholds or scientific choices changed. Both failures remain preserved. Successful replay22.257seconds; full statistics206.368seconds, four CPU BLAS threads, 1,000 paired video resamples in each of four cohorts. All580 extracted input files rehashed unchanged after the run. Raw vectors are external with committed hashes; metrics, bootstraps, diagnostics and figures are committed.

## Novelty and stopping decision

Removing weighting is already a VERA ablation; CoReVAD also ablates it. The candidate contribution was a controlled practical robustness finding, not a new removal algorithm. This audit measures such dependence but misses the agreed go/no-go threshold. Novelty/priority is not established, and the result is not proof that the effect is zero. Stop further compute in this direction; no automatic GPU or XD extension. Do not lower the threshold after observing 2.721 pp. Preserve this negative resource-allocation decision and the measurable empirical result.
