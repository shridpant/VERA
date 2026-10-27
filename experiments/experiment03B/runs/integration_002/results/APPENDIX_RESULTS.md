# experiment03B appendix results — run `integration_002`

Generated from `results/table_a1.json`, `results/primary.json` and `results/secondary.json` by `code/appendix_table.py`. Protocol v1.1. Engine: `real`.

## Planned, executed and failed

- Inputs in this run: 2 (status: ok 2).
- Baseline original alerts: 2/2; baseline labels missing: 0.
- Stages failed in the ledger: none.
- Independent validation: PASSED (20 checks passed, 0 failed)

## Primary estimate (confirmation set)

No eligible confirmation input: NA. (The confirmation set excludes the development videos; see the sensitivity row.)

| Sensitivity | Estimate, pp | 95% CI | N videos | N inputs |
|---|---:|---:|---:|---:|
| tie_free | NA | NA | 0 | 0 |
| scorer_agreement | NA | NA | 0 | 0 |
| with_development | 3.6 | 3.6 to 3.6 | 1 | 2 |
| metadata_clean | NA | NA | 0 | 0 |

## Table A1. Prespecified appendix summary

| Population | N videos | N windows | N alerts | N complete | Top-2 flip % | Random-pair flip % | Paired advantage, pp (95% CI) | Mean top-2 overlap, blur/grey |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **All original alerts, confirmation set** | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Event-centered windows | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Annotation-free anomaly-video windows | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Normal-video windows | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Abuse | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Arrest | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Arson | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Assault | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Burglary | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Explosion | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Fighting | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| RoadAccidents | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Robbery | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Shooting | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Shoplifting | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Stealing | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Vandalism | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Sensitivity: including development videos | 1 | 2 | 2 | 2 | 50.0 | 46.4 | 3.6 (descriptive) | 2.00 |

*Table note.* model revision: 6fb9ad6924f69424e57fab2ab061d707688f0296; source head: 008ed4019781c39e7cb16244b7500244f3986f92; prompts: VERA learner instruct VERA_learner_instruct.txt at 008ed401; released guiding questions sha256 0650ebf87422; no-sound = question 5 removed; question sha256 by target {'label_only_score': ['fbda861cbe61'], 'nosound_generation': ['d3d101a142fd'], 'original_generation': ['f3d5e7c9894c']}; scorer: label_only_first_token_v1; rankings on S_hi; native margin = logit_1 - logit_0; replacements: construction blur (PIL GaussianBlur radius 17.92 at 448x448); evaluation grey RGB(127, 127, 127); excluded development videos: ['RoadAccidents127_x264', 'Normal_Videos_018_x264']; complete case: 0 of 0 confirmation alert inputs have d_w (0 videos); 0 missing; missing bounds pp: [None, None]; bootstrap: seed 0, 10000 video-cluster resamples; weighting: flip percentages and advantages are video-weighted (within-video mean, then mean over videos); engine: real; run id: integration_002; run flags: {'partial': False, 'skipped_budget_lines': 0, 'error_lines': 0, 'record_conflicts': 0}; analyzer sha256: 0231997bbfa3. `NA` rows give no eligible complete alert; category and stratum rows are descriptive, not separate hypothesis tests.

NA reasons: All original alerts, confirmation set: no selected window; Event-centered windows: no selected window; Annotation-free anomaly-video windows: no selected window; Normal-video windows: no selected window; Abuse: no selected window; Arrest: no selected window; Arson: no selected window; Assault: no selected window; Burglary: no selected window; Explosion: no selected window; Fighting: no selected window; RoadAccidents: no selected window; Robbery: no selected window; Shooting: no selected window; Shoplifting: no selected window; Stealing: no selected window; Vandalism: no selected window.

## Fixed versus adaptive grey prefixes (secondary)

| Mode, k | Label-change rate (video-weighted) | Mean q·ΔS_hi | Inputs | Videos | Attributable new calls |
|---|---:|---:|---:|---:|---:|
| fixed_k1 | 0.500 | 1.088 | 2 | 1 | 2 |
| fixed_k2 | 0.500 | 1.718 | 2 | 1 | 0 |
| fixed_k3 | 0.500 | 2.363 | 2 | 1 | 4 |
| fixed_k4 | 1.000 | 2.300 | 2 | 1 | 4 |
| adaptive_k1 | 0.500 | 1.088 | 2 | 1 | 0 |
| adaptive_k2 | 0.500 | 1.718 | 2 | 1 | 0 |
| adaptive_k3 | 0.500 | 2.331 | 2 | 1 | 14 |
| adaptive_k4 | 1.000 | 2.630 | 2 | 1 | 14 |

Prompts used: `results/prompts/` (exact question strings for the original, no-sound and label-only targets, the no-sound diff, and a comparison with experiment02 run_002).

Secondary scope: `all`. Companion files: `results/inputs_metrics.csv`, `results/pairs.csv`, `results/trajectories.csv` (native-BF16 sensitivity and fixed/adaptive k = 1..4), `results/secondary.json`.

Regenerate: `setup/environment/env/bin/python experiments/experiment03B/code/run_workflow.py --config <this run's frozen config> --run-id integration_002 --resume --stages analysis validation report --seeds 0`

