# experiment02: temporal_attribution_feasibility

**Question.** Can we measure which of VERA's 8 input frames its local judgment depends on? And does replacing
neighbouring frames together reveal effects that one-at-a-time replacement misses? Spatial heatmaps were a separate,
bounded question.

**Spec.** [`BRIEF_v2.md`](BRIEF_v2.md) (the v2 brief; the earlier 32-frame design was superseded and not run).

**Config.** [`config.yaml`](config.yaml) (v2; the v1 copy is frozen in `runs/run_001/resolved_config.yaml`).

**Code.** [`code/`](code). It is experiment-local and reuses the setup loaders without changing them:
- `run_attr.py`: the runner (stages prep / A / B / C);
- `engine.py`: the model, generation and the numeric scorer;
- `attr_lib.py`: masks, perturbations, cache keys and effects;
- `analyze_attr.py`, `validate_attr.py`, `spatial_probe.py`, `spatial_border_check.py`, `test_attr_lib.py`.

| Run | What | Status |
|---|---|---|
| `runs/run_001` | Stage A under v1 rules | stopped: numeric scorer blocked by an exact bf16 tie; kept as the pre-declared outcome |
| `runs/run_002` | Stages A–D under v2 (post-hoc changes disclosed) | complete; validator passes; see [`runs/run_002/REPORT.md`](runs/run_002/REPORT.md) |

**Decisions (run_002).**

| Area | Decision |
|---|---|
| Original binary temporal sensitivity | CONDITIONAL GO |
| Numeric temporal attribution | CONDITIONAL GO (provisional) |
| Spatial heatmaps | CONDITIONAL GO for technical validation only; NO-GO for participants |

**Errata.** See [`ERRATA.md`](ERRATA.md). E1 (2026-10-23): a change in S_hi of less than one bf16 step can still
move the deployed bf16 score. The native differences (`Delta_bf16`) are what to read alongside S_hi.

Next step: [`runs/run_002/NEXT_STEP.md`](runs/run_002/NEXT_STEP.md).
