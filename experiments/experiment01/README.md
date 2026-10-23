# experiment01: one-video end-to-end VERA validation

**Purpose.** Check that the reusable VERA setup in `setup/` runs all three inference steps on one paper-referenced
video and leaves inspectable, reproducible artifacts. This is a **setup validation, not a research result**. It does
not reproduce the benchmark AUC/AP, and it says nothing about whether explanations help people.

| | |
|---|---|
| Video | UCF-Crime `RoadAccidents127_x264.mp4`, the example in VERA appendix B.5. It is the primary choice, not the fallback. |
| Model | `OpenGVLab/InternVL2-8B` @ `6fb9ad69…`, BF16, frozen, `eval()`, flash-attn |
| Source | VERA `15b8bcb8` + `setup/patches/0001-*.patch` → `008ed401` (branch `vera-research-local`) |
| Prompt | VERA learner instruction + the 5 released UCF-Crime guiding questions, verbatim. Frames only; no audio. |
| Sampling | anchor every 16 frames, window [a−150, a+150) clipped, 8 frames, 448×448, greedy, ≤1024 new tokens |
| Step 2 features | the authors' released ImageBind features for this video (one member of their zip, by HTTP range) |
| Config | [`config.yaml`](config.yaml) |

Each attempt gets its own `runs/run_NNN/` directory and is never overwritten. See `runs/run_*/REPORT.md` for results.

Rerun (on oz, from `~/vera_research`):

```bash
RUN=experiments/experiment01/runs/run_002   # next unused number; never reuse a run directory for a fresh attempt
bash setup/scripts/launch.sh --name exp01 --log-dir $RUN/logs -- \
  setup/environment/env/bin/python setup/scripts/run_experiment.py --config experiments/experiment01/config.yaml --run-dir $RUN --new-run
setup/environment/env/bin/python setup/scripts/validate_run.py --run-dir $RUN
```

To redo only the refinement on an existing run (no VLM calls; reuses `results/raw_responses.jsonl`):

```bash
setup/environment/env/bin/python setup/scripts/run_experiment.py --config experiments/experiment01/config.yaml \
  --run-dir experiments/experiment01/runs/run_001 --stages refine
```
