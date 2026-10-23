# experiment01 / run_002: fresh rerun (reproducibility check)

**Status: COMPLETE.** This directory checks that the documented command reruns from the retained shared caches
(model, video, frames, features). It ran the documented command with `--run-dir experiments/experiment01/runs/run_002
--new-run`; the source revision, config and environment were the same as in run_001.

- **Job:** `exp01_run002_20261022T155919_14733`, 15:59:20 to 16:05:34 EDT on 2026-10-22, exit 0.
- **GPU:** RTX 6000 Ada (GPU 4), the same GPU as run_001.
- **Inference was fresh.** All 162 windows were generated again; the frame cache was reused after its identity checks.
- **Validation:** `validate_run.py` passes 15/15.
- **Same result as run_001.** Comparing all 162 records:
  - raw responses: **162/162 identical**;
  - labels: identical;
  - model token inputs: identical;
  - frame scores: identical (`np.array_equal`).

  Greedy decoding with this stack (torch 2.1.2, flash-attn 2.5.8, transformers 4.37.2, one RTX 6000 Ada) was
  therefore deterministic for this video.

Findings, examples and deviations: see [`../run_001/REPORT.md`](../run_001/REPORT.md). This run adds no new findings.

Reproduce:

```bash
RUN=experiments/experiment01/runs/run_003
bash setup/scripts/launch.sh --name exp01 --log-dir $RUN/logs -- \
  setup/environment/env/bin/python setup/scripts/run_experiment.py --config experiments/experiment01/config.yaml --run-dir $RUN --new-run
setup/environment/env/bin/python setup/scripts/validate_run.py --run-dir $RUN
```
