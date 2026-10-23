# Runner tests

All tests, test documentation, test configs and reports live under `tests/`.
Run from the `VERA-configurable` checkout using Python 3.11 with requirements-runner.txt
installed. On this Mac the existing interpreter is
`../VERA/experiments/vanilla/.env/bin/python`.

## Categories

| Folder / suite | What it checks | Uses trained model weights? |
|---|---|---|
| `configuration/` | Unknown config fields are rejected | No |
| `datasets/` | Named dataset/split lookup, manifest membership, duplicate ID errors | No |
| `preprocessing/` | Window sampling, OpenCV/JPEG byte equivalence, resized RGB equality against original VERA | No |
| `backends/` | Exact prompt equality with original forward using a fake model | No |
| `interventions/` | Baseline identity, exact blur pixels, exact grey pixels | No |
| `parsing/` | Valid labels and refusal to invent labels for invalid responses | No |
| `refinement/` | Numerical equality against unmodified compute_auc.py on synthetic data | No |
| `integration/` | Synthetic video → mock predictions → probe outputs and refinement/evaluation | No; explicitly mock |
| `runtime/` | Prior-process guard, including stopped processes with recent heartbeats | No |
| `real_model/` | Real UCF clip → InternVL2-8B → one complete parsed response and saved inputs/token IDs/logs | Yes; explicit opt-in |
| `reports/` | Historical and timestamped test reports, logs and pass/fail/skip counts | Evidence, not tests |
| `artifacts/` | Ignored numbered real-model runs, their exact configs and output files | Local artifacts |

The fake/mock checks validate implementation and plumbing. They do not establish that
real model inference succeeds, fits memory, reproduces CUDA output, or yields accurate
anomaly predictions. A skipped or blocked real-model test is NOT a pass.

## Run the local automated suite

```bash
../VERA/experiments/vanilla/.env/bin/python tests/run_tests.py --suite all
```

This runs the categories and explicitly skips real-model acceptance. A timestamped
`tests/reports/all/<timestamp>/tests.log` and `summary.json` are saved, with the skip
count shown separately. Select one category with e.g. `--suite preprocessing` or
`--suite datasets`. Standard `python -m unittest discover -s tests -v` also works;
it defaults to skipping real-model acceptance.

## Real-model acceptance

1. Copy `real_model/config.example.json` to `real_model/config.local.json` and configure
   a verified checkpoint path, dataset registry and one available test video. This
   machine's local config already points to the staged InternVL2-8B and RoadAccidents011.
2. Ensure the previous 8B inference run is stopped or complete. The local config checks
   its recorded PID directly. A stopped process is allowed immediately, even if its heartbeat is recent; an existing process or inaccessible process state blocks a second model.
   The test never kills the old process or guesses it is dead.
3. Run in regular Terminal, where MPS is accessible:

```bash
../VERA/experiments/vanilla/.env/bin/python tests/run_tests.py --suite real-model
```

The real config uses the original prompt, eight resized frames and greedy generation,
with BF16/MPS and explicit CPU fallback for unsupported MPS operators. The test is
bounded to one original window and a 300-second runner wall limit; there is no
reduction to one token or change to another backbone just to pass the test.

Pass criteria: a completed nonsynthetic inference run, exactly one raw response, a
strict 0/1 label, captured input token IDs and all eight saved input images. Label
correctness against the video is not a pass criterion of this execution smoke test.
It does not test full-video inference or real feature refinement/evaluation. Those
remain separate acceptance work after this first successful real-model request.

Logs are in `tests/reports/real-model/<timestamp>/`. Subprocess console output is
also saved to `tests/reports/real_model/last_console.log`. Actual run artifacts,
including `diagnostic.log`, are under `tests/artifacts/real_model/run N/output/`.
A hardware/precondition failure can occur before a run folder is created; in that
case the test traceback and reason remain in the timestamped test report.

Do not commit model weights, local machine paths, source video frames or generated
run artifacts. Reports are local/ignored by default; historical reports already
tracked are retained explicitly as historical evidence. No push is part of testing.
