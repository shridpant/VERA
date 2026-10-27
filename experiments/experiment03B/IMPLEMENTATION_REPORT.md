# Current status — October 26, 2026

Experiment03B full_v1 completed: 290 videos / 433 inputs. Original primary +15.166 pp [9.860, 20.600]. Original validation 16/20 remains preserved; the explicit post-data R1 integrity audit passes 20/20 without model reruns or changed results. See [R1 amendment](R1_AMENDMENT.md). The earlier not-launched/planned sections below are historical.

---

# experiment03B implementation report (2026-10-23)

**Status: protocol v1.1 is frozen.**
- **Built.** The workflow is implemented and validated on CPU fixtures, an independent validator, an adversarial
  review and a bounded GPU integration check on two development windows.
- **Not launched.** The 290-video run has not been launched. No dataset-wide or confirmation result exists.
- **Development numbers.** Every number below that comes from a model run is development-only. None of them is a
  finding.

## 1. What was built

| Component | File | Role |
|---|---|---|
| Protocol | `PROTOCOL.md` v1.1 | Population, readouts, fixed and adaptive definitions, endpoint, analysis, reporting, workflow. Includes the fixed-ranking statement verbatim. |
| Record contract | `RECORDS.md` v1.1 | The interface between the runner, the analyzer and the independent validator |
| Manifest builder | `code/manifest3b.py` → `manifest/manifest_v1.json` (sha256 `a8a209cf…`) | 452 window records and 433 unique inputs, from metadata only |
| Core | `code/xb_core.py` | Effects, fixed ranking with ties, adaptive trajectory, `d_w` and its bounds, Kendall τ_b, the video-cluster bootstrap |
| Runner | `code/run_workflow.py` | The stage DAG, dry-run, resume, lock, ledger, limits, retries, cache layers, completeness gates |
| Engine adapter | `code/engine3b.py` | experiment02's engine unchanged (identical prompts and cache keys), plus an analytic mock for tests |
| Retrieval | `code/acquire3b.py` | Preflight; per-member range fetch; sparse extraction with VERA's decode loop; fidelity gate; purge |
| Analyzer | `code/analyze3b.py` | Everything recomputed from raw records: primary, sensitivities, secondaries and Table A1 |
| Table generator | `code/appendix_table.py` | `APPENDIX_RESULTS.md` from the analyzer outputs only |
| Independent validator | `code/validate3b.py` | Written separately, from `PROTOCOL.md` and `RECORDS.md` only, using the standard library. 20 checks. |
| Tests | `code/tests/run_tests.py`, `code/tests/test_validate3b.py`, `code/tests/independent_manifest.py` | CPU suite (17 tests), validator self-tests (37), independent manifest re-derivation |

## 2. Implemented / tested / untested

| Behaviour (brief's requirement) | Implemented | Tested | Evidence |
|---|---|---|---|
| 1 Known additive scorer, signs for both targets | yes | CPU | `t_additive` |
| 2 Interacting scorer: fixed `[0,2,1,…]` vs adaptive `[0,1,2,…]` | yes | CPU | `t_interaction`; validator fixture; synthetic run |
| 3 Fixed combined masks from original pixels; no hidden re-ranking | yes | CPU (real pixel path) + GPU | `t_pixels` (slot hashes of {3}, {4}, {3,4}); `t_end_to_end`; validator `slot_identity` |
| 4 Adaptive audit (conditional baselines, no reselection, q locked) | yes | CPU + GPU | `t_adaptive_audit`; validator `adaptive_invariants` on integration_002 |
| 5 Time adjacency by original slots | yes | CPU | `t_adjacency` |
| 6 Cross-family isolation | yes | CPU | `t_end_to_end` (grey override); validator `cross_family_isolation` |
| 7 Ties, missing values, NaN | yes | CPU | `t_ties_missing`, `t_retry`, missing-label synthetic input |
| 8 BF16 crossing; E1 reproduced from raw records | yes | CPU | `t_bf16`; validator `erratum_e1` (36 of 109) |
| 9 Inputs, cache keys, scheduled vs unique calls | yes | CPU + GPU | `t_keys` (3B keys equal experiment02's); integration counts |
| 10 Workflow: DAG, dry-run purity, crash and resume, limits, immutability, seeds | yes | CPU | `t_config`, `t_dry_run`, `t_end_to_end` (hard crash, then resume), `t_budget_stop`, `t_torn`, `t_partial_gate`, `t_immutable` |
| 11 Independent analyzer and validator | yes | CPU + GPU | validator 20/20 on synthetic_001 and on integration_002 |
| Window manifest rules | yes | CPU | independent re-derivation: 452/452 records identical |
| Fetch path (range fetch, identity, sparse extraction, purge) | yes | network + CPU on one development video | `runs/retrieval_rehearsal_001` |
| Fetch of non-development videos, metadata-mismatch flags on real data, very long videos | yes | **untested** | first exercised by the full run |
| Parallel GPUs on one run directory | **not supported** | — | one exclusive lock per run (§9) |

**Test runs on Oz** (project interpreter, `CUDA_VISIBLE_DEVICES=` empty, so CPU only):

```bash
setup/environment/env/bin/python -B experiments/experiment03B/code/tests/run_tests.py      # 17/17 passed
setup/environment/env/bin/python -B experiments/experiment03B/code/tests/test_validate3b.py # Ran 37 tests: OK
```

## 3. Review and independence

- **Adversarial review of v1.0** (46 agents, four lenses, one skeptic per finding). 41 findings were confirmed and
  1 refuted (`review/code_review_v1.0.json`). Every confirmed finding was fixed before any model output, which made
  v1.1 (`AMENDMENTS.md`). The main fixes:
  - a reached limit now stops the run instead of truncating it silently;
  - errors are retried inside the request;
  - the validation and report pipeline unblocked;
  - completeness gates before analysis;
  - the code freeze extended to the analysis code;
  - the native-margin definition;
  - hardened retrieval;
  - corrected denominators and scope.
- **Independent validator.** It was written by a separate agent from `PROTOCOL.md` and `RECORDS.md` only, then
  adversarially checked; that check fixed 5 validator defects (`review/independent_validator_build.json`). The only
  later edit was the v1.1 native-margin rule and its two pin tests.

## 4. Bounded integration check

**Cases.** experiment02's R-accident (anchor 2240) and R-pedestrian (anchor 496), both in the development video
RoadAccidents127, with seed 0 on one RTX 6000 Ada (GPU 4, pinned by UUID through `setup/scripts/launch.sh`).

| Run | What happened |
|---|---|
| `integration_001` (GPU) | **Limits:** 84 new transformed inputs (limit 100); 297.5 GPU-seconds, of which the model load was 170 s (limit 0.5 h); wall time 5 min 15 s; scratch peak about 8 MB (limit 500 MiB).<br>**Calls:** 119 fresh model calls (85 forwards, 36 generations at the engine counter) and 129 served from experiment02's cache. All stages through `analysis` completed.<br>**Checks passed:** a fresh recomputation of one cached score and one cached generation is byte-identical to experiment02's records (S, S_hi, logits, raw text, label, slot and input hashes); all three prompt strings are identical to experiment02's; the sparse frames are byte-identical to the full VERA cache (fetched Normal_Videos_018, sha256 matches).<br>**Independent validation failed 1 of 20 checks** (`rankings_compare`). The runner's `derived/rankings.jsonl` used the stored `S` field for the native ranking, which differs by about 1e-7 from the v1.1 margin `logit_1 − logit_0` that the analyzer and validator use. No model output or analyzer result was affected. |
| Fix | The runner now writes `logit_1 − logit_0`. The mock engine stores a noisy `S`, as the real engine does, so the end-to-end test would catch a regression. The code freeze then required a new run ID. |
| `integration_002` (GPU pinned, model never loaded) | Same cases, with `integration_001`'s forward cache as a prior layer. All 250 requests were served from caches (0 model calls, 0 new inputs, 29 s). Independent validation passed 20/20, and the report generated `results/APPENDIX_RESULTS.md`. |
| `retrieval_rehearsal_001` (CPU and network) | The fetch path on Normal_Videos_018 (`prefer_local: false`): fidelity check, member and byte-size check against the manifest, sparse extraction of the manifest window (anchor 592), identity file, and scratch purge. All 8 frames are byte-identical to the full VERA cache. |
| Code provenance | `integration_002` and `synthetic_001` ran exactly the committed code (their `frozen/code_sha256.json` matches). `integration_001` and `retrieval_rehearsal_001` ran the earlier revision: only `run_workflow.py` (the native-margin fix) and `engine3b.py` (the mock's noisy `S`) differ. The retrieval module `acquire3b.py` is identical. |
| `synthetic_001` (mock engine) | The table pipeline on 11 synthetic inputs in 9 videos with hand-worked values. The primary estimate is 42.857 pp, matching the hand-worked 3/7. Validation passed 20/20. The table carries a "SYNTHETIC — NOT A RESULT" banner. |

**Development observations** (integration_002; not results; development videos are outside the confirmation set):

| | R-accident (2240) | R-pedestrian (496) |
|---|---|---|
| Baseline original label `L0` / no-sound label | 1 / 1 | 1 / **0** (prompt-dependent alert, as in experiment02) |
| Baseline `S_hi` / native margin | 2.855 / 2.75 | −0.131 / 0.00 (exact BF16 tie) |
| Fixed blur order; `T_w` | 4 3 5 1 6 2 7 0; {3,4} | 7 4 5 0 2 6 3 1; {4,7} |
| Adaptive order | 4 3 5 7 (differs from fixed at k = 4) | 7 4 0 1 (differs at k = 3) |
| Grey pairs that change the label | 1/28; `T_w` does not | 25/28; `T_w` does |
| `d_w` | −0.036 | +0.107 |
| Blur/grey top-two overlap, τ_b | 2, 0.86 | 2, 0.64 |
| Native/S_hi top-two overlap, τ_b | 2, 0.87 | 1, 0.48 |
| Scorer class agrees with VERA on grey pairs | 27/28 | 17/19 (9 exact native ties are undecided) |

The confirmation row is NA, as designed. The development-inclusive sensitivity estimate (3.6 pp, 2 inputs, 1 video)
only shows that the code path runs.

## 5. Full run: locked counts, projection, resources

**Locked manifest v1** (sha256 `a8a209cf…`):
- 290 videos: 140 anomalous and 150 normal, in 13 categories.
- 162 intervals.
- 452 window records: 162 event-centred, 140 negative (19 of them unavailable) and 150 normal.
- 433 unique inputs, 430 of them outside the development videos.
- 16 clipped windows.
- 3,464 frames needed.

**Calls (upper bound, no cache overlap).** 88 forwards and 35 generations per input give 38,104 forwards and 15,155
generations in total, on at most 37,671 new transformed inputs. Cache sharing between the fixed and adaptive
prefixes lowers this slightly.

**GPU time.**
- At the measured integration rates (0.346 s per forward, 1.93 s per generation): about **11.8 core GPU-hours**.
- At experiment02's rates (0.356 s and 2.48 s): **14.2**.
- Plus about 3 minutes of model loading per invocation and any retries.
- On one RTX 6000 Ada, plan for **roughly 12–15 hours of elapsed GPU time**.
- One run directory runs one invocation at a time, so four GPUs do not shorten it unless the run is split into
  separate run IDs, which v1.1 does not merge.
- The config's safety cap is 25 GPU-hours.

**Downloads.**
- 7.96 GB from the official zip members, one video at a time, into `/local/datasets/sp2117/vera_research_scratch/experiment03B`
  (local NVMe, 784 GB free).
- Peak scratch is the largest member (906 MB, Normal_Videos_935); each video is purged after extraction.
- This is CPU and network time (decoding is sequential over every frame of each video). Run it first, without a GPU.

**Home storage** (quota 139/150 GB used):
- about 140 MB of sparse frames (3,464 JPEGs at about 40 KB);
- about 0.3 GB of run records (measured at 2 KB per request line and 1.7 KB per cached call).

**Commands.** The form of each was tested in the integration or the rehearsal; nothing below has been run on the
full config.

```bash
cd ~/vera_research
PY=setup/environment/env/bin/python
# 0. plan only (tested on the full config: 433 inputs, 14.21 h upper bound at experiment02 timings)
$PY -B experiments/experiment03B/code/run_workflow.py --config experiments/experiment03B/config.json --seeds 0 --dry-run
# 1. retrieval first, CPU/network only (form tested by retrieval_rehearsal_001)
$PY -B experiments/experiment03B/code/run_workflow.py --config experiments/experiment03B/config.json --seeds 0 --resume \
    --stages preflight errata manifest retrieval
# 2. model stages on one pinned Ada; re-issue the same command to resume after any interruption
bash setup/scripts/launch.sh --name exp03b_full --log-dir experiments/experiment03B/runs/full_v1_jobs --time-limit 86400 -- \
    $PY -B experiments/experiment03B/code/run_workflow.py --config experiments/experiment03B/config.json --seeds 0 --resume
# optional: fixed mode first, adaptive later (the adaptive stage reuses every shared forward)
#   ... run_workflow.py --config ... --seeds 0 --resume --ranking-mode fixed
#   ... run_workflow.py --config ... --seeds 0 --resume --ranking-mode adaptive
# 3. extra bootstrap seeds (deterministic model outputs are shared; seeds change resamples only)
$PY -B experiments/experiment03B/code/run_workflow.py --config experiments/experiment03B/config.json --resume \
    --stages analysis validation report --seeds 0 1 2
```

**Seeds.** The plan lists seeds [0, 1, 2]; the default and only executed seed is 0. Any further non-negative seed can
be passed with `--seeds`. Extra seeds write `results/seeds/primary_seed<s>.json`, and the stability seeds `s+1…s+4`
are always reported. Seeds are not model replications.

## 6. Caveats

- **What passing checks mean.** They establish implementation conformance, not that the rankings are faithful.
- **What the integration covered.** Two windows of one development video. It exercised the whole real-model path but
  says nothing about the population result. Fetching non-development videos, real metadata flags and very long
  videos are first exercised by the full run.
- **Display-only replacement examples.** The images in `runs/*/results/example_inputs/` are dataset-derived and stay
  local (ignored by Git).
