# experiment03B handoff

Snapshot: **2026-10-24, 11:30 UTC (07:30 EDT).** Run status and results in the table and §0 below. The §4 "running
now" text is the 2026-10-23 snapshot, kept for the record.

**In one line:** experiment03B checks whether VERA's two "most important" frames (ranked by a separate numerical
scorer under blur) flip VERA's own anomaly alert under grey replacement more often than a random pair of frames,
across all 290 UCF-Crime test videos.

| | State |
|---|---|
| Design, code, review, validation | **Done** (protocol v1.1, frozen) |
| Retrieval for the full run | **Done**: 433/433 inputs, 0 failures |
| Full GPU run `full_v1` | **Model stages done** 2026-10-24 09:15 UTC: 15.1 GPU-h, 15 h 05 min wall, 0 errors, 0 missing labels |
| Persistent video archive | **Done**: 290/290 archived, every sha256 matches the run (19:58 UTC) |
| Analysis | **Done** (`results/primary.json`, `secondary.json`, `table_a1.json`) |
| Independent validation | **FAILED 16/20** — all 4 failures are record-integrity checks explained by the data (§0.2); every recompute check passed |
| Report (`APPENDIX_RESULTS.md`) | **Skipped** by the runner because validation failed; awaits a decision (§0.3) |

---

## 0. Results of `full_v1` (2026-10-24)

### 0.1 Primary endpoint (protocol §7.1; population = original VERA alerts, development videos excluded)

| | Value |
|---|---|
| Inputs / videos | 199 alerts / 138 videos; 0 missing `d_w` |
| Ranked top-2 grey pair removes the alert | 26.8 % |
| Mean over all 28 grey pairs | 11.6 % |
| **Paired advantage `d_w`** | **+15.2 pp, 95 % CI [9.9, 20.6]** (video-cluster bootstrap, B = 10,000, seed 0) |
| Stability seeds 1–4 | CI endpoints within [9.90, 10.00] and [20.54, 20.74] |
| Sensitivities | tie-free +15.2 [9.9, 20.6] (no ties at the cut); scorer–VERA agreement +14.9 [9.0, 21.0] (147 inputs); with development +15.4 [10.2, 21.0]; metadata-clean +15.2 (no flags) |
| By stratum (descriptive) | event-centred +10.3 pp (126 alerts); annotation-free anomaly windows +19.3 (56); normal-video false alerts +28.8 (17) |

Secondary (430 confirmation inputs, 288 video IDs): grey numeric margin +0.356 nats [0.319, 0.396]; fixed grey
prefix label-change rate 12.7 / 15.3 / 17.4 / 20.4 % at k = 1..4, adaptive 12.7 / 17.0 / 21.2 / 25.9 % (adaptive
used 5,990 extra calls); blur/grey top-2 overlap 1.44, τ_b 0.61; native/S_hi top-2 overlap 1.35, τ_b 0.66; scorer
agrees with VERA on 333/410 baselines and 84 % of grey pairs; normal-video 0→1 artefact flips 2.7 % of grey pairs;
no-sound changes 28/430 labels; reciprocal check +0.364 nats [0.329, 0.403].

### 0.2 The four failed validator checks (diagnosed read-only from raw records)

| Check | Count | Cause | Effect on results |
|---|---|---|---|
| `score_rounding` | 45 of 38,934 score lines (40 inputs) | The FP32 head readout lies at most 0.00017 past a BF16 rounding midpoint, so it rounds to the neighbouring BF16 value. The engine's flag allows half a step + 1e-3 of FP32 summation-order slack; the validator demands exact round-to-nearest. | None on labels. `S_hi` is the declared readout. |
| `slot_hashes_present`, `slot_identity` | 20 lines, 2 inputs | Slot 0 of Explosion039_x264@112 and RoadAccidents128_x264@144 is frame 0, an all-black image (every pixel 0). Blurring black gives black, so the "replaced" slot is pixel-identical. | Blur effect of that slot is exactly 0, which is correct. Grey still changes it. |
| `mask_hash_consistency` | 88 sequence keys, 1 video pair | Normal_Videos_936_x264 and Normal_Videos_937_x264 are byte-identical files in UCF-Crime (same sha256 `9655db12…`), so their windows are identical and share cache entries. | Neither is in the primary population (both L0 = 0). Secondary analyses count this one video twice (288 IDs, 287 distinct). |

### 0.3 Decision needed

The code freeze forbids changing the validator inside `full_v1`. Options: (a) new run ID `full_v1_r1` that replays
all model outputs from cache (0 GPU) with a narrowly amended validator (engine tolerance, constant-frame no-op,
byte-identical source videos), disclosed as a post-data integrity-only amendment; or (b) keep `full_v1` as is and
report validation 16/20 with this diagnosis as a declared deviation.

---

## 1. Where everything is

| What | Where |
|---|---|
| Server | RIT Oz ("Narnia"): `ssh sp2117@oz.gccis.rit.edu` |
| Project (git checkout) | `~/vera_research` = `/home/stu11/s5/sp2117/vera_research`, branch `research/experiment03B-appendix` at `fb4d96d` |
| Mac mirror | `/Users/shrid/Documents/research/vera_research/` (copy only; **not** a git checkout) |
| Python | `setup/environment/env/bin/python` (on Oz) |
| Model weights | `shared/models/OpenGVLab/InternVL2-8B/6fb9ad69…` |
| VERA source | `setup/source/VERA` at `008ed401` (upstream base `15b8bcb8`) |
| Sparse frames used by the run | `shared/frames/ucf_crime_sparse/` (`<video>.frames.json` holds each video's `source_sha256`) |
| Run directory | `experiments/experiment03B/runs/full_v1/` |
| Job directory | `experiments/experiment03B/runs/full_v1_jobs/exp03b_full_20261023T141017_63447/` |
| Download scratch (purged per video) | `/local/datasets/sp2117/vera_research_scratch/experiment03B` |
| Persistent video archive | `/local/datasets/sp2117/ucf_crime_test_videos/` (mode 700) |

### Git and pull requests (repo `shridpant/VERA`, all drafts, stacked, none merged)

| PR | Branch | Content |
|---|---|---|
| [#1](https://github.com/shridpant/VERA/pull/1) | `research/validated-baseline-1-setup-experiment01` | Setup + experiment01 |
| [#2](https://github.com/shridpant/VERA/pull/2) | `research/validated-baseline-2-experiment02` | experiment02 |
| [#3](https://github.com/shridpant/VERA/pull/3) | `research/validated-baseline` | experiment02 erratum E1, pinned VERA source |
| [#4](https://github.com/shridpant/VERA/pull/4) | `research/experiment03A-protocol` | experiment03A 30-window protocol draft v0.4 (owned by the other session; not run) |
| [#5](https://github.com/shridpant/VERA/pull/5) | `research/experiment03B-appendix` | experiment03B (this study) |

**Pushing:** Oz has no GitHub credentials. Make a `git bundle` on Oz, copy it to the Mac, fetch it into a bare clone
there and push from the Mac (keychain user `shridpant`). Never force-push; do not merge.

**Not yet committed on Oz:** `runs/full_v1/`, `runs/full_v1_jobs/`, the new `shared/frames/ucf_crime_sparse/*.frames.json`
files, `setup/scripts/archive_test_videos.py`, and this file.

---

## 2. Experimental setup

### 2.1 Model and targets

- **Model:** InternVL2-8B, revision `6fb9ad6924f69424e57fab2ab061d707688f0296`, BF16, FlashAttention 2, greedy
  decoding, `max_new_tokens` 1024. VERA's released UCF guiding questions and learner prompt, unchanged.
- **Input unit:** one VERA 10-second window = **8 sampled frames**, resized by VERA's `dynamic_preprocess` to 448×448.
- **Three requests per input:**

| Request | What it gives | Role |
|---|---|---|
| `original_generation` | VERA's explanation + 0/1 label (strict parser) | **Primary outcome** |
| `nosound_generation` | same prompt minus guiding question 5 (sound) | baseline diagnostic only |
| `label_only_score` (`label_only_first_token_v1`) | `S_hi` = FP32 head-readout margin; native margin = `logit_1 − logit_0` (BF16) | `S_hi` builds the rankings; native is a sensitivity check |

The scorer is a separate prompt; its agreement with VERA's alert is measured, never assumed.

### 2.2 Population (manifest v1, sha256 `a8a209cf…`, frozen from metadata before any inference)

- **290 videos** = all UCF-Crime test videos: 140 anomalous (13 categories) + 150 normal.
- **452 window records:** 162 event-centred, 140 benchmark-negative (19 unavailable), 150 normal.
- **433 unique inputs**; 430 outside the development videos. 16 windows are clipped at a video edge. 3,464 frames.
- **Development videos** RoadAccidents127_x264 and Normal_Videos_018_x264 are excluded from the confirmation
  estimate (they were used in experiment02 and the integration check); they stay in coverage and one sensitivity
  estimate.

### 2.3 Replacements (applied to the original pixels each time; never cumulative)

| Family | Spec |
|---|---|
| `blur` | PIL `GaussianBlur(radius = 0.04 × 448 = 17.92)` |
| `grey` | constant RGB (127,127,127); same as experiment02's `constant`, so cache entries are shared |

### 2.4 Rankings and masks per input

- **Effect:** `D(G | R) = q·[S_hi(x_R) − S_hi(x_R∪G)]`, with `q = ±1` locked to the baseline VERA label `L0`.
- **Fixed ranking (primary):** blur single-frame effects on the unperturbed input, sorted once (ties → lower slot).
  Top two = `T_w`. Prefixes `R_k = {π_1..π_k}` are never re-ranked.
- **Adaptive ranking (secondary):** recompute every remaining slot's conditional blur effect after each removal, up
  to k = 4. It uses extra queries, so it is not an equal-cost competitor.
- **Masks evaluated:** all 8 singles and all **28 pairs in both families**; one fixed and one adaptive prefix at
  k = 3, 4. **No exhaustive triple (56) or quadruple (70) sweep.**
- **Calls per input (upper bound):** 88 scorer forwards + 35 VERA generations. Whole run: ≤38,104 forwards and
  ≤15,155 generations.

### 2.5 Primary endpoint

- For each input: `d_w = F(T_w) − mean over the 28 grey pairs of F(P)`, where `F = 1` if VERA's alert flips 1→0
  under that grey pair.
- Average within video, then across videos; report in percentage points.
- 95 % interval: video-cluster bootstrap, B = 10,000, `random.Random(seed)`, seed 0; stability seeds 1–4.
- Worst-case bounds for missing labels. Sensitivity estimates: tie-free, scorer–VERA agreement, with development
  videos, metadata-clean.
- **A null result is a valid appendix result** and is reported the same way.

### 2.6 Secondary analyses (descriptive)

Grey numerical margin; fixed vs adaptive prefixes at k = 1..4 (with extra-call accounting); blur/grey and
native/`S_hi` ranking agreement (top-two overlap, Kendall τ_b); scorer/VERA class agreement; pair non-additivity;
normal-video 0→1 artefact flips; per-category breakdowns; no-sound diagnostic; annotation coverage; reciprocal
check. Full definitions: `PROTOCOL.md` §7.2.

### 2.7 Workflow safeguards

- Stages: preflight → errata → manifest → retrieval → baseline → fixed → adaptive → evaluation → analysis →
  validation → report.
- One lock per run directory; `--resume` continues after any interruption.
- **Code freeze:** changing any code file (runner, analyzer, validator, table) requires a new run ID; model outputs
  are reused through `prior_caches`.
- Analysis is gated on completeness (override `--allow-partial` flags the results).
- Limits: 25 GPU-hours, 2,048 MiB scratch. A reached limit stops the run rather than truncating it.
- Errors are retried inside the request (≤3 attempts); parse failures are regenerated up to 2 more times.
- Cache keys equal experiment02's `request_key`, so experiment02's outputs are reused.
- An independent validator (`code/validate3b.py`, 20 checks) recomputes everything from raw records.

---

## 3. What is complete

| Item | Evidence |
|---|---|
| experiment01 (one-video reproduction) and experiment02 (3-window attribution feasibility), erratum E1 | PRs #1–#3; `RESEARCH_NOTES.md` |
| experiment03A protocol draft v0.4 (not run) | PR #4, other session |
| experiment03B protocol v1.0 → v1.1 after a 46-agent adversarial review (41 findings confirmed and fixed) | `AMENDMENTS.md`, `review/code_review_v1.0.json` |
| Runner, analyzer, table generator, independent validator | `code/`; CPU suite 17/17, validator tests 37/37 |
| Synthetic end-to-end run | `runs/synthetic_001`: validation 20/20 (synthetic, not results) |
| Bounded GPU integration (2 development windows) | `runs/integration_001`: 84/100 new inputs, ~0.08 GPU-h, fresh recompute byte-identical to experiment02; one validation failure (native margin) found and fixed |
| Integration replay after the fix | `runs/integration_002`: 0 model calls, validation 20/20, table generated (development only, not results) |
| Retrieval rehearsal | `runs/retrieval_rehearsal_001` (fidelity check passed) |
| **Full-run retrieval** | `full_v1` ledger: retrieval done 18:10 UTC; 433/433 ok, 0 failed, no metadata flags |
| Running config verified | Oz `config.json` and all 15 code files match commit `fb4d96d` |

---

## 4. What is running now

### 4.1 Full GPU run `full_v1`

- **Launched** 2026-10-23 14:10 EDT (18:10 UTC) via `launch.sh` on **GPU 4** (UUID `GPU-6a490dc2…`), RTX 6000 Ada,
  24 h wall-clock limit.
- **Command:** `run_workflow.py --config experiments/experiment03B/config.json --seeds 0 --resume` (all stages,
  both ranking modes, seed 0).
- **Progress at 18:45 UTC:** baseline stage, 1,165 request lines of ~1,299 (433 inputs × 3), 0.59 GPU-h,
  GPU at 98 %, 19.4 GB VRAM. No errors.
- **Expected finish:** ~12–15 h after launch, i.e. **about 2–5 am EDT on 2026-10-24**. Analysis, validation and the
  report run at the end of the same job.

Check it:

```bash
ssh sp2117@oz.gccis.rit.edu 'cd ~/vera_research; R=experiments/experiment03B/runs/full_v1; tail -3 $R/ledger.jsonl | cut -c1-200; wc -l $R/requests/*.jsonl; cat $R/gpu_time/*.json'
```

The job is finished when `full_v1_jobs/exp03b_full_20261023T141017_63447/exit_status` exists.

### 4.2 Persistent archive of the 290 test videos

- **Why:** the run purges each downloaded video after extracting its 8-frame windows; the archive keeps the source
  videos for reproducibility. It runs separately and only reads the run's files.
- **Script:** `setup/scripts/archive_test_videos.py` (uncommitted). It range-fetches each video from the official
  Dropbox zips, checks size and zip member against the manifest, and compares each sha256 with the
  `source_sha256` the run recorded. Resumable.
- **Outputs:** `/local/datasets/sp2117/ucf_crime_test_videos/<video>.mp4` + `.provenance.json`,
  `archive_ledger.jsonl`, `ARCHIVE_MANIFEST.tsv` (written at the end), `archive_run.log`.
- **Result (2026-10-23 19:58 UTC): done.** 290/290 archived, 7.96 GB (7.5 GiB on disk), and
  `ARCHIVE_MANIFEST.tsv` shows `matches_run` for all 290. The first pass got 289; Normal_Videos_935 (906 MB) hit
  Dropbox HTTP 503 three times and was fetched by a second pass (`archive_run2.log`). The first pass's table is
  kept as `ARCHIVE_MANIFEST.run1.tsv`.

---

## 5. What is left

1. ~~Rerun the archive for failed videos.~~ Done: all 290 verified. For reference, the resumable command (do not
   pass `--only`: it rewrites `ARCHIVE_MANIFEST.tsv` with just those rows):

   ```bash
   ssh sp2117@oz.gccis.rit.edu 'cd ~/vera_research && python3 -I -B setup/scripts/archive_test_videos.py --manifest experiments/experiment03B/manifest/manifest_v1.json --config experiments/experiment03B/config.json --frames-root shared/frames/ucf_crime_sparse --run-ledger experiments/experiment03B/runs/full_v1/ledger.jsonl --archive-dir /local/datasets/sp2117/ucf_crime_test_videos'
   ```

2. **Wait for `full_v1` to finish.** If it stops early (limit, device error, wall-clock), re-issue the same
   `launch.sh` command (§6); `--resume` continues from the cache. Do not change code in between (code freeze).
3. **Check the end of the run:** `exit_status` = 0; ledger shows `done` for every stage through `report`;
   `runs/full_v1/validation.json` reports 20/20; completeness gate passed without `--allow-partial`.
4. **Read the results:** primary `d_w` with its CI and bounds (`results/primary.json`), sensitivities, secondaries,
   and Table A1 (`results/secondary.json`, `results/table_a1.json`, `results/APPENDIX_RESULTS.md`). If the ranked pair does not beat the random-pair reference, report the
   null.
5. **Optional extra bootstrap seeds 1 and 2** (no new model calls; §6, step 3).
6. **Update the docs**, which still say "not launched": `README.md`, `IMPLEMENTATION_REPORT.md`,
   `RESEARCH_NOTES.md`, `EXPERIMENTS.md`. Add the archive location.
7. **Commit and push** the run records (respecting `.gitignore`: example images, `run.lock` and pid files stay
   out), the new frame identity files, `archive_test_videos.py` and this file; update PR #5. Copy to the Mac
   mirror.
8. **Possible extension: test all 56 triples and 70 quadruples per window** (not run; would need a new protocol version).
   Estimated from full_v1's speeds (0.359 s per scorer pass, 2.27 s per VERA answer, ×1.15 overhead):
   - blur score + grey score + grey VERA answer, all 433 windows: 23 GPU-h (triples) + 29 (quadruples) = **~52 GPU-h**;
   - the same on the 200 alert windows only: **~24 GPU-h**;
   - grey score + VERA answer only, all windows: ~46 GPU-h.

   That is about 2 days on one Ada, or ~13 h across GPUs 4–7 if the runner gains sharding. It needs ~0.6 GB in
   home and no downloads. A visual walkthrough of one window is in `walkthrough/explosion011_walkthrough.html`
   (dataset images; keep local).
9. **Later, outside this appendix:** the human pilot study; choose its clips by a recorded rule, not by which
   windows produced dramatic effects here.

---

## 6. Commands (run from `~/vera_research` on Oz)

```bash
PY=setup/environment/env/bin/python
# plan only (433 inputs, 14.21 h upper bound at experiment02 timings)
$PY -B experiments/experiment03B/code/run_workflow.py --config experiments/experiment03B/config.json --seeds 0 --dry-run
# 1. retrieval (done)
$PY -B experiments/experiment03B/code/run_workflow.py --config experiments/experiment03B/config.json --seeds 0 --resume --stages preflight errata manifest retrieval
# 2. model stages on one Ada (running); re-issue the same command to resume
bash setup/scripts/launch.sh --name exp03b_full --log-dir experiments/experiment03B/runs/full_v1_jobs --time-limit 86400 -- $PY -B experiments/experiment03B/code/run_workflow.py --config experiments/experiment03B/config.json --seeds 0 --resume
# 3. extra bootstrap seeds after the run
$PY -B experiments/experiment03B/code/run_workflow.py --config experiments/experiment03B/config.json --resume --stages analysis validation report --seeds 0 1 2
# CPU tests
$PY -B experiments/experiment03B/code/tests/run_tests.py
$PY -B experiments/experiment03B/code/tests/test_validate3b.py
```

---

## 7. Rules and pitfalls

- Don't touch the running experiment: no code edits, no second invocation on `full_v1` (one lock per run dir).
- Never kill other users' processes or look in their directories. GPUs are shared; pin by UUID.
- Don't clean up old experiments or caches; only this workflow's own scratch, by its ledger.
- Oz home quota is tight (~139/150 GB before the run); large data goes to `/local/datasets/sp2117`.
- Detached jobs over SSH: use `setsid nohup … < /dev/null &`, or the SSH session stays attached.
- Dropbox rate-limits large range fetches (HTTP 503); retry later rather than in a tight loop.
- Dataset-derived images (`runs/*/results/example_inputs/`) stay local, never in Git.
- Every model-run number from the integration runs is development-only, not a finding.

## 8. Key files

| File | What |
|---|---|
| `PROTOCOL.md` | The frozen protocol v1.1 (definitions, endpoint, analysis, reporting) |
| `AMENDMENTS.md` | v1.0 decisions and the v1.0 → v1.1 changes |
| `RECORDS.md` | Raw-record contract shared by runner, analyzer and validator |
| `IMPLEMENTATION_REPORT.md` | What was built and tested; cost projection; §5 launch commands |
| `config.json` | Full-run config (`run_id` `full_v1`) |
| `manifest/manifest_v1.json`, `manifest/windows_v1.csv` | The locked population |
| `code/run_workflow.py` | Runner (stages, resume, limits, cache) |
| `code/analyze3b.py`, `code/appendix_table.py` | Analyzer and Table A1 generator |
| `code/validate3b.py` | Independent validator |
| `../../EXPERIMENTS.md`, `../../RESEARCH_NOTES.md` | Project registry and running notes |
