# Firearm A/G/AG study

Firearm localization and its effect on VERA's anomaly judgments, on the 398-video VISILAB
firearm-action dataset with frozen InternVL2-8B. **Results:** [RESULTS.md](RESULTS.md) (report: `results/firearm_a_g_ag_v1/report.pdf`). Specification: [HANDOFF.md](HANDOFF.md).
Plan: [../../docs/superpowers/plans/2026-10-25-firearm-a-g-ag-study.md](../../docs/superpowers/plans/2026-10-25-firearm-a-g-ag-study.md).

## Contents

- [Conditions](#conditions)
- [Files](#files)
- [Commands](#commands)
- [Run folder](#run-folder)
- [Knobs](#knobs)
- [Records](#records)

## Conditions

| ID | Prompt | Output |
|---|---|---|
| A | VERA's prompt, byte-identical to `run_vera.py` | anomaly label and explanation |
| G | grounding only ([prompts/](prompts/)) | eight `FrameK:` box lines on a 0–1000 grid |
| AG | A plus two declared insertions (grounding as question 6, box lines before the final `Output:`) | box lines, explanation and label |

- All three conditions send the same eight images through the same `batch_chat` call.
- InternVL2 packs all eight at `Frame1: <image>`; the other seven `<image>` strings stay text.
- Each call is fresh, with no history.
- The prompts are locked by sha256 (version v2). Their history is in [PROMPT_LOG.md](PROMPT_LOG.md).

## Files

| File | Role |
|---|---|
| [experiment_A_G_AG.annotated.yaml](experiment_A_G_AG.annotated.yaml) | every knob, commented and tagged scientific/operational |
| `experiment_A_G_AG.json`, `.runner_A.json` | `convert` output; the second is a plain `run_vera.py` config (condition A) |
| [AUDIT.md](AUDIT.md), [annotation_policy.json](annotation_policy.json) | label audit: origin 1, frame statuses, size bins |
| [PROMPT_LOG.md](PROMPT_LOG.md) | prompt versions, pilot evidence, probe, lock |
| `../../run_firearm_study.py` | all commands below |
| `../../vera_runner/{study,ledger,conditions,slots,localization,firearm_data,firearm_annotations,imagebind_features}.py` | run, resume, prompts, parsers, metrics, data, features |
| `analysis/`, `report/` | tables, figures, numbers; the LaTeX report |
| `jobs/` | Oz job scripts (pilot, probe, full run) |

## Commands

On Oz (`PY=~/vera_research/setup/environment/env/bin/python`, ImageBind:
`IB=/local/datasets/sp2117/firearm_visilab/envs/imagebind/bin/python`, `D=/local/datasets/sp2117/firearm_visilab`):

```bash
$PY run_firearm_study.py prepare --original $D/original/Gun_Action_Recognition_Dataset --data-root $D
$PY run_firearm_study.py audit --inventory $D/inventory.json --original $D/original/Gun_Action_Recognition_Dataset --out $D/audit
$PY run_firearm_study.py convert --yaml studies/firearm/experiment_A_G_AG.annotated.yaml --out studies/firearm/experiment_A_G_AG.json
$PY run_firearm_study.py validate --config studies/firearm/experiment_A_G_AG.json
bash studies/firearm/jobs/pilot.sh firearm_pilot_v2          # smoke: 6 videos x 2 windows, slot check, A vs run_vera.py
bash studies/firearm/jobs/full_run.sh                        # run + resume until every unit has a disposition
$PY run_firearm_study.py reconcile --config studies/firearm/experiment_A_G_AG.json
$IB run_firearm_study.py features-check --config studies/firearm/experiment_A_G_AG.json --video <UCF mp4> --released <npy> --labels <segment_scores.json> --out <dir>
$IB run_firearm_study.py features --config studies/firearm/experiment_A_G_AG.json
$PY run_firearm_study.py refine --config studies/firearm/experiment_A_G_AG.json
$PY run_firearm_study.py analyze --config studies/firearm/experiment_A_G_AG.json --original $D/original/Gun_Action_Recognition_Dataset --out $D/analysis --features-check <dir>/features_check.json
bash studies/firearm/report/build.sh $D/analysis             # tables/figures -> report/, pdflatex + bibtex, source zip
```

Launch GPU steps with `~/vera_research/setup/scripts/launch.sh --gpu <Ada UUID>` and an absolute
script path. Run one GPU process at a time.

Tests: `.venv/bin/python tests/run_tests.py --suite firearm` (mock backend, synthetic videos).

## Run folder

```
runs/<study_id>/
  spec.json            immutable: runner config, prompts and hashes, selection, inventory hash (read-only)
  prompts/{A,G,AG}.txt the exact prompt texts
  ledger.jsonl         start/end of every attempt (complete, failed, timeout, interrupted, preprocess_failed)
  records/{A,G,AG}.jsonl  every attempt's full record, append-only
  sessions/NNN/        session.json (commit, GPU, model hash), status.json heartbeat, environment.txt, git_diff.patch
  videos/<id>/         video.json, windows.json, frames/, inputs/<anchor>/{00..07.png,hashes.json}
  input_ids/<sha>.json token ids sent to the language model (deduplicated)
  model_input_{A,G,AG}.txt  first rendered model input per condition
  reconcile.json       every scheduled unit with its disposition
  refined/             refinement.csv, feature_hashes.json, {A,AG}/<id>/{scene,temporal,frame_scores}.npy
```

A unit is skipped on resume only when a complete record matches the config, model, prompt and
input hashes. A changed spec is refused.

## Knobs

Every knob is in the annotated YAML.

- **Scientific:** prompt version and order, frame count, stride, window, nominal FPS, resolution,
  precision, attention, generation cap, coordinate grid, IoU thresholds, refinement and bootstrap.
- **Operational:** paths, timeouts, attempts and GPU.

## Records

| Field | Meaning |
|---|---|
| `label` | strict final-Output label (A, AG). None when invalid or truncated; never 0 |
| `strict_label` | the runner's rule (exactly one Output field) |
| `upstream_label` | VERA's original rule |
| `grounding` | slots (G, AG): `boxes`, `empty`, `unknown`, `malformed`, `missing`, `duplicate` or `truncated`; boxes on 0–1000; whole-image flags |
| other | `eos`, `truncated`, `generated_tokens`, `seconds`, `peak_gpu_bytes`, `rgb_sha256`, `hashes`, `attempt`, `session` |
