# Training 1: VERA's verbalized learning on firearm categories and actions

The frozen InternVL2-8B learns its own guiding questions (VERA's `training.py` procedure) so that
its answers explain the scene, then name the category (Handgun / Machine_Gun / No_Gun), the action
group and the action code. No weights change and no boxes are used; boxes are Training 2.
Design: [../../docs/superpowers/specs/2026-10-26-firearm-vera-training1-design.md](../../docs/superpowers/specs/2026-10-26-firearm-vera-training1-design.md).

## Contents

- [Files](#files)
- [Commands](#commands)
- [Run folder](#run-folder)
- [Faithful vs changed](#faithful-vs-changed)

## Files

| File | Role |
|---|---|
| [codebook.py](codebook.py) → `codebook.json`, `prompts/` | categories, 6 action groups and 41 codes from the paper's Tables 1–3; the learner and optimizer prompts; seed questions |
| [prepare.py](prepare.py) | leak-free split (recording groups never span splits, every code in every split), label files, Training 2 box files |
| [training1.json](training1.json) | the experiment: model, generation, batch 2, 8 frames, 10 epochs, validation every 100 batches, seed |
| `../../vera_runner/verbalized.py` | learner, optimizer, VERA's question update, validation, selection, test, resume |
| [analysis/](analysis/), [report/](report/) | tables S1–S10, figures F1–F4, the LaTeX report |
| [jobs/](jobs/) | Oz smoke and full-run scripts |

## Commands

On Oz (`PY=~/vera_research/setup/environment/env/bin/python`, analysis: `AN=/local/datasets/sp2117/firearm_visilab/envs/analysis/bin/python`, `D=/local/datasets/sp2117/firearm_visilab`):

```bash
$PY run_firearm_training.py prepare --inventory $D/inventory.json --original $D/original/Gun_Action_Recognition_Dataset --out $D/training1/data
bash studies/firearm_training/jobs/smoke.sh training1_smoke1       # 2 steps, 4 val and 4 test videos
bash studies/firearm_training/jobs/train.sh                        # train (resumed until complete), then test
$AN run_firearm_training.py analyze --config studies/firearm_training/training1.json --out $D/training1/analysis
bash studies/firearm_training/report/build.sh $D/training1/analysis
```

Launch GPU jobs with `~/vera_research/setup/scripts/launch.sh --gpu <Ada UUID> -- bash <absolute script path>`.
Tests: `.venv/bin/python tests/run_tests.py --suite firearm`.

## Run folder

```
runs/<experiment_id>/
  spec.json           immutable: runner settings, training settings, prompt texts, codebook and split hashes
  steps.jsonl         every step: videos, frames, questions before/after, learner answers, predictions, targets,
                      optimizer prompt and answer, timing (failed attempts kept)
  validations.jsonl   step 0 (seed, reference) and every 100th batch: per-video answers and metrics
  test.jsonl          seed and selected questions on the test split
  sessions/NNN/       session.json (commit, GPU, model identity, library versions), environment.txt
```

## Faithful vs changed

- **Faithful to `training.py`** (checked by executing its `forward` and `training_step` on a fake model):
  - the prompt assembly;
  - batch_chat for 2 videos;
  - chat with 16 frames for the optimizer;
  - the five-line question update;
  - greedy, 1,024 tokens;
  - validation every 100 batches within an epoch, 10 epochs.
- **Changed:**
  - three categories plus actions instead of binary labels;
  - strict parsing (invalid = wrong);
  - neutral seed questions;
  - a seeded order;
  - a step-0 reference validation;
  - resumable logs;
  - selection on category, then action-group accuracy.
