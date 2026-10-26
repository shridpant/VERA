# Training 1: VERA's verbalized learning on the VISILAB firearm dataset

Agreed in chat on 2026-10-26. The user's decisions are marked (user); the rest are my calls,
marked (mine).

## Contents

- [Intent](#intent)
- [Decisions](#decisions)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation and report](#evaluation-and-report)
- [Out of scope](#out-of-scope)

## Intent

Adapt VERA to the firearm dataset the way its authors adapt it to a new dataset: the frozen
InternVL2-8B learns its own guiding questions from labelled training videos (verbalized
learning), and those questions are selected on validation. Each answer explains what the model
saw and why (VERA's answers to its questions, as on UCF-Crime), then gives the category and the
action. Boxes come later, in Training 2, supervised by the dataset's boxes.

## Decisions

- Label = the three dataset categories Handgun / Machine_Gun / No_Gun (user).
- Action output = both a coarse group and the exact code; the coarse group is the main score and
  the exact code is exploratory (user, on my recommendation).
  - Coarse groups: carry, aim (firearm videos; reload variants have aim codes) and move, phone,
    bottle, heavy_object (No_Gun) (mine, from the paper's tables).
  - 41 exact codes exist in the data. PAW7 (kneel while aiming) is in the paper but has no
    video, so it is left out of the label set.
- Training 1 is VERA's method only: frozen model, the optimizer rewrites at most five guiding
  questions, no weight updates and no boxes (user).
- One fixed train/val/test split (user): about 60/15/25 by videos, stratified by action code,
  with every recording group (repeated takes) in one split (mine). A subject-held-out split is
  impossible: Handgun has subjects V1–V2 only.
- Category and action descriptions are drafted from the dataset paper's tables 1–3 (user asked
  me to draft them). They live in `studies/firearm_training/codebook.json`.
- Run on one RTX 6000 Ada, separate from the A/G/AG study's GPU (user). The work goes on branch
  `experiment/firearm-vera-training1` off `feature/configurable-runner`, with a PR into it
  (user).

## Preprocessing

- Inventory and staging come from the cherry-picked data commits (`firearm_data`, the audit).
- Split: per action code, seeded shuffle of recording groups, assigned greedily to the split
  furthest below its video target, with at least one group in each split.
  - Output `splits.json` (immutable, hashed).
  - Report the camera, subject and location mix of each split.
- Sampling: VERA's training sampling for every split — 8 frames at
  `np.arange(0, n, n/8).astype(int)` over the whole video (videos are 5–18 s, about one window).
  - Only those frames are decoded, with OpenCV and the default JPEG, exactly as the runner's
    preprocessing.
  - Then the runner's 448×448 PIL resize and InternVL2 transform.
- Label files per split: id, frames, fps, category, coarse action, exact code, attributes.
  Training 1 reads no boxes. Box files for Training 2 (audited origin 1, 0–1000 grid) are made
  separately.

## Training

Faithful to `training.py`; differences are listed in the report:

- learner = `batch_chat` on a batch of 2 videos, 8 frames each, greedy, 1,024 tokens, BF16,
  FlashAttention;
- optimizer = `chat` with both videos' 16 frames and VERA's optimizer prompt, adapted from binary
  labels to [category, action group, code] predictions and targets;
- new questions are read as VERA reads them (the text after "New Prompt Questions", at most five
  lines);
- validation every 100 training batches within an epoch, 10 epochs, shuffled with a fixed seed.

Changes (mine):

- the learner and optimizer prompts describe three categories, the action groups and the codes;
- seed questions are neutral for this task;
- strict parsing (an invalid field is wrong, never a default);
- the seed questions are also validated at step 0, for reference only;
- the run is resumable from its step log.

Selection: the highest validation category accuracy, then coarse-action accuracy, then the
earliest step (mine; VERA selects on validation accuracy).

## Evaluation and report

Test split, video level:

- the seed questions and the selected questions, each run once;
- category accuracy and macro-F1, coarse-action accuracy, exact-code accuracy and invalid
  outputs;
- the paired learned − seed difference with a recording-group bootstrap;
- confusion tables, per-camera breakdowns, the training trajectory, the learned questions,
  qualitative explanations chosen by a fixed rule, and compute.

The report is LaTeX, built like the A/G/AG report.

## Out of scope

- Boxes and Training 2.
- Weight updates or LoRA.
- VERA's temporal score refinement: it is defined for one binary anomaly score.
- New annotation.
