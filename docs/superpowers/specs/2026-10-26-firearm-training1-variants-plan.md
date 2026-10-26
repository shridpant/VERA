# Training 1: replications and variants, overnight plan

Declared 2026-10-26 00:20 ET, before any replication or variant result existed. The user approved
the plan in chat: GPUs 6 and 7, until about 12:30 ET on 26 October, exploration and depth,
reproducible runs. Methods: [`vera_runner/verbalized_variants.py`](../../../vera_runner/verbalized_variants.py);
prior art: [`studies/firearm_training/LITERATURE.md`](../../../studies/firearm_training/LITERATURE.md).

## Contents

- [Why](#why)
- [Runs](#runs)
- [Decision rules](#decision-rules)
- [Report](#report)

## Why

The faithful run's logs (GPU 5, about step 590 of 1,200) show three failure modes:

- the optimizer chases the last batch: it rewrote the questions at every step (279 distinct sets in
  593 steps);
- it has no memory of which sets scored well;
- "describe…" questions trigger greedy repetition loops that hit the 1,024-token cap (12, 21 and
  17 of 62 validation answers at steps 340, 460 and 580).

Validation category accuracy of its learned questions so far: 38.7–46.8 %. The seed questions
score 54.8 %.

## Runs

Each run trains (resumed until complete), then runs the seed and the selected questions once on
the test split (`jobs/train.sh`).

| GPU | Order | Config | Run seed |
|---|---|---|---|
| 5 | running since 25 Oct 20:56 | `training1.json` (faithful) | 20261026 |
| 6 | 1 | `training1_seed2.json` (faithful) | 20261027 |
| 6 | 2 | smoke, then `training1c_bank.json` | 20261026 |
| 6 | 3 | `training1c_bank_random.json` (ablation) | 20261026 |
| 6 | 4 | `training1c_bank_noscore.json` (ablation) | 20261026 |
| 7 | 1 | `training1_seed3.json` (faithful) | 20261028 |
| 7 | 2 | smoke, then `training1b_fixed.json` | 20261026 |
| 6, 7 | last | depth runs, see below | 20261027, 20261028 |

The run seed sets every random choice of a run: the training order, the fixed variant's check set
and batches, and the bank's pair, video and scoring draws.

## Decision rules

1. **Within a run**, selection is unchanged:
   - the highest validation category accuracy;
   - then action-group accuracy;
   - then the earliest step.
2. **Depth-run candidates** are `vera_fixed` and `bank`. The ablations explain the bank; they are
   not candidates.
3. **The winner** is the candidate with:
   - the higher selected validation category accuracy;
   - then the higher action-group accuracy;
   - then fewer training GPU hours.

   It is decided from `validations.jsonl` alone, once both candidates have finished training.
4. **The winner is successful** if its selected validation category accuracy is above the seed
   questions' (54.8 %, 34 of 62).
5. **Depth runs** form a queue: the winner's `_seed2` and `_seed3` configs, then the runner-up's.
   - At the end of its chain, each GPU claims the next unclaimed run (`jobs/depth.sh`; amended
     00:25 ET, still before any variant result).
   - The decision waits for both candidates until 10:30 ET, then ranks the finished ones. No depth
     run starts after 10:30 ET; a run started before then finishes, past 12:30 ET if needed, and
     is reported when it ends.
   - Depth runs happen even if the winner is not successful. They then measure the spread of a
     method that did not beat the seed questions.
6. **Test results of the variants are not read** before the depth decision. `test.jsonl` is written
   at the end of each run; the comparison is run once all runs have finished.
7. **No run is stopped for its scores.** A run is stopped and replaced only if it crashes or its
   smoke run fails; failures are reported.
8. **Validation selects among several methods** on the same 62 videos, so the winner's validation
   score is optimistic. Claims rest on:
   - the paired test comparison with the seed questions (recording-group bootstrap);
   - the replications.

## Report

`run_firearm_training.py compare` writes tables C1–C4 and figures G1–G3. They form the section
"Variants and replications" of the Training 1 report (`report/compare.tex`).
