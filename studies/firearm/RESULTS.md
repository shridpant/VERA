# A/G/AG results (run firearm_a_g_ag_v1)

The verified results of the study in [HANDOFF.md](HANDOFF.md). Every number below comes from
`results/firearm_a_g_ag_v1/numbers.json`, which `run_firearm_study.py analyze` wrote from the run's saved records. The full report is
[results/firearm_a_g_ag_v1/report.pdf](results/firearm_a_g_ag_v1/report.pdf) (Overleaf source:
`report_source.zip`).

## Contents

- [Run](#run)
- [Findings](#findings)
- [Completion against the handoff](#completion-against-the-handoff)
- [Where things are](#where-things-are)

## Run

- **Coverage:** 19,125 window × condition units; 19,125 complete, none failed, timed out or retried.
- **Session:** one, on one RTX 6000 Ada (GPU-6a490dc2), finished 2026-10-26 05:58 EDT.
- **Spec:** sha256 `aa84eaf71552f9e5ec352b1e78b62687964d4bbbe6f97ad5532dc02bdae08f7f`.
- **Model:** sha256 `54410957a4159f5e6eb96231b15aa536dd68f813bdd1f8dfd94b75e0e75d0255` (InternVL2-8B 6fb9ad69, BF16, FlashAttention, greedy, 1,024 tokens).
- **Prompts:** v2, locked by hash (see [PROMPT_LOG.md](PROMPT_LOG.md)).

## Findings

1. **Unchanged VERA (A)** labels 76.3 % of windows anomalous.

   | Category | Anomalous windows |
   |---|---|
   | Handgun | 80.4 % |
   | Machine_Gun | 87.9 % |
   | No_Gun | 57.2 % |

   These are judgments, not accuracy: the categories are not anomaly labels.
2. **Localization:** under VERA's eight-in-one packing, the frozen backbone does not localize firearms.
   - **G:** `[]` on 50,804 of 51,000 slots.
   - **AG:** `[]` on 50,960 slots.
   - **Boxes written:** all are whole-image (195 in G, 39 in AG).
   - **Recall** against 23,841 annotated firearms is 0 at IoU 0.50 and 0.75, for both G and AG.
3. **Asking for boxes in the same answer (AG)** changes the anomaly judgments.
   - **Paired windows:** on 6,370 windows valid in both conditions, AG − A = -3.8 pp.
   - **Intervals:** 95 % CI -5.2 to -2.3 resampling videos; -5.4 to -2.2 resampling recording groups.
   - **Refined scores:** on the 274 videos refined in both conditions, the per-video mean refined score changes by -0.027 (-0.039 to -0.014).
   - **Interpretation:** AG almost never localized, so this is the effect of a bundled prompt change, not of localization.
4. **Parsers:** the strict, runner-strict and original parsers agree on all A answers. 5 AG answers have no valid label; the original rule would have called them normal.
5. **Refinement:**
   - 278 videos are refined with A labels and 274 with AG labels.
   - 120 videos have fewer than 15 windows and are excluded, not padded.
   - 4 more videos are excluded for AG, each for an invalid AG label.
   - ImageBind features match VERA's released features with median row cosine 0.9949, not bit for bit ([FEATURES_VALIDATION.md](FEATURES_VALIDATION.md)).
6. **Compute:** inference took 3.25, 2.23 and 4.54 GPU-hours for A, G and AG.

## Completion against the handoff

| Section 10 requirement | State |
|---|---|
| every inventory item and scheduled condition accounted for | 19,125/19,125 units complete (`reconcile_summary.json`) |
| original A preserved | `run_vera.py` and study A byte-identical on 12 pilot windows, in 3 separate runs |
| valid outputs and failures traceable | `records/*.jsonl`, `ledger.jsonl`, `sessions/` on Oz (no failures occurred) |
| T1–T11, P1–P9 | all produced; plus T2b (parser agreement), T8d/T8s/T8o (paired differences, size, origin 0), P7b |
| no invented ground truth or metrics | no anomaly accuracy/AUC/AP; no mAP; categories never used as anomaly truth |
| report compiles | `report.pdf` (26 pages), built on Oz with `report/build.sh`; source in `report_source.zip` |
| exact configurations and commands | [experiment_A_G_AG.annotated.yaml](experiment_A_G_AG.annotated.yaml), `spec.json` on Oz, [README.md](README.md) commands |
| independent review before full execution | [REVIEW.md](REVIEW.md) (an AI review, not a human signature) |

## Where things are

- **Committed here:** `results/firearm_a_g_ag_v1/`.
  - every table as CSV;
  - the numbers;
  - the qualitative cases;
  - the refinement summary;
  - the report PDF and source.
- **On Oz:** `/local/datasets/sp2117/firearm_visilab/` holds the complete run (35 GB, frames and inputs included), the features and the full analysis folder (`analysis/`, with figures and window sheets).
- **On the Mac:** a copy of the analysis folder is in `~/Documents/research/firearm_a_g_ag_results/`.
