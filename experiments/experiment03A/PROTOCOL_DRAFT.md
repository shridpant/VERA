# experiment03A: interval-level perturbation study (PROTOCOL DRAFT v0.4)

**Status: DRAFT. Not frozen, not approved, not run.** No model has been called and no video downloaded for this
experiment. The only inputs read so far are metadata (§3).

**History.**
- v0.1 was reviewed by a 41-agent adversarial workflow (`review/protocol_review_round1.json`). It confirmed 37
  findings and refuted none.
- v0.2 answered each of them (`review/RESPONSE_TO_REVIEW.md`, round 1).
- v0.2 was reviewed by a round-2 workflow with two lenses, completeness and executability
  (`review/protocol_review_round2.json`). Skeptics confirmed 14 findings (2 critical, 12 major) and refuted 1; 22 minor
  findings were reported without verification.
  - The critical finding, raised by both lenses: v0.2's primary estimand averaged over all adjacent groups. An adjacency
    effect confined to the event slots therefore could not reach SUPPORTED and could be reported as absent.
- v0.3 answered every round-2 finding (`RESPONSE_TO_REVIEW.md`, round 2).
- v0.3 was checked by a bounded round-3 workflow with two lenses, resolution and consistency
  (`review/protocol_review_round3.json`). Skeptics confirmed 7 findings (1 critical, 6 major) and refuted none; 19 minor
  findings were reported. Almost all concerned the exact specification of the §13 fixture gates. v0.4 answers them
  (`RESPONSE_TO_REVIEW.md`, round 3).
- **Design evidence:**
  - `review/round2_design_sim.py`, a stdlib simulation of D1 under null and planted-effect fixtures, with its outputs;
  - `review/round3_power_gate.py`, the 1,000-draw reference run of the power gate, with its outputs (§13).

**Name.** This study was called experiment03 until 2026-10-23 and is now experiment03A. experiment03B, an appendix
validation across all 290 UCF test videos, builds on it (`../experiment03B/`). Seed and key strings keep the original
`experiment03-` prefix (§3, §4 C, §13.2) and must not be changed, because they determine the selection and fixture
outputs.

**What happens next.**
- **Approval.** The user must approve the budget (§10) and this design, knowing the expected outcome (§15): D1 will
  probably be INSEPARABLE or INCONCLUSIVE, because evidence concentrated in consecutive slots cannot be matched by
  non-consecutive slots.
- **Freeze F1, before any download:** the protocol, selection outputs, all code and the fixtures.
- **Freeze F2, before the first model call:** the config with video hashes and the final window list.

**Builds on** experiment02 (`../experiment02/runs/run_002/REPORT.md`, `NEXT_STEP.md`). Its two videos (RoadAccidents127,
Normal_Videos_018) formed the hypotheses and are **held out**.

## 1. Questions

| # | Question | Outcome (§7) |
|---|---|---|
| Q1 | How often does replacing input frames change VERA's original local label (or move the auxiliary score by ≥ 1 nat)? Asked for true alerts, alerts outside the annotation, and normal windows. | O1-bin, O1-score |
| **Q2 (primary)** | In true-alert windows with measurable single-frame evidence, does replacing the window's **strongest consecutive sampled slots** (its peak adjacent pair and triple) change the auxiliary score more than replacing **non-consecutive slots whose single-frame effects match one by one** and remove at least as much evidence? | O2 (decides D1); O3 consistency |
| Q3 | For alerts with rankable single-frame evidence, do the top-2 attribution-ranked frames remove more support than an average pair? Rankings are built in one replacement family and evaluated in the other. | O4 |
| Q4 | Do the top-2 frames agree across replacement families? | O5 |
| Q5 (exploratory) | Do alerts outside the annotation flip under more replacements than true alerts in the same video, and is that more informative than baseline margin? | O6 |

**Definitions and the licensed claim.**
- **"Consecutive" (adjacent)** means slots i and i+1 of VERA's 8 sampled frames: 37.5 frames (1.25 s) apart, and
  also consecutive 256-token image blocks in the prompt.
- **Peak groups.** For a window and family, the adjacent pair and the adjacent triple with the largest summed
  single-frame effect m(G) (§4). They are fixed from closed B1 singles before any B2 mask runs.
- **Why the peaks.**
  - In experiment02 every label-flipping adjacent group was a high-evidence group.
  - A display built from D1 (D3) shows the strongest interval.
  - An effect confined to the event slots is diluted below any useful threshold when it is averaged over all 13
    adjacent groups (§13.3).
- **Mechanism.** A positive Q2 licenses "non-additivity specific to the strongest consecutive sampled slots, beyond
  one-by-one matched single-frame evidence". It does **not** identify a mechanism, which could be temporal context,
  token position or visual redundancy (§15).
- **Not decided by D1.** Redundancy among weak neighbouring frames is not a peak-group effect: two frames that each
  change little can change a lot together. It is reported by the all-group secondary O2-all and the redundancy subset
  (§7), and never decides D1.

## 2. Fixed from experiment02 (pre-registered here, not post hoc)

| Element | Setting |
|---|---|
| Model and source | InternVL2-8B @ `6fb9ad69`, BF16, FlashAttention 2. VERA checkout `008ed401` (base `15b8bcb8` + experiment01 patch). |
| Target 1 (primary label) | VERA's original prompt (learner instruction + 5 released UCF questions, verbatim). Greedy decoding, `max_new_tokens` 1024; InternVL adds eos 92542. Strict parser; unparseable means missing, never a label. |
| Target 2 (auxiliary score) | Contract `label_only_first_token_v1`. These sections of experiment02's `scoring_contract.md` apply verbatim: "Three targets", "Prompts", "How S is computed", "S_hi" and "Effects" (`Delta = S(original) − S(perturbed)`). Its "Measured in Stage A" and "Gate changes in v2" do **not** apply; they are superseded by §9 D5. Its statement that S_hi differences below 0.25 cannot change the deployed bf16 output probabilities is wrong and does not apply (next row). Config `post_hoc_amendment: false`. |
| Effect scale | `S_hi` (fp32 readout of the same bf16 computation); `S` (bf16) is reported alongside every S_hi quantity. Resolution = bf16 spacing of the candidate logits (0.25 at |logit| ∈ [32, 64)). The two scales can disagree by a full step, because an S_hi change below 0.25 can cross a bf16 rounding boundary. In experiment02, 36 of the 109 effects with |S_hi Δ| < 0.25 changed bf16 S, 4 of them in the opposite direction. For example, R-accident blur slot 0 moved S_hi by 0.131 but bf16 S by a full step (both negative); opposite signs occurred in R-pedestrian blur slot 3 (S_hi Δ −0.115, bf16 Δ +0.25) and N-middle blur slot 3 (−0.045, +0.25). |
| Preprocessing and replacements | VERA `dynamic_preprocess` → 448×448, then the perturbation, then normalisation. **blur**: PIL GaussianBlur σ = 17.92 px (Pillow radius = σ). **constant**: RGB (127, 127, 127). Blur is the **primary** family; constant is secondary. |
| Direction | `c = 2·L0 − 1`, from the **fresh** original-prompt baseline label L0. One direction rule everywhere (O1, O2, B2 matching, C rankings, O4, O5). Windows with missing L0 are excluded from direction-dependent quantities. The scorer class (S_hi ≥ 0 → 1) is reported but sets no direction. |
| Arithmetic | All analysis arithmetic is float64 on the stored S_hi records, never on CSV text. **With a 1e-9 tolerance:** composition `D ≤ 0.25 + 1e-9`; caliper `m(H) ≥ m(G) − 1e-9`; "above resolution" `e > 0.25 + 1e-9` (T, rankability, O2-evid, the evidence fixture), and its complement `e ≤ 0.25 + 1e-9` (redundancy subset). **Exact:** every other comparison, that is, the D1 thresholds (p < 0.05, HL ≥ 0.25, upper bound < 0.25), O1-score (c·Δ ≥ 1.0), D4 (HL ≥ 0.2) and all orderings (peak argmax, G* argmin), whose ties are exact equality broken by the slot tuple. |
| Out of scope | VERA's refined timeline, heatmaps, ImageBind, training and humans |

Code status against experiment02:
- `engine.py`: byte-identical, sha256 `7bb6ae95…`, checked by the validator.
- `attr_lib.py` (experiment02 sha256 `7acf4acf…`): model-path functions unchanged (`slot_tensors`, `apply_family`,
  `family_spec`, `sequence_key`, `request_key`, the parsers); a §14 test compares them with experiment02's source.
- New and changed modules are listed in §14.

## 3. Windows (`selection/`: `windows.json`, `backup_windows.json`, `selection_log.json`, `windows.csv`, `CHECKS.md`)

**Generation.** `selection/select_windows.py` produces the list deterministically, using the standard library only.

- **Inputs (metadata only, all hashed in `selection_log.json`):**
  - `Data/UCF_Eval.json` from VERA `008ed401`. Its sha256 is `f3857cdd…`. The §14 validator checks that it is
    identical at the base `15b8bcb8` (`git show` at both commits).
  - The authors' released Step-1 labels: `VERA_prediction_scores_UCF_Crime.zip`, sha256 `b7457cee…`, 655,612 B.
  - The five official UCF-Crime zip directory listings.
- **Ordering:** ascending sha256 hexdigest of `"experiment03-20261023:" + key`. The key templates are `class:{class}`,
  `S1video:{video}`, `S2:{video}:{anchor}` and `S3video:{video}`.
- **Reproducibility checks** (`CHECKS.md`, which records the sha256 of the script and of every output checked):
  - every frame list and window bound equals VERA's numpy arithmetic (133/133 records: 30 windows and 103 backups);
  - all four outputs are byte-identical under Python 3.13.5 (oz) and 3.14.2 (Mac);
  - the replacement tests pass on both hosts;
  - `windows.json` and `windows.csv` are byte-identical to v0.2.

**Common rules.**
- A window is anchor `a ∈ range(0, n, 16)`, with frames `[a−150, a+150)` **unclipped** (a−150 ≥ 0, a+150 ≤ n), and
  VERA's 8 frames.
- Intervals are the manifest's `(start, min(end, n))` pairs that are not −1.
- Exclusions:
  - experiment02's videos are held out;
  - videos over 100 MB are excluded (12);
  - videos with no unclipped anchor are excluded (7).

| Stratum | n | Rule (as implemented) |
|---|---|---|
| **S1 true alerts** | 10 | **Joint eligibility:** the anomaly video has both an S1-eligible and an S2-eligible anchor. **S1-eligible anchor:** released label exactly 1, unclipped, inside an annotated interval. **Choice:** maximise Σ over intervals of the overlap between window and interval; ties go to the smallest distance from the anchor to any interval midpoint, then the smallest anchor. **Classes:** slot k (k = 1…10) belongs to the k-th class in class-key order; within a class, the first jointly eligible video in video-key order. |
| **S2 alerts outside the annotation** | 10 | **Always the S1 video** (the script fails rather than pair across videos). **Eligible:** released label 1, unclipped, window ≥ 150 frames (5 s) from every interval. **Preference:** side `pre` (window ends ≥ 150 frames before the first interval) first, then other sides; hash order within a side. These are *benchmark-negative alerts*. Whether each is a semantic error is not verified. |
| **S3 normal** | 10 | First 10 normal videos in hash order. **Anchor:** the unclipped anchor closest to n/2, ties to the earlier. Label-agnostic. |

The 30 windows (details in `windows.csv`):
- *Slots in annotation* are the 8-frame slot indices inside an annotated interval (`annotated_slots`).
- **Localized** (`localized_event`, a pre-registered moderator): some annotated interval lies wholly inside the window
  and contains the anchor, **and** 2–4 of the 8 slots are annotated. This is the definition in `select_windows.py`; it
  marks S1-01, S1-03 and S1-07.
- *Released 1-share* is the share of the video's anchors with released label 1.

| Pair | Video | S1 anchor / window (s) | Slots in annotation | S2 anchor / window (s) / side | Released 1-share |
|---|---|---|---|---|---|
| 01 | RoadAccidents001 | 256 / 3.5–13.5 | 3,4,5 (localized) | 1168 / 33.9–43.9 / post | **1.00** |
| 02 | Vandalism015 | 2352 / 73.4–83.4 | 0–7 | 448 / 9.9–19.9 / pre | 0.79 |
| 03 | Shoplifting020 | 2400 / 75.0–85.0 | 3,4,5 (localized) | 624 / 15.8–25.8 / pre | 0.38 |
| 04 | Burglary035 | 928 / 25.9–35.9 | 0–7 | 3536 / 112.9–122.9 / post | 0.54 |
| 05 | Fighting003 | 2464 / 77.1–87.1 | 0–7 | 1040 / 29.7–39.7 / pre | 0.56 |
| 06 | Arson041 | 2880 / 91.0–101.0 | 0–7 | 528 / 12.6–22.6 / pre | 0.76 |
| 07 | Abuse030 | 1328 / 39.3–49.3 | 3,4 (localized) | 512 / 12.1–22.1 / pre | 0.45 |
| 08 | Assault011 | 864 / 23.8–33.8 | 0–7 | 2064 / 63.8–73.8 / post | 0.48 |
| 09 | Arrest024 | 2048 / 63.3–73.3 | 0–7 | 208 / 1.9–11.9 / pre | 0.88 |
| 10 | Explosion013 | 2752 / 86.7–96.7 | 0–7 | 736 / 19.5–29.5 / pre | 0.64 |

**S3 normal windows** (Normal_Videos …):

| Video | 246 | 606 | 926 | 932 | 910 | 100 | 901 | 210 | 778 | 870 |
|---|---|---|---|---|---|---|---|---|---|---|
| Released label | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |

**Size.** 20 distinct videos, 354.9 MB of zip members to range-fetch, as single members with CRC checks. Only the 8
needed frames per window are written (§14). If the needed-frames extractor fails validation (§13.6), the fallback is
VERA's full extractor into the shared cache: 51,243 frames, about 1.5 GB at experiment02's ~29 KB per frame. That
exceeds the 1 GB storage cap and needs the user's approval first.

**Replacement**, before F2 only. A video is replaced if any of these happens:
- (a) the member cannot be fetched or fails its CRC;
- (b) the decoded frame count ≠ the manifest length;
- (c) the 8-frame check fails: any of the 8 indices does not decode, or is ≥ the decoded count;
- (d) any of its windows reached the model before F2.

How it is done (implemented in `select_windows.py --exclude`, tested by `selection/test_replacement.py`):
- **Slot-stable IDs.** Slot k keeps its window IDs (S1-k, S2-k, S3-k) whatever else is excluded; no other window is
  renumbered.
- **S1/S2 slot k.** Its candidates are the first 1 + 5 jointly eligible videos of its class, in video-key order. The
  first one not excluded is used, and it brings its own S2 window from the same video. Listed backups per class:
  RoadAccidents 5, Vandalism 2, Shoplifting 5, Burglary 5, Fighting 4, Arson 5, Abuse 1, Assault 1, Arrest 3,
  Explosion 5. A video beyond a class's listed candidates is never used.
- **Exhausted class.** If none of a slot's candidates is left, the slot takes the first next class in class-key order
  that is not yet used and still has a candidate: Robbery (3 candidates), then Shooting (6), then Stealing (4). The
  slot keeps its ID.
- **Dropped slots.** If no next class has a candidate left, the pair is dropped and logged. The S3 rule is the same
  with the 5 listed S3 backups. A dropped slot simply has no window: every D1, D2 and D4 minimum is unchanged, and
  O6 loses that pair. The D5 thresholds are fractions of the windows present, and a dropped slot 01 moves the gate
  windows (§9 D5). `test_replacement.py` covers both kinds of drop.
- **Pre-listing.** Every video the rerun can ever pick is pre-listed with its full windows in `backup_windows.json`
  (103 backup windows; each backup S2 record names its S1 in `paired_with`). A replacement window is identical to its
  pre-listed record apart from its ID, `paired_with` and the replacement fields (`replaces_video`, `backup_record`).
  The validator uses the same field set as `test_replacement.py`.
- **Download cap.** Replacement downloads count toward the download cap (§10). If fetching a replacement would exceed
  the cap, prep pauses before the fetch and the user is asked. The study is not stopped and no backup is skipped
  automatically.
- **Result.** The rerun's `windows.json` becomes `windows_final.json`, with the replacement log from
  `selection_log.json`. The validator checks that `windows_final.json` is byte-identical to a fresh
  `select_windows.py --exclude <logged ids>` run, that it differs from `windows.json` only in logged slots, and that
  every replaced window matches its backup record.

**Fresh vs released labels.** Strata are fixed by selection. Fresh L0 may differ from the released label: in
experiment01, the fresh label was 1 on 10 of the 13 anchors the released file labelled 1. Every outcome states its
label basis (§7).

## 4. Masks per window and family

**B1, fixed (22).** 8 single frames; 7 adjacent pairs (i, i+1); 6 adjacent triples (i, i+1, i+2); all eight.

**B3, fixed experiment02 controls (6, descriptive only).** Pairs (1,6), (2,4), (4,6); triples (0,3,7), (0,4,7),
(3,5,7). They are size-matched but not evidence-matched; this is experiment02's confounded design, and it never enters
a decision.

**B2, composition-matched controls.** Built deterministically from **closed** B1 (§5) of the same window and family,
for every adjacent group. The peaks decide D1; the rest feed secondary analyses.
- **Inputs.** Let `e(i) = c·Δs(i)`, the direction-adjusted single-frame effect on S_hi, and for a slot set G let
  `m(G) = Σ_{i∈G} e(i)`.
- **Peak groups.** For k = 2 and k = 3, the adjacent k-group with the largest m(G); ties go to the smallest slot tuple.
- **Candidates.** For an adjacent group G of size k, H ranges over all nonadjacent k-sets (no two slots consecutive):
  21 pairs and 20 triples. H **may share slots with G**. That is intended: with a shared slot s, the comparison is an
  adjacent partner of s against a distant partner with matched single-frame evidence. Disjoint matching is infeasible
  where evidence sits in consecutive slots.
- **Composition distance.** `D(G,H) = max_j |sort↓(e over G)_j − sort↓(e over H)_j|`.
- **Eligible H:** `D(G,H) ≤ 0.25` (one bf16 step) **and** `m(H) ≥ m(G)` (zero slack; v0.2 allowed 0.05). The control
  removes at least as much direction-adjusted single-frame evidence.
  - **Why zero slack.** With v0.2's 0.05 slack, a sum-only saturating scorer with three near-equal consecutive slots
    gives single comparisons up to +0.24 nat with no adjacency term (round-2 comp-6). In the reference fixtures with that slack (`round2_design_sim_output_slack005.txt`), single-draw null medians of O2 reached +0.223 nat, close to δ; with zero slack the largest was −0.002.
  - **What zero slack guarantees.** Take a control that differs from G in a single slot, sharing k − 1 slots. Its
    single effect in that slot must be at least as large as G's. Consider a sum-only scorer, where a set's effect is
    one **strictly** increasing function of the summed latent evidence of its slots. Under it, that slot carries at
    least as much latent evidence, so d ≤ 0. Scorers with flat regions (thresholds, quantisation such as the bf16 S
    scale) are not covered.
  - **Bias.** Controls that differ from G in more than one slot are not covered by that argument. In the §13.2
    reference fixtures, the median O2 under the null was negative in every cell with a defined median
    (260 of 327; −0.289 to −0.007 nat).
- **Choice.** `G*` = argmin over eligible H of (D, |m(H) − m(G)|, lexicographic slot tuple). Composition balance is kept
  first. Ordering by |m(H) − m(G)| first was also simulated; it changed no D1 verdict share by more than
  1 percentage points (`round2_design_sim_output_order_sum.txt`).
- **No match.** If no H is eligible, G is **UNMATCHABLE**. A window whose two peak groups are both UNMATCHABLE is
  `peak_unmatchable`.
- **Logged per (G, G*):** D, the signed sum error `m(G*) − m(G)`, |G∩G*|, the minimum slot gap in G*,
  `complete_removal` (annotated slots ⊆ G and not ⊆ G*), and how many groups each G* serves. **Logged per
  UNMATCHABLE G:** the minimum D over all H, and the sum slack `m(H) − m(G)` of the H with that minimum.
- **Weighting (estimand).** Each adjacent group's contrast is with its own matched control, and each adjacent group
  counts once. A control serving several groups is not down-weighted (round-1 #17 skeptic's fix, minor #3). The
  masks run are the distinct G* (≤ 13 per family).
- **Missing data:** §5.

**C, cross-family ranking.**
- **Rankable** in a source family means: L0 present, S_hi present for the baseline and all 8 singles, ≥ 2 distinct
  values, and ≥ 2 singles with `e(i) > 0.25` (signed, so the top-2 slots are above resolution in the direction of L0).
- **Rankings.** The singleton ranking orders slots by `e(i)`, descending, ties by slot index. The group ranking uses
  the size-balanced mean of `c·Δ/|G|` over adjacent pairs and triples. Rankings are built in blur and evaluated with
  constant, and the reverse.
- **O4 comparator (budget 2, removal).** The exact mean over **all 28 slot pairs** of the evaluation family's
  `c·Δ(pair)`. These are score-only masks: no target-1 generation for pairs not already in B1, B2 or B3. Up to 18 extra
  per family per rankable window.
- **Secondary.** Group ranking, budget 4, retention mode, and 3 per-window random rankings: slots ordered by
  `sha256("experiment03-20261023:random:{video_id}:{anchor}:{k}:{slot}")`, k = 0, 1, 2, used in both evaluation
  families. The key uses the video and anchor, not the window ID.

**Per mask:** one target-1 generation plus one target-2 score, except the C comparator pairs, which are score-only.

## 5. Execution order, closure, missing data and blinding

1. **prep:** fetch and identity checks, the needed-frames extractor, and `windows_final.json`.
2. **Stage A, all 30 windows:**
   - original prompt, label-only generation, label-only score, no-sound ablation;
   - repeat ×3 and alignment on S1-01, S2-01 and S3-01;
   - the 60 all-eight label-only scores (30 windows × 2 families).
3. **Stage A closure and the gate.**
   - Errored or skipped requests are retried in at most 2 further invocations; after that they are *missing*.
   - The D5 gate is written once to `scorer_diagnostics.json`, hashed into the manifest, and never recomputed.
   - Stage A cannot be re-invoked once any B request exists.
4. **B1 + B3**, all windows, both families. B1's all-eight reuses the Stage A request.
5. **B1 + B3 closure**, same retry limit: `b1_closure.json` and `b3_closure.json`.
6. **B2**: only if the gate is not blocked; built only from closed B1. Then **B2 closure**, same retry limit,
   `b2_closure.json`; requests not run because a cap was reached are listed there as `skipped_budget`. A G* that stays
   missing is not replaced by the next-best H.
7. **C**: only if the gate is not blocked; rankability from closed B1. Priority order:
   - first the O4-primary masks (singleton removal at budget 2, plus the 28-pair comparator), in stratum round-robin
     (S1-01, S2-01, S3-01, S1-02, …);
   - then the secondary masks.

   Then **C closure**, same retry limit, `c_closure.json`. Requests not run because a cap was reached are listed there
   as `skipped_budget`.
8. **Analysis**, using the frozen analyzer. It starts only when every closure file exists and is hashed into the
   manifest.

**Missing data, per target.** Every closure file lists, per record, request-level failures and, in separate fields,
target-1 parse failures and target-2 values that are non-finite or fail the S_hi rounding check (D5). "Missing" in
this protocol means listed there.
- **Target-2 (S_hi) quantities:** B2, testability (§6), rankability (§4 C), O2, O4, O5 and O1-score. They need L0 and
  S_hi only. A target-1 parse failure on a single never removes a window from B2, T, P or C.
- **Target-1 quantities:** O1-bin, the Fragile flag, O3, O6 and R-a/R-b/R-b′. A target-1 failure on a B1, B2, B3 or
  C mask affects only these. A missing L0, the baseline label, removes the window from every direction-dependent
  quantity.
- **O2:** a comparison is dropped (logged) if S_hi is missing for G or G*; P is counted after these drops.
- **O3:** a comparison is dropped (logged) if L0, L(G), L(G*) or the label of any single in G ∪ G* is missing.
- **O4:** a direction is unavailable if S_hi is missing for the baseline, the top-2 removal or any of the 28 comparator
  pairs (§7).

**Blinding.**
- Stage return values, stdout and `run_manifest.json` carry only counts, statuses, errors and budget figures.
- Labels, flips, rankability and gate details go to sealed results files.
- A monitor script prints request counts per stage and status, and it is the only view used during the run.
- No outcome table is opened before step 8, except the gate file, which is needed for step 6.

## 6. Units of analysis and populations

- **T (testable alerts):** S1 windows, one per video, so the window is the unit, with fresh L0 = 1, S_hi present for
  the blur baseline and all 8 blur singles, and ≥ 2 blur singles with `e(i) > 0.25`.
- **P (primary for D1):** windows in T where at least one of the two blur peak groups has a matched comparison with
  S_hi present for G and G* after closure. Windows in T but not in P are logged by reason:
  - `peak_unmatchable`: both peaks are UNMATCHABLE;
  - lost: a peak was matched but its comparison was lost to missing data or a cap. A window with one UNMATCHABLE peak
    whose other peak comparison was lost counts as lost.

  D1 rule 2 treats the two reasons differently.
- **Expected size.** experiment01's fresh-alert rate (10 of 13) and experiment02's blur testability (1 of 2 alerts)
  give |T| ≥ 6 with probability about 0.14; with 80 % testability it is about 0.67. D1 rule 1 (INCONCLUSIVE) is
  therefore a real possibility.
- **Secondary, each reported separately and never fed into D1:**
  - S2;
  - the constant family;
  - pooled 20-video analyses (S1 and S2 averaged within video);
  - every D1 statistic recomputed without fragile windows;
  - **S3 placebo:** S3 windows that pass the T rule with c from their own L0, analysed exactly like P. If the placebo
    has ≥ 6 windows and HL ≥ δ, the D1 report states that the effect is not specific to true alerts. This never
    changes the D1 verdict.
- **D2 population:** fresh alerts, S1 ∪ S2 with L0 = 1. Unit = video (windows averaged within video).
- Masks, families and seeds are never counted as sample size.

## 7. Outcomes

`flip` means `L(mask) ≠ L0`, with missing values excluded. `f(i) = 1` if blur single i flips L0.

| Id | Definition | Label basis and denominator |
|---|---|---|
| **O1-bin** | A window has ≥ 1 B1 single or adjacent mask, in either family, that flips L0 | windows with L0 present; by selection stratum, split by L0 |
| **O1-score** | ≥ 1 B1 single or adjacent mask with `c·Δ ≥ 1.0` nat (4 bf16 steps) | windows with L0 and S_hi present |
| **Fragile** (flag) | In some family, ≥ 3 of 8 singles flip L0 | windows with L0 present |
| **O2 (primary)** | For each matched **peak** comparison in blur: `d = c·[Δ(G) − Δ(G*)]` (raw difference; composition already matched). Window value = mean of d over the matched peaks (the pair, the triple, or both). | set P |
| **O3** (consistency, decision role) | For the same blur peak comparisons, counted only when no single in G or G* flips L0 (`Σf(G) = Σf(G*) = 0`, "emergent flips"): `flip(G) − flip(G*)`. Window value = mean over the counted peaks. | windows in P that are not Fragile in blur and have ≥ 1 counted comparison; fragile windows logged by stratum |
| **O3b** (coverage, descriptive) | In blur, per window of T: the number of UNMATCHABLE adjacent groups, how many of them flip L0, the share of flipping groups that are UNMATCHABLE (per window, and pooled over T with and without fragile windows), `peak_unmatchable`, the largest m(G) over all adjacent groups and over matched ones, and the minimum D and sum slack of each UNMATCHABLE group | T |
| **O4** | Per rankable window and direction: `R = c·Δ(top-2 singleton removal) − mean over 28 pairs of c·Δ(pair removal)`, in the evaluation family. A direction is **available** only if the evaluation family's baseline S_hi, the top-2 mask and all 28 comparator values are present after closure. Window value = mean over the available directions; a rankable window with none is "rankable, not evaluated" (logged). Video value = mean over evaluated windows. | D2 population, rankable |
| **O5** | Per window rankable in both families: 1 if the top-2 sets of the singleton ranking are identical across families, else 0. **Video value:** 1 if every such window of the video has O5 = 1, else 0. A video is eligible for O5 if ≥ 1 of its D2-population windows is rankable in both families. | D2 population (S1 ∪ S2, fresh L0 = 1); S3 and L0 = 0 windows reported separately |
| **O6** | Per family, flips ÷ evaluable masks among the 21 B1 single and adjacent masks (≥ 17 evaluable required). Window value = mean over families. Primary: S2 − S1 within video, for pairs with L0 = 1 in both windows. | pairs |

**Covariates (descriptive):**
- the margin `m0 = c·S_hi(baseline)`;
- the released-label run length at the anchor;
- the released 1-share of the video;
- the event side (S2);
- the annotated slots and the localized flag (S1, §3);
- the mean pixel MSE between consecutive slot images at 448×448, as a visual-redundancy measure computed on CPU.

**Secondary analyses (pre-labelled, descriptive):**
- **O2 variants:**
  - **O2-all:** the v0.2 estimand, the mean over sizes k ∈ {2, 3} of the mean d over every matched adjacent group
    (the available size if only one), on T windows with ≥ 1 matched group;
  - **O2-evid:** as O2-all, restricted to adjacent groups that contain a consecutive pair with e > 0.25 in both
    slots;
  - **redundancy subset:** d for matched adjacent groups whose singles are all at or below resolution
    (`e ≤ 0.25`), with the group's own `c·Δ(G)`;
  - O2 by size (peak pair only, peak triple only);
  - O2 in the constant family;
  - O2 with the `I`-difference `c·[(Δ(G) − m(G)) − (Δ(G*) − m(G*))]` and slope variants s ∈ {0, 1, ŝ}, where ŝ is
    the pooled blur OLS slope of Δ(H) on m(H) over nonadjacent masks;
  - excluding `complete_removal` comparisons;
  - split localized vs full-coverage S1 windows;
  - excluding fragile windows (D1 statistics recomputed);
  - with distinct-G* weighting (contrasts sharing a G* averaged first), and the number of adjacent groups per
    window and family that have an eligible H sharing no slot with G;
  - restricted to windows where scorer class = L0 with S ≠ 0, and further to |S_hi| > 0.25;
  - in S2 and in S3 (the placebo, §6);
  - every O2 variant also on the bf16 S scale.
- **O3 variant:** the v0.2 count-filtered O3 (comparisons where G and G* contain the same number of single-flip slots,
  fragile windows included), over all matched adjacent groups.
- **experiment02 replication block (by stratum and family; different set sizes, so not a test).** Among windows with
  complete target-1 B1 singles and zero single-frame flips:
  - R-a: the number in which some adjacent group flips L0;
  - R-b: the same for B3 groups;
  - R-b′: the same for B2 controls.

  Report k / set size for each.
- **B3 contrast:** the mean flip of the adjacent k-groups minus the mean flip of the 3 B3 k-groups, averaged over k.
  Labelled "size-matched, evidence-confounded".
- **Everything else:**
  - O4 with the group ranking, at budget 4, and in retention mode;
  - O5 with the video mean instead of the all-windows rule (labelled sensitivity analysis);
  - τ_b over all 8 singles, and over singles above resolution in either family when there are ≥ 4 of them;
  - agreement between released and fresh labels;
  - agreement between the scorer class and L0;
  - no-sound ablation changes and sound sentences by stratum;
  - claim-keyword changes on flips;
  - the candidate-mass flag rate.

## 8. Analysis methods (implemented in the frozen analyzer and tested against fixtures)

- **Libraries.** The tests, HL and sign counts use the standard library only. The descriptive bootstrap uses numpy at
  the version pinned in the frozen environment (`setup/environment/requirements.txt`). scipy is used only in tests,
  as a reference.
- **Zeros.** Exact-zero window values are dropped before the Wilcoxon test, HL and its bounds alike; the analyzer
  reports how many were dropped.
- **Wilcoxon signed-rank:** exact one-sided p by enumerating all 2^n sign assignments of the non-zero window values
  ("wilcox" method), using average ranks.
  - With n = 6 the smallest p is 1/64 ≈ 0.016; with n = 10 it is 1/1024.
- **Hodges–Lehmann (HL):** the median of the Walsh averages of the non-zero window values.
  - Its one-sided 95 % lower and upper confidence bounds are the k-th smallest and k-th largest Walsh averages. Here
    k − 1 is the largest c such that P(W⁺ ≤ c) ≤ 0.05 under the exact null distribution (for n = 6, k = 3).
  - With fewer than 5 non-zero values no such bound exists. Then neither SUPPORTED nor NOT SUPPORTED can be reached,
    and D1 falls to INCONCLUSIVE.
- **Sign counts** (#positive vs #negative window values, used in O3 conditions and O6 descriptions) are reported as
  counts. Where a p-value is shown, it is an exact two-sided binomial p via `math.comb`, labelled exploratory.
- **Bootstrap** (descriptive CIs only): numpy `Generator(PCG64(20261023))`, 10,000 resamples of units.
- **Single confirmatory test:** O2 in D1, at α = 0.05. No Holm is needed. Every other p-value is labelled exploratory and unadjusted.

## 9. Decision rules (fixed before data; each is applied in the stated order)

**D1. Q2, adjacency beyond matched evidence.** Apply the first rule that holds.

0. Scorer `blocked` (D5) → **NOT TESTED**. B2 cannot be built. B3 contrasts are descriptive only.
1. |T| < 6 → **INCONCLUSIVE (too few testable alerts)**.
2. Let N_pu be the number of `peak_unmatchable` windows in T.
   - 2a. |T| − N_pu < 6 → **INSEPARABLE at 8-slot resolution**. In too many windows the strongest adjacent evidence
     cannot be matched by non-adjacent slots, so adjacency cannot be separated from evidence composition here. This
     holds whatever data is missing.
   - 2b. Otherwise, |P| < 6 → **INCONCLUSIVE (peak comparisons lost to missing data or a cap)**. Report N_pu, the
     number of windows lost, and the cause of each loss: retry exhausted, S_hi rounding failure, or cap reached.
3. **SUPPORTED** if all of the following hold:
   - exact one-sided Wilcoxon p < 0.05 on P's O2 values;
   - HL(O2) ≥ δ = **0.25 nat**, one bf16 step: the grid spacing of the deployed bf16 output at the observed logit
     magnitudes. δ is the smallest effect of interest for a single peak comparison; it does not claim that smaller
     S_hi differences leave the bf16 output unchanged (§2). The native bf16 O2 is reported beside it;
   - O3 not reversed: #positive ≥ #negative among O3 window values. If no window has an O3 value, the condition is
     recorded as "not evaluable" and does not block;
   - under status `provisional_diverges_from_original`: P's agreement subset (scorer class = L0, bf16 S ≠ 0) is
     non-empty and its median O2 is > 0. If the subset is empty, D1 cannot be SUPPORTED and goes on to rule 4.

   If SUPPORTED but HL(O2) ≤ 0 once `complete_removal` comparisons are excluded, the verdict is
   "SUPPORTED, driven by complete-removal comparisons". If fewer than 5 non-zero values remain after that exclusion,
   the check is not evaluable and SUPPORTED stands, with that note.
4. **NOT SUPPORTED** if the one-sided 95 % upper confidence bound of HL(O2) is < δ **and** O3 is not positive-dominant
   (#positive ≤ #negative; not evaluable counts as not positive-dominant).
5. Otherwise **INCONCLUSIVE**.

Every D1 verdict is reported with its scope ("|P| of |T| testable alerts, k of them localized") and with O3b.

**D2. Candidate explanation stimuli for a later pilot.** D2 never authorises a human study. The criteria:
- (i) Among S1 fresh alerts that are **not fragile**, ≥ 60 % have O1-bin = 1, with ≥ 6 such windows evaluable.
- (ii) Video O4 > 0 in ≥ 70 % of evaluated video units, with ≥ 5 units. Rankable units that are not evaluated
  (errors, or a cap reached mid-run) are left out of the denominator and reported.
- (iii) Video O5 = 1 (§7) in ≥ 60 % of the D2-population video units eligible for O5, with ≥ 5 such units.

Verdicts, in this order of precedence:
- **NO-GO** if (i) fails, or if the scorer is blocked (then (ii) and (iii) fail).
- **DEFERRED** if (i) holds but a planned cut (§10), errors or a cap reached mid-run pushed a (ii) or (iii) unit count
  below its minimum.
- **GO:** all three hold.
- **CONDITIONAL GO:** (i) holds and exactly one of (ii) or (iii) fails.
- **NO-GO:** otherwise.

**D3. Display unit**, applied only if D2 ∉ {NO-GO, DEFERRED}. The display shows **target-1 label changes**. Measured
S_hi values appear only if D5 = `usable_auxiliary`; the δ threshold in the D1 note is always printed with the score
name. Under `provisional_diverges_from_original` the display carries the provisional label.
- **Windows shown:** every D2-population window, in window-ID order.
- **Candidates.** For each window shown and each size k ∈ {2, 3}: only B1 adjacent groups, B2 controls G* and B3
  masks. All C masks are excluded (ranking, random and comparator masks alike).
- **"Strongest" set.** Rank the candidates by:
  1. the number of families in which the mask flips L0 (0–2);
  2. then the blur flip;
  3. then the blur `c·Δ` on S_hi, used for ordering only and never printed unless D5 = `usable_auxiliary`;
  4. then the lexicographic slot tuple.

  Show the top mask per size. A displayed adjacent mask that is UNMATCHABLE is tagged "adjacency not tested (no
  matched control)".
- **D1 SUPPORTED** → show the peak adjacent intervals.
- **D1 NOT SUPPORTED** → show the strongest sets, whatever their adjacency, with this fixed note: "On VERA's auxiliary
  first-token score (blur), in |P| of |T| alert windows (k of them localized) whose strongest adjacent pair or triple
  could be matched, the strongest adjacent slot sets did not shift the score by one bf16 step (δ = 0.25 nat) or more
  beyond non-adjacent sets matched on single-frame effects. Adjacency effects away from the strongest sets were not
  tested. U adjacent groups, F of which flipped the label, had no matched control and were not tested. Equivalence
  was not tested for VERA's label."
  - U and F count blur adjacent groups over the windows of T, as in O3b.
  - If T has ≥ 1 localized window and every localized window in T is `peak_unmatchable`, append: "This result does
    not cover brief localized events, whose strongest groups could not be matched."
- **D1 "SUPPORTED, driven by complete-removal comparisons", either INCONCLUSIVE, INSEPARABLE or NOT TESTED** → show
  the strongest sets with no adjacency claim.

**D4. Fragility, exploratory only.**
- **Signal:** "exploratory signal" if all of these hold:
  - ≥ 6 primary O6 pairs;
  - exact one-sided Wilcoxon p < 0.05 on S2 − S1;
  - HL(S2 − S1) ≥ 0.2.
- **Recommendation:** "flag fragile alerts" only if O6 orders S2 above S1 in more primary pairs than the margin m0
  orders S2 below S1. Otherwise "flag low-margin alerts", which needs no perturbation.
- **Sensitivity analyses:**
  - pre-event S2 only: requires ≥ 3 pre-event primary pairs and median S2 − S1 > 0 in them; otherwise D4 reports
    "no signal in the pre-event sensitivity analysis";
  - excluding videos whose released 1-share is ≥ 0.95 (RoadAccidents001).

**D5. Numeric scorer status.** The gate is evaluated once, after step 3 of §5. Definitions:
- a window is **decided** if its baseline bf16 S ≠ 0 and L0 is present;
- N = the number of baseline windows with bf16 S ≠ 0;
- W = the number of windows present after replacement (30 unless a slot was dropped, §3);
- the **gate windows** are S1-01, S2-01 and S3-01. If a slot 01 was dropped, the lowest-numbered present slot of that
  stratum takes its place.

| Criterion | Pass condition |
|---|---|
| tokenization | candidate tokens single and prefix-stable; query reconstruction exact |
| finiteness | all baseline S and S_hi finite (W/W) |
| repeat (gate windows) | |ΔS| ≤ 0.05 on 3/3 |
| alignment (gate windows) | |ΔS| ≤ 0.25 and argmax agreement on 3/3 |
| candidate mass | ≥ 0.5 on ≥ 90 % of the W baselines (27/30) |
| sign vs label-only generation | among the N windows, disagreements plus unparseable label-only replies ≤ ⌊0.1·N⌋ |
| S_hi rounding | 100 % of gate-input records (W baselines + 2W all-eight). A later record that fails the check has missing S_hi (§5); the rate over all records is reported. |
| all-eight moves S | blur |ΔS_bf16| > 0 on ≥ 90 % of the W windows (27/30); constant reported only |

Status consequences:

| Status | Rule | Consequence |
|---|---|---|
| `usable_auxiliary` | all pass; scorer class = L0 on ≥ 80 % of decided windows | all outcomes and decisions as specified |
| `provisional_diverges_from_original` | all pass; agreement < 80 % | same computations; the provisional label is printed beside O2, O4, O5, D1, D2 and D3; the extra D1 SUPPORTED condition applies; it may not be used to drop or reweight anything else |
| `blocked` | any criterion fails | B2 and C not run; O2, O3, O3b, O4 and O5 NOT TESTED; D1 NOT TESTED; D2 NO-GO; O1 uses label flips only; the D4 recommendation is not computed (m0 needs S_hi) |

**D6. Sound question (descriptive).** If removing question 5 changes L0 in ≥ 3 of the windows where both labels parse,
recommend a pre-registered no-sound arm in any later study.

## 10. Budget (approval required)

**Projection** from experiment02's timings: 1.73–3.34 s per generation, 0.36 s per score, Stage A about 170 s, plus 8 % overhead.

| Item | Projection | Cap |
|---|---|---|
| Generation + score inputs | ≤ 30 × (44 B1 + ≤ 26 B2 + 12 B3 + ≤ 40 C) = 3,660, before cache reuse | **4,000** |
| Score-only inputs | 28-pair comparator, ≤ 30 × 2 × 18 | **1,200** |
| GPU time on one RTX 6000 Ada | about 2.4–4.3 h | **5 GPU-h**, cumulative over all experiment03A run directories except `runs/smoke_*`, from `setup/jobs/RESOURCE_LOG.tsv` |
| Smoke tests (§13.7) | — | separate line, `runs/smoke_*` only: ≤ 0.25 GPU-h and ≤ 100 inputs |
| Downloads | 354.9 MB of members, plus HTTP overhead (about 1.3 MB per member) and replacements | 0.5 GB = 500,000,000 bytes received over HTTP, counted for every fetch |
| New home storage | about 0.4 GB | 1 GB; home was at 138.5 / 150 GiB |

**Projection checkpoint after the gate.** It uses only response lengths and the maximum mask counts, and looks at no
outcomes. The maximum counts are 13 distinct G* per window and family for B2, 40 C masks per window (20 per evaluation
family) and 18 comparator pairs per family, for every window as if all were rankable. If the projection exceeds the remaining cap, cut in this fixed
order before B1:
1. secondary C masks;
2. constant-family B2 and B3;
3. O4-primary C masks, in reverse round-robin order.

B1 and blur B2 are never cut. A cap reached mid-run is treated like a cut. If cuts push any D2 minimum below its
threshold, D2 = DEFERRED.

## 11. Stopping, resumption and amendments

- **Technical stops:**
  - The first device (CUDA) error ends the invocation, as the runner already does.
  - A `--time-limit` TERM marks the invocation incomplete.
  - The study stops after 3 consecutive invocations that complete no new request, or when the GPU or input cap is
    reached. A download-cap conflict during prep pauses for the user instead (§3).
- **Resumption:**
  - The same run directory and identical code hashes are required.
  - Every resumption is logged with its reason.
  - The decision to resume uses manifest status and counts only.
- **Validator, two parts:**
  - Technical: identity, 8 frames, held-out videos absent, masks, B2 determinism, closure files, budgets, freeze
    hashes, and the `windows_final.json` replacement checks (§3). No outcome fields. It may run between invocations.
  - Full: run after analysis.
- **Amendments**, recorded in `AMENDMENTS.md` (dated, reasoned), never in this file:
  - (a) Analyzer bug fixes: re-hash, disclose, and report results under both the F1 analyzer and the amended one. The
    validator accepts an analyzer hash that is listed in a dated `AMENDMENTS.md` entry, provided the F1-analyzer
    results also exist.
  - (b) Model-path fixes (engine, attr_lib, runner, prep, B2 builder, extractor): new run directory, and every
    affected request is rerun. No cached record is imported, because the request key hashes the intended inputs, not
    the executed code. The budget stays cumulative.
  - (c) Changes to outcomes or decisions: amendment only, reporting the original-plan results first.

## 12. Freeze

- **F1, before any download.** sha256 of:
  - this file;
  - the four selection outputs, `select_windows.py`, `test_replacement.py` and `CHECKS.md`;
  - `review/round2_design_sim.py`, `review/round3_power_gate.py` and their outputs (design evidence);
  - every file in `code/`: runner, engine, attr_lib, B2 builder, needed-frames extractor, replacement step, analyzer,
    validator, tests, fixture generator (`fixtures.py`) and the fixture results, monitor;
  - a config template (all fields except video sha256s).

  These go into `FREEZE.json`.
- **F2, before the first model call.**
  - The config with each video's sha256 and decoded length, `windows_final.json`, the replacement log, and the
    extractor validation record go into `FREEZE.json`.
  - The main run directory must be empty at F2.
  - The validator checks that no record for an experiment03A window predates F2. It also checks that the analyzer and
    validator hashes at analysis time equal the F1 values, or are listed in `AMENDMENTS.md` under §11 (a).

## 13. Pre-freeze testing (before F1, no experiment03A window ever reaches the model)

1. **CPU end-to-end run with a deterministic stub engine.** Fake labels and S_hi through prep → A → B1 → B2 → C →
   analyzer → validator.
2. **Null fixtures (pass/fail before F1).** `code/fixtures.py` generates synthetic S1 windows with **no adjacency
   term** and runs them through the frozen B2 builder and analyzer.
   - **Normative reference.** `review/round2_design_sim.py` defines the generator: `gen()`, `run_tanh()`,
     `run_convex()`, their random-number call order and the seed strings below. `fixtures.py` must reproduce its
     per-cell verdict counts exactly (§14).
   - **Seeds.** Each cell uses Python `random.Random(seed)` with:
     - `seed = f"experiment03-designsim:{mix}:{A}:{w}:{syn}:{mode}:{margin}"` for tanh cells, with Python `str()` of
       each value (for example `experiment03-designsim:S1like:4:1.5:0.0:event:(0.2, 1.5)`);
     - `seed = f"experiment03-designsim:convex:{mix}:{a}"` for convex cells.
   - **Scorers.**
     - Saturating `S(G) = A·tanh((E − Σ_{i∈G} x_i − off)/w)`, with label L = [S > 0]. The baseline margin
       `u = (E − off)/w` is drawn per window.
     - Convex `Δ(G) = s + a·s²` with s = Σ_{i∈G} x_i. Evidence is drawn at scale 1 (w = 1), and the label flips when
       Δ(G) ≥ 1.5.
   - **Evidence layouts**, with x in units of w. The **planted event** is the set of strong slots, except that spread
     has none and the R-accident profile's event is its slots 3, 4, 5 after the shift.

     | Layout | Strong slots | Background |
     |---|---|---|
     | contiguous | 2–3 consecutive, U(0.3, 1.0) | U(0, 0.05) |
     | scattered | 2–3 nonadjacent, U(0.3, 1.0) | U(0, 0.05) |
     | spread | all 8, U(0.05, 0.5) | — |
     | event3 | 3 consecutive, U(0.3, 0.6) | U(0.1, 0.5) |
     | two pairs | slots 1, 2, 5, 6, U(0.3, 0.6) | U(0, 0.1) |
     | plateau3 | 3 consecutive near-equal (±3 %), level U(0.4, 1.2) | U(0, 0.05) |
     | plateau_bg | 3 consecutive near-equal, level U(0.5, 1.0) | U(0.1, 0.5) |
     | plateau_adv | 3 consecutive near-equal, level U(1.0, 1.6) | U(0, 0.02) |
     | R-accident profile | experiment02's blur profile at slots 1, 3, 4, 5 (0.15, 0.40, 0.90, 0.30, ±20 %), shifted by −1…+2 slots | U(0, 0.02) |
     | heterogeneous | 4–6 slots, one random consecutive pair plus 2–4 other random slots, U(0.3, 1.0) | U(0, 0.05) |

   - **Mixes** of 10 windows: each layout above alone, except contiguous and plateau_adv; S1-like (3 contiguous, 4
     event3, 3 spread); contiguous-heavy (6 contiguous, 4 event3); plateau mix (4 plateau3, 3 event3, 3 spread). The
     contiguous layout is covered by the S1-like and contiguous-heavy mixes.
   - **Null cells** (200 draws each):
     - tanh: the 11 mixes (S1-like, event3, two pairs, spread, scattered, contiguous-heavy, plateau3, plateau_bg,
       plateau mix, R-accident, heterogeneous) × A ∈ {4, 6, 8} × w ∈ {1.5, 2, 3} × u from U(0.2, 1.5), U(0.05, 0.3)
       or U(1.5, 3.0): 297 cells;
     - adversarial: plateau_adv × A ∈ {4, 8, 12} × w ∈ {1, 1.5, 3} × u from U(1.5, 3.0) or U(2.5, 4.0): 18 cells;
     - convex: a ∈ {−0.1, 0.1, 0.3, 1.0} × {S1-like, event3, spread}: 12 cells.
   - **Acceptance at every null cell:**
     - SUPPORTED in ≤ 5 % of draws;
     - among the draws that reach D1 rule 3 (|T| ≥ 6 and |P| ≥ 6), the median of the per-draw P-median O2 is
       ≤ +0.05 nat. A cell with no such draw passes this criterion vacuously, because D1 cannot be SUPPORTED there.
       Such cells are listed with their INSEPARABLE and |T| < 6 shares.

     NOT SUPPORTED, INSEPARABLE and INCONCLUSIVE are all acceptable null verdicts. The negative null bias is reported
     as a power cost.
   - **Reference result** (`round2_design_sim_output.txt`, 327 cells, 65,400 null datasets):
     - SUPPORTED in 0 draws;
     - in the 260 cells with a defined median, the median O2 ranges over −0.289 to −0.007, and the largest
       single-draw median is −0.002;
     - the other 67 cells pass vacuously (all 27 R-accident cells, 22 of the 27 contiguous-heavy cells, 4 plateau_adv cells at U(2.5, 4.0) and 14 other cells at U(1.5, 3.0); every draw there was INSEPARABLE or |T| < 6);
     - the contiguous-heavy mix returns INSEPARABLE in 199–200 of 200 draws at the main margin.

     With v0.2's 0.05 slack (`round2_design_sim_output_slack005.txt`), the largest single-draw null median was
     +0.223 nat; no draw was SUPPORTED under either rule.
3. **Power fixtures (pass/fail before F1).** The same generator and seed strings, with a planted synergy of
   **+0.5 nat** added to Δ(G):
   - **event:** when G contains two consecutive slots of the planted event;
   - **evidence:** when G contains two consecutive slots that both have e > 0.25 + 1e-9.

   **Power cells.** All use the tanh scorer, A ∈ {4, 6, 8} × w ∈ {1.5, 2, 3}, and **1,000 draws** per cell
   (`review/round3_power_gate.py`; its first 200 draws per cell are those of the 200-draw reference run).
   - Under "event": event3, two pairs, S1-like, plateau_bg, R-accident and heterogeneous.
   - Under "evidence": spread and the same six.
   - Margins: u from U(0.2, 1.5) and U(0.05, 0.3) are gated. U(1.5, 3.0) is reported only (§15).
   - **Not power cells.** Scattered (both modes) and spread under "event" are left out: the planted effect never or
     almost never fires there, so they are nulls, already covered by §13.2. The convex scorer and the adversarial block
     are null-only.

   **Acceptance**, evaluated once, on the frozen `fixtures.py`, B2 builder and analyzer, with these seeds and no
   re-seeding:
   - **C1:** NOT SUPPORTED in ≤ 5 % of draws in every power cell at the gated margins.
   - **C2:** in the gated layouts (under "event": event3, two pairs and heterogeneous; under "evidence": spread,
     event3, two pairs and heterogeneous), in every cell at the gated margins with ≥ 50 evaluable draws, the one-sided
     90 % Clopper–Pearson lower bound of SUPPORTED / evaluable is ≥ 0.40.
     - **Evaluable** means a verdict that is not INSEPARABLE and not INCONCLUSIVE for |T| < 6. Fixtures have no
       missing data, so rule 2b never applies.
   - **Reporting.** "Ungated" means exempt from C2 only. Cells with fewer than 50 evaluable draws are reported with
     their INSEPARABLE share, which is the expected power cost. If the gate fails, the design change and the failed
     result are recorded, and the gate is re-run with the same seeds.
   - **Why S1-like, plateau_bg and R-accident are exempt from C2:**
     - in the S1-like mix the effect is present in only some windows of P, which should give INCONCLUSIVE rather than
       SUPPORTED;
     - plateau and R-accident peaks are mostly unmatchable, so INSEPARABLE is the correct verdict there.
   - **Reference result** (`round3_power_gate_output_*.txt`; `round3_power_gate.py --check`): the gate passes.
     - **C1:** 234 cells at the gated margins; NOT SUPPORTED in at most 27 of 1,000 draws (2.7 %, S1-like at
       U(0.05, 0.3)).
     - **C2:** 115 gated cells with ≥ 50 evaluable draws. The smallest 90 % lower bound is 0.427 (event3, A = 8,
       w = 3: SUPPORTED in 0.476 of 187 evaluable draws).
     - 11 heterogeneous cells at A = 8 had 31–49 evaluable draws and are reported only.
     - INSEPARABLE took 10.5–93.8 % of draws in the gated cells.
4. **Statistics fixtures.** The exact Wilcoxon, HL, HL bounds and sign-count outputs are checked on fixed vectors
   against hand-computed and scipy values, including vectors with exact zeros, ties and fewer than 5 non-zero values.
5. **Analyzer fixtures**, each with its expected result:
   - a single with a target-1 parse failure and S_hi present: the window stays in B2, T and P;
   - a missing baseline L0: the window leaves every direction-dependent quantity;
   - an unparseable single inside G ∪ G*: that O3 comparison is dropped;
   - a missing G*, a missing comparator pair, and a cap reached in the middle of a window's C masks;
   - |T| = 7 with N_pu = 1 and both peak comparisons of one other window lost: INCONCLUSIVE (missing data);
   - |T| = 7 with N_pu = 2: INSEPARABLE;
   - two above-resolution singles of opposite sign: not rankable and not O5-eligible;
   - a window with no flipping adjacent group, and one window that dominates the pooled O3b share;
   - 3 localized windows (`peak_unmatchable`), 2 fragile windows and 1 matchable window: |T| = 6, |P| = 3, INSEPARABLE;
   - label noise (random single-flip patterns with ≥ 3 of 8 flips) applied to the same windows: D1 is identical with
     and without the noise, because O2 uses no labels and O3 excludes fragile windows;
   - a D2 video with two O5-eligible windows, one with O5 = 1 and one with O5 = 0 (video O5 = 0);
   - an empty agreement subset under `provisional_diverges_from_original` (D1 not SUPPORTED);
   - a blocked scorer: D1 = NOT TESTED, D2 = NO-GO, O1 from flips only;
   - a dropped slot 01: the next present slot is a gate window, and the D5 fractions use W.
6. **Extractor validation.** Byte-for-byte JPEG comparison of the needed-frames extractor against VERA's full
   extractor, run on held-out RoadAccidents127, whose full frame cache is already on disk.
7. **GPU smoke tests.** Only on experiment02's held-out cases, in `runs/smoke_*`, within the separate budget line.
   Outputs are never pooled.

## 14. Required code changes against experiment02 (each with a test)

| Change | Test |
|---|---|
| Multi-video cases from `windows_final.json`; per-window prep with identity checks | stub end-to-end |
| Needed-frames extractor: VERA's decode loop, writing only the 8 frames per window, plus the decoded count and per-frame sha256 | §13.6 |
| B2 builder (peak groups, composition matching with zero slack, UNMATCHABLE, logs, float64 with tolerance) | null and power fixtures; determinism |
| Fixture generator `fixtures.py` as specified in §13.2–13.3 | run with the §13.2 seed strings, it reproduces the per-cell verdict counts in `round2_design_sim_output.txt` and `round3_power_gate_output_*.txt` exactly |
| Stage C rankability (signed); 28-pair score-only comparator; per-window random rankings keyed on video and anchor; random masks only for rankable windows; O4 availability | fixtures: one single above resolution → not rankable; opposite-sign singles → not rankable; a missing pair still allows the singleton ranking but makes that direction unavailable |
| Gate rewrite to D5 (explicit numerators and denominators; `post_hoc_amendment: false`) | gate fixtures, including a blocked scorer |
| Closure and sealing: retry limit, per-target fields, `b1_closure.json`, `b3_closure.json`, `b2_closure.json`, `c_closure.json`, counts-only stage returns, monitor script | stub end-to-end |
| Analyzer for O1–O6, D1–D6 and the §8 statistics | §13.2–13.5 |
| Validator: V12 snapshots experiment01, experiment02 and setup; V13 filters on `experiments/experiment03A` (cumulative, excluding `runs/smoke_*`); V15 dropped; freeze-hash, amendment-hash, closure-file and F2-timestamp checks; `windows_final.json` replacement checks; `CHECKS.md` hashes against the frozen selection files; `Data/UCF_Eval.json` sha256 at `008ed401` and `15b8bcb8` via `git show` | validator fixtures |
| attr_lib: the named model-path functions are source-identical to experiment02's `attr_lib.py` (sha256 `7acf4acf…`) | AST/source equality test |
| Replacement step (`select_windows.py --exclude` rerun into `windows_final.json`): **done** in v0.3 | `selection/test_replacement.py`, including dropped slots (passes on oz and the Mac) |
| Preflight: idle ≥ 40 GB GPU via `launch.sh`, ≥ 2 GiB home free, experiment02 snapshot | dry run |

## 15. Limitations, expected outcome and what results will license

- **Expected D1 outcome.** INSEPARABLE or INCONCLUSIVE is the most likely D1 verdict.
  - Under these rules, experiment02's only testable blur alert (R-accident) is `peak_unmatchable`. Its peak pair
    (3,4) has m = 2.34 and its peak triple (3,4,5) has m = 2.88, and no nonadjacent set reproduces either
    composition. R-pedestrian's blur singles are all below resolution, so it is not testable. N-middle has L0 = 0.
  - In the S1-like fixture mix (3 localized, 4 event, 3 spread windows; main margin), D1 returned INSEPARABLE in about 75–95 % of draws under the null (149–189 of 200). With a planted +0.5 nat event synergy it returned INSEPARABLE in 74–96 % of draws, SUPPORTED in at most 1 of 200 and NOT SUPPORTED in at most 7 of 200 (3.5 %). With the evidence synergy (every consecutive above-resolution pair), it returned SUPPORTED in 5–25 % and INSEPARABLE in 76–95 % of draws.
  - So a SUPPORTED verdict needs ≥ 6 true alerts whose strongest adjacent evidence is spread enough to be matched,
    with the effect present in most of them.
  - **Power where it exists.** In the event3 layout under event synergy (main margin, 1,000 draws), SUPPORTED came in 0.68–0.70 of evaluable draws at A = 4, 0.56–0.62 at A = 6 and 0.48–0.56 at A = 8 (`round3_power_gate_output_main_event.txt`). Even where peaks can be matched, then, a real +0.5 nat effect reaches SUPPORTED only about half to two thirds of the time.
  - **High-margin alerts.** At high baseline margins (u from U(1.5, 3.0), reported only), the heterogeneous layout lets the planted effect sit at a consecutive pair that is not the peak. There D1 returned NOT SUPPORTED in up to 6.9 % (event) and 11.9 % (evidence) of draws, and SUPPORTED in only 0.02–0.32 of evaluable draws; the other layouts stayed at ≤ 0.7 %. D1 NOT SUPPORTED speaks only about the peak groups, and the D3 note
    says so.
- **What still comes out if D1 is INSEPARABLE.** O1, O3b (how often and where the strongest evidence is consecutive),
  O4–O6, the replication block and D2–D4. D3 then shows frame sets with no adjacency claim.
- **Selection.** S1 and S2 are selected on the authors' released labels, so results concern **attributing alerts**,
  not misses. S2 "benchmark-negative alerts" may include aftermath or ambiguous events. Pre-event S2 sensitivity
  analyses are pre-registered.
- **Scale.**
  - There are 10 S1 windows, 10 of 13 classes, and low power. D1 can be INSEPARABLE or INCONCLUSIVE by design (§6
    expected size).
  - S1 windows are mostly inside long events (7 of 10 full-coverage). The motivating configuration, a brief event
    spanning 2–3 slots, is the localized minority (3 of 10), reported as a moderator.
- **Matching.**
  - Composition matching uses auxiliary-score single-frame effects. For VERA's label it equalises evidence only approximately.
  - Its linear `I` correction assumes slope 1 (reported as a sensitivity analysis).
  - D1 tests the peak groups only. Redundancy among weak neighbouring frames is reported (O2-all, redundancy subset)
    but not decided.
  - A positive O2 identifies non-additivity specific to the strongest consecutive slots, not a mechanism. Visual
    redundancy between neighbouring frames is reported as a covariate.
- **Replacements.** Blur keeps the colour layout, and grey is out of distribution.

## 16. Design alternatives considered (not adopted)

- **Slot permutation** (move the same frames to non-consecutive positions and compare). It matches content exactly,
  but it changes the temporal order of the remaining frames, which is a separate manipulation. It is kept for a future
  study.
- **Scoring all 41 nonadjacent sets per window.** Matching already searches all 41 using B1 singles only, so this
  would not add matches. It would allow averaging over all eligible H, or a within-window Δ-versus-m curve, for about
  30 % more scores. Not adopted, to keep the budget.
- **Selecting only localized S1 events.** That gives more R-accident-like windows, but localized evidence makes peak
  groups unmatchable (INSEPARABLE). The class-diverse rule is kept, with localization reported as a moderator.
- **All-group or evidence-bearing primary O2** (the v0.2 rule, and round-2 option a). Simulated in
  `round2_design_sim.py`. Under a planted +0.5 nat effect confined to the event slots on an informative background,
  they return NOT SUPPORTED in 43–70 % (all-group) and 42–70 % (evidence-bearing) of draws under v0.3 matching, and
  the all-group estimand in 26–59 % under v0.2's 0.05 slack. They are kept as secondary analyses (O2-all,
  O2-evid).
- **v0.2's 0.05 caliper slack.** It gives more matches and more power, but it lets sum-only saturation produce
  positive d (§4; §13.2 reference result). Replaced by zero slack.
- **Elementwise dominance** (every sorted single effect of H at least the matching one of G, and at most 0.25 above
  it). It would guarantee d ≤ 0 under any sum-only scorer, for every control. In simulation
  (`round2_design_sim_output_dom.txt`), though, it gave no further null protection (largest single-draw null median −0.002, as with zero slack) and cut SUPPORTED by up to 38 % in the gated cells, with up to 30 more INSEPARABLE draws per 200. Not adopted.
- **An effect-relative composition tolerance**, `D ≤ max(0.25, 0.25·max e(G))`, with zero slack
  (`round2_design_sim_output_reltol25.txt`). It reduced INSEPARABLE (S1-like null: from about 75–95 % to 60–68 % of draws), but under event synergy in the S1-like mix it raised false NOT SUPPORTED to 31 of 200 draws (15.5 %), which fails the §13.3 gate. Not adopted.
- **The v0.2 pooled flip-share rule for INSEPARABLE** (≥ 50 % of L0-flipping adjacent groups UNMATCHABLE, pooled over
  T). It weighted windows by their number of flipping groups, so one fragile window could outweigh several localized
  ones. It was undefined when nothing flipped. With peak-based P it also added INSEPARABLE verdicts without reducing
  false NOT SUPPORTED in simulation. It is replaced by the peak-coverage rule (D1 rule 2a). The flip shares are reported
  in O3b, and they are named in the D3 note.
