# Response to the round-1 protocol review (v0.1 → v0.2)

**Review.** `protocol_review_round1.json`: a 41-agent workflow with 4 lenses (statistics, construct validity,
selection, feasibility). Every critical or major finding was checked by an independent skeptic: 37 were confirmed (5
of them critical) and 0 refuted. A further 16 minor findings were reported without verification. Those severities
are the reviewers'. The skeptics' corrected severities are 6 critical, 28 major and 3 minor (round-2 comp-19; see the
round-2 corrections below).

**How to read this.** Numbers refer to the order of `confirmed` and `minor` in the JSON. Section numbers refer to
`PROTOCOL_DRAFT.md` v0.2.

## Confirmed findings

| # | Finding (short) | Response in v0.2 |
|---|---|---|
| 0 | D1 threshold unreachable under pooled aggregation | Adopted, with changes: the primary population is S1 only (§6); D1 uses HL with δ. **Deviation:** δ = 0.25 nat (one bf16 step, the deployed output's resolution), not the suggested 0.10. With composition matching the comparison is a raw difference, so experiment02's plug-in values no longer calibrate δ. Achievability is checked by the §13.3 power fixture instead. |
| 1 | Sum-only matching leaves a composition/nonlinearity confound | **Adopted:** composition matching (sorted single-effect vectors within 0.25), a one-sided evidence caliper, UNMATCHABLE groups and an INSEPARABLE verdict (§4 B2, §9 D1.2), plus evidence-only null fixtures as a pass/fail gate (§13.2). |
| 2 | A blocked scorer fell back to the confounded B3 design | Adopted: D1 = NOT TESTED; B3 is descriptive only (§9 D1.0, D5 table). |
| 3 | Primary population unstated and diluted | Adopted: set P (S1, fresh L0 = 1, testable blur singles, ≥ 1 matched comparison); window = unit; S2 and S3 secondary; S3 is a placebo (§6). |
| 4 | D1 categories overlapped; Holm unclear | Adopted: D1 is an ordered rule; O2 is the single confirmatory test, with no Holm; O3 is a consistency condition (§8, §9). |
| 5 | NOT SUPPORTED from non-significance alone | Adopted: NOT SUPPORTED needs the one-sided 95 % upper bound of HL < δ; D3 has separate wordings (§9). |
| 6 | Random comparator shared and position-biased | Adopted: the exact mean over all 28 pairs (score-only) is the O4 comparator; secondary random rankings are hashed per window (§4 C). |
| 7 | Consequences of D5 statuses undefined | Adopted: a status-to-consequence table (§9 D5). |
| 8 | Label basis and denominators unclear | Adopted: a label-basis/denominator column for every outcome (§7). The experiment01 agreement figure was corrected to 10/13 (§3). |
| 9 | Missing-data rule ambiguous | Adopted: closure files and per-target rules (§5, §7). |
| 10 | O5/D2(iii) had no minimum and was noise-sensitive | Adopted: top-2 identity, ≥ 5 video units, 60 % threshold; τ_b descriptive (§7, §9). |
| 11 | Backups lacked same-video S2 | Adopted: joint eligibility, no cross-video fallback, assertions (selection code). |
| 12 | B2 under-matching hid the confound | Adopted: the one-sided caliper `m(H) ≥ m(G) − 0.05`, UNMATCHABLE and O3b reporting, and INSEPARABLE (§4, §7, §9). |
| 13 | Pooled 20 units with normal videos | Adopted (see #3); S3 is a placebo. |
| 14 | Direction from the scorer while O3 used VERA's label | Adopted with a simplification: one direction `c = 2·L0 − 1` everywhere; agreement-subset secondaries; an extra D1 condition under `provisional_diverges` (§2, §7, §9). |
| 15 | O3 not matched on target-1 evidence | Adopted: O3 counts only comparisons with equal numbers of single-flip slots (§7). |
| 16 | Linear I-correction assumes slope 1 | Adopted: the primary O2 is the raw difference under composition matching; I-difference and slope variants s ∈ {0, 1, ŝ} are secondary; the assumption is stated in §15. |
| 17 | Overlap of G and G* undeclared | Adopted: overlap allowed and explained; each adjacent group counts once; |G∩G*| and reuse logged (§4 B2). |
| 18 | Blocked fallback to B3 | Adopted (see #2). |
| 19 | NOT SUPPORTED and D3 wording | Adopted (see #5); the D3 fallback shows frame *sets* (target-1 sensitivity needs sets) (§9 D3). |
| 20 | S1 rule selects mostly full-coverage windows | Partly adopted: the class-diverse rule is kept, with localization pre-registered as a moderator and `annotated_slots` recorded. The rationale is in §16: localized evidence makes peak groups unmatchable. |
| 21 | S2 fragility confounded with margin | Adopted: margin and run-length covariates; the D4 recommendation compares O6 with margin; D4 is "exploratory signal" (§7, §9). |
| 22 | experiment02 replication outcomes dropped | Adopted: replication block R-a/R-b/R-b′ (§7). |
| 23 | Partial-coverage S1 windows confound complete removal | Adopted: `annotated_slots`, the `complete_removal` flag and sensitivity, and a D1 qualifier (§3, §4, §9). |
| 24 | Undocumented cross-video S2 fallback | Adopted: deleted; the script asserts pairing (selection code). |
| 25 | Backup counts wrong; exhaustion undefined | Adopted: actual counts stated; next-class rule; `--exclude` rerun; full pre-listed backups (§3). |
| 26 | Pre/post-event S2 mixed | Adopted: `side` recorded; pre-event preferred (changes S2-03 only: anchor 3712 → 624); pre-event D4 sensitivity (§3, §9). |
| 27 | Freeze did not cover the analyzer and other code | Adopted: F1/F2 hash every code file; `AMENDMENTS.md`; validator hash checks (§12). |
| 28 | Stage order undefined | Adopted: execution order and closure (§5). |
| 29 | Contract "verbatim" imported the v2 gate | Adopted: the applicable contract sections are named; `post_hoc_amendment: false`; the gate has explicit numerators (§2, §9 D5). |
| 30 | Blocked-branch O3 undefined | Adopted (NOT TESTED; descriptive B3 contrast specified in §7). |
| 31 | Budget too tight; cut order | Adopted: a projection range, a 5 GPU-h cap (cumulative), a checkpoint and a fixed cut order (§10). |
| 32 | Stopping and resumption did not match the runner | Adopted: runner-level stops, resumption rule, two-part validator, amendment classes (§11). |
| 33 | Backup list could not support replacement | Adopted (see #25). |
| 34 | Freeze vs replacement ordering | Adopted: two-step freeze, F1 and F2 (§12). |
| 35 | GPU smoke tests could leak outcomes | Adopted: smoke tests on held-out experiment02 cases only, with their own budget line (§13.6). |
| 36 | Implementation list incomplete | Adopted: required-changes table with tests (§14). |

## Minor findings

| # | Response |
|---|---|
| 0 | Exact enumeration Wilcoxon with HL; zeros dropped (§8) |
| 1 | One direction from L0 (§2) |
| 2 | D4 is "exploratory signal" with ≥ 6 pairs (§9) |
| 3 | "matched comparison" terminology (§4, §7) |
| 4 | Operational adjacency definition and licensed claim (§1) |
| 5 | Sizes weighted equally; by-size secondaries (§7) |
| 6 | O1 split into binary and score; D2(i) uses binary and excludes fragile windows (§7, §9) |
| 7 | Multi-interval rule written as implemented (§3) |
| 8 | Key templates and ordering stated (§3) |
| 9, 15 | Both commits named; manifest hash identical at both (§3, `selection_log.json`) |
| 10 | The S3 anchor is the unclipped anchor closest to n/2; the exclusion is "no unclipped anchor" (§3) |
| 11 | Code status per file (§2) |
| 12 | Extractor validation on held-out RoadAccidents127; full-extraction fallback needs approval (§3, §13) |
| 13 | Counts-only stage returns, sealed results and a monitor (§5) |
| 14 | Reused controls: each G counts once; reuse logged (§4) |

# Response to the round-2 review (v0.2 → v0.3)

**Review.** `protocol_review_round2.json`: two lenses (completeness and executability) on v0.2, sha256 `a1779a12…`.
- Every critical or major finding was checked by an independent skeptic: 14 were confirmed (2 critical, 12 major) and
  1 was refuted.
- A further 22 minor findings were reported without verification.
- The first run (`wf_7758d908-2c3`) stopped before the completeness reviewer and one skeptic finished. The
  continuation (`wf_660fe42f-0f8`) re-ran the completeness reviewer with the same prompt, its skeptics, and the
  missing skeptic.

**How to read this.** IDs are `exec-N` and `comp-N`, where N is the index in that lens's findings list
(`protocol_review_round2.json` → `confirmed`, `refuted`, `minor`). Section numbers refer to `PROTOCOL_DRAFT.md` v0.3.

**Design evidence.** v0.3's estimand and matching rule were chosen by simulation before writing them down:
`round2_design_sim.py`, with outputs `round2_design_sim_output*.txt` (§13.2–13.3). The simulation implements v0.3's
D1 exactly (peak O2, zero slack, |P| < 6, emergent-flip O3), plus the v0.2 and round-2 alternatives.

## The central change (exec-0, comp-0, both critical)

v0.2 averaged d over every matched adjacent group. An effect confined to the 2–5 groups that contain consecutive event
slots was diluted below δ, so it could not reach SUPPORTED. It also led to NOT SUPPORTED with a false absence note in
D3. And §13.3 was "reported, not a gate", so the protocol could have been frozen with no power.

v0.3:
- **Estimand.** O2 now uses each window's **peak** adjacent pair and triple: the largest m(G), fixed from B1 singles
  before B2 (§1, §4, §7). This is the option comp-0 itself proposed first ("per size the matched adjacent group with
  the largest m(G)"). The all-group mean (O2-all) and the evidence-bearing subset (O2-evid) remain as secondary
  analyses.
- **Why not the other options.** In simulation, under a +0.5 nat effect confined to three event slots on an
  informative background, the all-group and the evidence-bearing estimands both returned NOT SUPPORTED in
  43–70 % of draws. The peak estimand returned NOT SUPPORTED in at most 1 of 200 draws in that layout, and at most 7 of 200 in any synergy cell of draws in any cell.
  Option (b), δ = 0.10 on the averaged scale, is not adopted. It would make the threshold depend on how many groups
  carry the effect, which is not known in advance.
- **δ = 0.25** is kept, re-justified as one bf16 grid step, the smallest effect of interest for a single peak
  comparison (§2, §9). The false sentence ("cannot change the deployed model's output probabilities") is deleted. §2
  now reports that 36 of experiment02's 109 sub-0.25 S_hi effects changed bf16 S, 4 of them in the opposite
  direction.
- **§13.3 is a pass/fail gate before F1** (deviation stated):
  - NOT SUPPORTED in ≤ 5 % of draws in every planted-effect cell;
  - SUPPORTED in ≥ 50 % of **evaluable** draws (not INSEPARABLE, not INCONCLUSIVE for |T| < 6) in every cell of the
    gated layouts that has ≥ 20 evaluable draws.

  **Deviation from the skeptics' wording.** The threshold is conditional on evaluability, not unconditional. An
  unconditional "SUPPORTED ≥ 0.5" would require matching peaks that cannot be matched, for example in localized or
  plateau windows. There INSEPARABLE is the correct verdict, so the INSEPARABLE share is reported per cell as the
  power cost.
- **Correction to the round-1 response, row 0.** v0.2 turned the round-1 verified requirement ("must return SUPPORTED
  in most simulated draws") into a report-only check without listing that as a deviation. v0.3 restores it as a gate,
  in the conditional form above.
- **Not changed: round-1 rows 3 and 13.** exec-0's skeptic refuted the sub-claim that they were mis-reported. The
  round-1 verified fix declined an m(G) filter on P, and v0.2 has none.
  - v0.3 still has no such filter on P: a window enters P if one of its peaks is matched.
  - The low-m redundancy groups that the round-1 verdict wanted kept, such as R-accident (1,2), are reported in the
    redundancy subset and O2-all.

## Confirmed findings

| ID | Finding (short) | Response in v0.3 |
|---|---|---|
| exec-0 | Window averaging plus δ: no power; report-only power check; false δ rationale | See above: peak estimand, δ re-justified, gated §13.3, D3 note scoped (§1, §2, §4, §7, §9, §13.3). |
| exec-1 | Pooled flip-share INSEPARABLE rule: undefined at 0/0, dominated by fragile windows | The rule is replaced by peak coverage: INSEPARABLE iff \|P\| < 6, with P requiring a matched peak (§6, §9 D1.2). There is no ratio, so there is no 0/0. The per-window and pooled flip shares, with and without fragile windows, are reported in O3b. D1 statistics without fragile windows are a secondary analysis. Fixtures cover the 0-flip case, one dominant window, and 3 localized + 2 fragile windows (§13.5). Recorded as a new v0.2 defect, not round-1 #15. |
| exec-2 | INSEPARABLE depended on label flips, O2 on S_hi; P could be built from low-evidence matches | P now requires a matched **peak**; windows without one are `peak_unmatchable` (§4, §6). O3b logs the largest m(G) over all groups and over matched ones. The D3 note names the scope and the unmatched groups. In the contiguous-heavy null mix, D1 returns INSEPARABLE in 199–200 of 200 draws (§13.2). |
| exec-3 | Null acceptance two-sided against a one-sided caliper; UNMATCHABLE clause excluded NOT SUPPORTED; generator unspecified; "any monotone" overclaim | The acceptance is one-sided (SUPPORTED ≤ 5 %, median O2 ≤ +0.05), and NOT SUPPORTED is an acceptable null verdict. The generator is fully specified (§13.2): scorers, margins, layouts, mixes, draws and seeds. The overclaim is replaced by the exact zero-slack argument (§4). The (\|m(H) − m(G)\|, D) ordering was tested and changed no verdict share by more than 1 percentage points, so D-first is kept. |
| exec-4 | CHECKS.md predated and under-covered the outputs it was cited for | The checks were rerun on the final v0.3 outputs: 133/133 frame lists against numpy on oz, and all four outputs byte-identical on oz (3.13.5) and the Mac (3.14.2). Replacement tests pass on both hosts. CHECKS.md now records the scope, date, host and sha256 of every file checked; the validator compares them (§14). |
| exec-5 | Replacement renumbered windows; next-class candidates not pre-listed | `select_windows.py` now keeps window IDs slot-stable and caps each class at its listed candidates. A slot whose class is exhausted takes the first unused next class (Robbery 3, Shooting 6, Stealing 4 candidates, all pre-listed). Dropped slots are logged. Random rankings are keyed on `video_id:anchor`. `test_replacement.py` covers own backup, class exhaustion, the next class's second candidate, S3 replacement, backup-record equality and unknown IDs. `windows.json` and `windows.csv` are byte-identical to v0.2 (§3, §4 C). |
| exec-6 | No closure for B2/B3/C; no rule for a missing G* or comparator pair; round-1 #9 and #32 items claimed but absent | Closure with the same retry limit and closure files for B3, B2 and C; analysis starts only when all exist (§5). A missing Δ(G) or Δ(G*) drops the comparison with no re-matching (§5, §7). An O4 direction is available only with all 28 comparator pairs. D2(ii) leaves out "rankable, not evaluated" units, and they can give DEFERRED (§9). Fixtures cover each case (§13.5). Corrections to round-1 rows 9 and 32 are below. |
| exec-8 | O5 population and video value undefined | O5 is defined on the D2 population. A video's value is 1 only if all its eligible windows agree; the video mean is a labelled sensitivity analysis. D2(iii) counts D2-population video units eligible for O5. A fixture covers a two-window video (§7, §9, §13.5). |
| comp-1 | INSEPARABLE 0/0; pooled ratio; empty agreement subset | The flip-share rule is removed from the decision (see exec-1). **Deviation:** the comp-1 skeptic's explicit-count version (F ≥ 3 from ≥ 2 windows, U ≥ F/2) is not adopted. In simulation a flip clause added INSEPARABLE verdicts without reducing false NOT SUPPORTED (§16). F and U are reported in O3b and printed in the D3 note. An empty agreement subset under provisional status means D1 cannot be SUPPORTED and goes on to rule 4 (§9 D1.3). |
| comp-2 | Missing data not split by target | There is now one per-target rule (§5). S_hi quantities (B2, T, P, C, O2, O4, O5) need L0 and S_hi only, so a target-1 parse failure never removes a window from B2, T, P or C. Target-1 quantities (O1-bin, Fragile, O3, O6, R-a/R-b) are affected only by target-1 failures. A D5 rounding failure makes S_hi missing. Closure files record both targets separately. Fixtures are in §13.5. |
| comp-3 | O3 still open to fragility noise | O3's decision role now uses blur only and emergent flips only (Σf(G) = Σf(G*) = 0), with windows that are Fragile in blur excluded and logged. The v0.2 count-filtered O3 is a secondary analysis. A fixture adds non-monotone label noise (§7, §13.5). Round-1 row 15 is corrected to "partly adopted in v0.2" below. |
| comp-4 | D3 "strongest" undefined; note conflicted with the provisional status and over-generalised | D3 now defines the candidates (masks with a target-1 generation) and a four-key ranking: flips across families, blur flip, blur c·Δ (ordering only), slot tuple. UNMATCHABLE adjacent masks are tagged. The note has fixed text, valid under both non-blocked statuses: it names the score, the scope \|P\| of \|T\|, the localized count, U and F, and that equivalence was not tested for VERA's label. It gets a localized-events sentence when it applies. D3 carries the provisional label (§9 D3, D5). |
| comp-5 | O5 population, video rule, and sign-blind rankability | Population and video rule as in exec-8. Rankability (§4 C) and testability (§6) are now signed: ≥ 2 singles with e(i) > 0.25. A fixture with opposite-sign singles must come out not rankable (§13.5). |

## Refuted finding

| ID | Claim | Outcome |
|---|---|---|
| exec-7 | RESPONSE row 17 said "Adopted" while v0.2 did the opposite | **Refuted.** v0.2 follows the round-1 skeptic's corrected fix: each adjacent group weighted once, overlap declared, reuse logged. The remaining minor points are adopted: row 17's wording is clarified below, §4 B2 names the weighting as the estimand, and §7 adds distinct-G* weighting and the count of disjointly matchable groups as secondary analyses. |

## Minor findings

| ID | Response in v0.3 |
|---|---|
| exec-9 | §8 states that numpy (pinned) is used only for the bootstrap, and scipy only in tests. |
| exec-10 | "Localized" is defined in §3, as in the code (S1-01, S1-03, S1-07). |
| exec-11 | The §16 rationale for "score all 41 sets" is reworded: averaging over H, or a Δ-vs-m curve, not more matches. |
| exec-12 | D5 defines "decided" (bf16 S ≠ 0 and L0 present) and N. An empty agreement subset fails the provisional condition. |
| exec-13 | float64 on the stored records, a 1e-9 tolerance in the lenient direction, and min D and sum slack logged for every UNMATCHABLE group (§2, §4). |
| exec-14 | Zeros are dropped for the Wilcoxon test, HL and its bounds alike. No bound exists with fewer than 5 non-zero values; a fixture covers this (§8, §13.4). |
| exec-15 | The projection checkpoint uses the maximum counts (§10). |
| exec-16 | The qualified SUPPORTED verdict maps to the no-adjacency-claim display. An O3 condition with no window values is "not evaluable" and does not block. D2 precedence is NO-GO > DEFERRED > GO / CONDITIONAL GO (§9). |
| comp-6 | The "any monotone" sentence is deleted. Zero slack is adopted after simulation showed that v0.2's 0.05 slack lets saturation produce positive d. Single-draw null medians reached +0.223 nat with the slack and −0.002 without it; no draw was SUPPORTED under either rule. The grid is pinned, including the reviewer's contiguous 3-slot saturating case as the plateau and adversarial layouts (§4, §13.2). |
| comp-7 | CHECKS rerun; see exec-4. |
| comp-8 | Downloads are counted as bytes received over HTTP for every fetch (0.5 GB = 500,000,000 B). If a replacement would exceed the cap, prep pauses and the user is asked; the study is not stopped (§3, §10, §11). |
| comp-9 | §3 describes the slot-stable code, including dropped slots and how D1, D2, D4 and O6 count them. Backup S2 records carry `paired_with`. The validator checks that `windows_final.json` is byte-identical to a fresh `--exclude` rerun. |
| comp-10 | The D4 pre-event analysis needs ≥ 3 pairs and median S2 − S1 > 0; otherwise it reports "no signal". |
| comp-11 | There is no cache import after any model-path fix (§11 b). An "available direction" for O4 is defined (§7). |
| comp-12 | The S3 placebo population is defined (the T rule with c from its own L0). Its only consequence is a stated sentence in the D1 report; it never changes the verdict. The expected \|T\| is stated: P(\|T\| ≥ 6) is about 0.14–0.67 (§6). |
| comp-13 | Libraries are stated; sign counts are descriptive with two-sided exact p; D4 uses HL; zero handling is stated (§8, §9). |
| comp-14 | The blocked row adds O3, O3b and the D4 recommendation; a blocked-scorer fixture is added (§9 D5, §13.5). |
| comp-15 | `runs/smoke_*` is excluded from the 5 GPU-h cap and from V13 and has its own line. DEFERRED precedence is stated, D3 does not apply under DEFERRED, and a mid-run cap is treated as a cut (§9, §10, §14). |
| comp-16 | The validator accepts an amended analyzer hash listed in `AMENDMENTS.md` if the F1-analyzer results also exist (§11, §12). |
| comp-17 | A later record that fails the S_hi rounding check has missing S_hi (§5, §9 D5). |
| comp-18 | The validator checks `Data/UCF_Eval.json` with `git show` at both commits. A test checks that the named attr_lib functions are source-identical to experiment02's (`7acf4acf…`) (§2, §14). |
| comp-19 | The round-1 header now gives both tallies (below and at the top of this file). |

## Corrections to the round-1 response

These rows above now read as follows. The round-1 tables are kept as written, for the record.

| Round-1 row | Correction |
|---|---|
| header | 37 confirmed and 0 refuted, as stated. By the skeptics' corrected severities: 6 critical, 28 major and 3 minor (#3 raised to critical; #16, #17 and #21 lowered to minor). |
| 0 | Not resolved in v0.2: the power check became report-only without a stated deviation. It is resolved in v0.3 by the peak estimand and the gated §13.3. |
| 9 | Partly adopted in v0.2: it had closure for Stage A and B1 only, and no O2 pair rule or per-target rule. Completed in v0.3 (§5). |
| 10 | Partly adopted in v0.2: no video rule and no population for O5. Completed in v0.3. |
| 12, 14 | The empty cases (0/0, empty agreement subset) were undefined in v0.2; defined in v0.3. |
| 15 | Partly adopted in v0.2 (count matching only). v0.3 adds emergent-flip O3 with fragile windows excluded. |
| 17 | Adopted as the round-1 skeptic's corrected fix: overlap declared, each adjacent group weighted once (consistent with minor #3), \|G∩G*\| and reuse logged. The reviewer's original fix 3 (per-control averaging) is not adopted. Its fix 4 (disjoint sensitivity analysis) is now a descriptive secondary analysis. |
| 25, 33, 34 | Not met by the v0.2 code: it renumbered slots and pre-listed only one video per next class. Met in v0.3 (§3). |
| 32 | The O4 inclusion item was dropped in v0.2; v0.3 defines O4 availability. |

## Matching feasibility (round 2)

**experiment02 under v0.3's rules (blur):**
- R-accident is testable but `peak_unmatchable`. Its peak pair (3,4) has m = 2.34 and its peak triple (3,4,5) has
  m = 2.88, and no nonadjacent set reproduces either composition.
- R-pedestrian is not testable: its largest blur single is 0.22.
- N-middle has L0 = 0.

**Under v0.2's rules:** R-accident blur had 6 of 13 adjacent groups UNMATCHABLE, including all 4 that flip L0, and its
matched groups had m ≤ 0.99.

**Implication.** If experiment03's alerts resemble experiment02's, D1 returns INCONCLUSIVE (|T| < 6) or INSEPARABLE.

**Simulation (v0.3 rules), S1-like mix:** D1 returned INSEPARABLE in about 75–95 % of draws under the null (149–189 of 200). With a planted +0.5 nat event synergy it returned INSEPARABLE in 74–96 % of draws, SUPPORTED in at most 1 of 200 and NOT SUPPORTED in at most 7 of 200 (3.5 %). With the evidence synergy (every consecutive above-resolution pair), it returned SUPPORTED in 5–25 % and INSEPARABLE in 76–95 % of draws.

**Null fixtures:** SUPPORTED in 0 of 65,400 null datasets, with null median O2 −0.289 to −0.007 nat. The
§13.2 acceptance is achievable.

# Response to the round-3 check (v0.3 → v0.4)

**Review.** `protocol_review_round3.json`: a bounded workflow on v0.3 with two lenses.
- **resolution:** were the round-2 findings resolved as claimed?
- **consistency:** internal consistency, simulation fidelity, and recomputation of every cited number.

Skeptics confirmed 7 findings (1 critical, 6 major) and refuted none; 19 minor findings were reported.
- **Resolution tally:**
  - 14 confirmed round-2 findings: 6 fully resolved and 8 partly resolved, all of the latter through the §13 gate
    specification;
  - 22 minor findings: 16 fully and 6 partly resolved;
  - none unresolved.
- **Recomputation.** Every cited simulation number recomputed from the output files, apart from the mislabels below.

IDs are `res-N` and `con-N` by lens. Section numbers refer to `PROTOCOL_DRAFT.md` v0.4.

## Confirmed findings

| ID | Finding (short) | Response in v0.4 |
|---|---|---|
| con-0 (critical), res-0 | The power gate passed only on an unstated subset of cells; scattered and spread-under-event cells are nulls and fail C1 | §13.3 now lists the exact power cells: the scorer, the mixes per mode, the (A, w) grid and the margins. Scattered (both modes) and spread under "event" are not power cells, and the convex and adversarial blocks are null-only. Margins U(0.2, 1.5) and U(0.05, 0.3) are gated. U(1.5, 3.0) is reported, and its known weakness is stated in §15 (option (a) of the con-0 skeptic). |
| res-1, con-2 | Seeds and grid not fully specified; §14's reproduction test impossible | `round2_design_sim.py` is normative for `gen()`, the call order and the seed strings, which §13.2 states exactly (`experiment03-designsim:{mix}:{A}:{w}:{syn}:{mode}:{margin}`, and `…:convex:{mix}:{a}`). The null and power cells are listed exactly. "Each layout alone" now excludes contiguous (covered by the S1-like and contiguous-heavy mixes) and plateau_adv (adversarial block only). The planted event is defined for every layout, and the convex block's scale, mixes and flip rule are stated. §14 requires per-cell verdict counts identical to the reference outputs. The output labels now distinguish INCONCL from INCONCL_T. |
| res-2 | C2 point threshold near coin-flip power at A ≥ 6; seed-sensitive | The power cells run at 1,000 draws, and C2 is now a one-sided 90 % Clopper–Pearson lower bound ≥ 0.40 on SUPPORTED / evaluable, in cells with ≥ 50 evaluable draws. The gate is evaluated once, with the stated seeds and no re-seeding. Reference result: the gate passes, with NOT SUPPORTED at most 2.7 % over 234 cells and the smallest C2 lower bound 0.427 over 115 cells. §15 states the conditional power in event3 at A ≥ 6. |
| con-1, res-6 | Null median undefined in cells with no evaluable draw | Such cells pass the median criterion vacuously and are listed (§13.2). §4 now says "in every cell with a defined median (260 of 327)". |
| con-3, res-5 | INSEPARABLE also fired when windows were lost to missing data | D1 rule 2 is split. 2a: \|T\| − N_pu < 6 gives INSEPARABLE. 2b: otherwise \|P\| < 6 gives INCONCLUSIVE (lost to missing data or a cap), with the losses itemised. Mirrored in §6 P, §5 step 6 (`skipped_budget` in `b2_closure.json`) and the D3 branches. Two fixtures are added (§13.5). |

## Minor findings

| ID | Response in v0.4 |
|---|---|
| res-3 | Corrected below (round-2 corrections). |
| res-4, con-7 | §2's example list is fixed. R-accident slot 0 is now given as a same-sign full-step change; the opposite-sign cases are R-pedestrian blur slot 3 (−0.115, +0.25) and N-middle blur slot 3. |
| res-7 | D5 uses fractions of the W windows present. A dropped slot 01 moves the gate window to the next present slot. `test_replacement.py` adds dropped-slot cases (pass). |
| res-8 | The two fixtures now have achievable expected results (§13.5): noise applied to the same windows; 3 localized + 2 fragile + 1 matchable windows give INSEPARABLE. |
| res-9 | §5: a target-1 failure on a mask affects only target-1 quantities; a missing L0 removes the window from all direction-dependent quantities. |
| res-10, con-5 | §2 lists every comparison that uses the 1e-9 tolerance and states that the D1 thresholds, O1-score, D4 and orderings are exact. `round2_design_sim.py` now applies the tolerance to the evidence-fixture threshold too. |
| res-11 | An empty complete-removal check is "not evaluable", and SUPPORTED stands with a note. |
| res-12 | Correction row added below (round-1 minor 9 and 15). |
| res-13, con-9, con-10 | D3 shows every D2-population window in window-ID order. Its candidates are B1 adjacent groups, B2 G* and B3 only, never C masks. U and F count blur adjacent groups over T. The localized sentence needs ≥ 1 localized window in T, with all of them `peak_unmatchable`. The note adds "Adjacency effects away from the strongest sets were not tested." |
| res-14 | §10: "40 C masks per window (20 per evaluation family)". |
| con-4 | §4: the guarantee is stated for strictly increasing sum-only scorers, with "at least as large"; flat or quantised scorers are excluded. |
| con-6 | Numbers relabelled. The all-group estimand gives NOT SUPPORTED in 43–70 % of draws and the evidence-bearing one in 42–70 % (both under v0.3 matching), and the all-group one gives 26–59 % under v0.2's 0.05 slack. Dominance: "up to 38 % in the gated cells". The S1-like ranges are scoped to the main margin. |
| con-8 | §3: a replacement may also differ in `paired_with`; the validator uses `test_replacement.py`'s field set. |
| others | Covered by the rows above (seeds, cells, tolerance, empty cases). |

## Corrections to the round-2 response

| Round-2 row | Correction |
|---|---|
| central change, estimand | The v0.3 estimand combines comp-0's peak idea with exec-2's coverage rule (a). The peak is chosen among **all** adjacent groups, and the window is `peak_unmatchable` if it cannot be matched. comp-0 had proposed the largest-m group among the **matched** groups. That would have let R-accident enter P through a lower matched pair; under v0.3 it is `peak_unmatchable`. |
| central change, deviations | v0.3 deviated from the round-2 skeptics in three ways, not one. First, exec-0's skeptic asked for SUPPORTED ≥ 0.8; v0.3 used 0.5, and v0.4 uses a 90 % lower bound ≥ 0.40 at 1,000 draws. Second, the threshold was conditional on evaluability. Third, the mixed and low-evaluable cells were exempt. |
| central change, figures | EVID's range is 42–70 %, not 43–70 %. The garbled sentence on option (b) is rewritten here: δ = 0.10 on the averaged scale was not adopted because it would make the threshold depend on how many groups carry the effect. |
| round-1 minor 9, 15 | The manifest hash was recorded at `008ed401` only. Its identity at `15b8bcb8` is checked by the §14 validator and is not yet verified. |
