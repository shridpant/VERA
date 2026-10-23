# experiment03A: interval-level perturbation study (pre-registration draft)

**Status: PROTOCOL DRAFT v0.4. Nothing has been run.** No model was loaded and no video was downloaded. The protocol is
not frozen; it awaits the user's approval of the design and the budget.

Design superseded for the appendix study by experiment03B (experiments/experiment03B/); kept as the 30-window interval-perturbation protocol.

Formerly experiment03. Seed and key strings keep the `experiment03-` prefix (see `PROTOCOL_DRAFT.md`, "Name").

**Question.** Does replacing a window's *strongest consecutive* sampled frames (its peak adjacent pair and triple)
change VERA's auxiliary score more than replacing *non-consecutive* frames? The comparison frames must have single-frame
effects that match one by one and remove at least as much evidence. This is the confound experiment02 could not
resolve.

Secondary questions:
- alert sensitivity;
- ranking usefulness against all 28 pairs;
- cross-family stability;
- fragility of alerts outside the annotation.

**Expected outcome (§15).** D1 will probably be INSEPARABLE or INCONCLUSIVE. Evidence concentrated in consecutive slots
cannot be matched by non-consecutive slots: experiment02's only testable alert is already unmatchable.

| Path | What |
|---|---|
| [`PROTOCOL_DRAFT.md`](PROTOCOL_DRAFT.md) | the protocol, v0.4: windows, composition-matched controls, outcomes, analysis, decision rules, fixture gates, budget, freeze |
| [`selection/`](selection) | deterministic window selection (`select_windows.py`, slot-stable replacement), `windows.json` (30 windows from 20 videos), `backup_windows.json` (103 pre-listed backup windows), `selection_log.json`, `windows.csv`, `test_replacement.py`, `CHECKS.md` |
| [`review/`](review) | three review rounds (`protocol_review_round1.json`, `_round2.json`, `_round3.json`), the point-by-point response (`RESPONSE_TO_REVIEW.md`), the design simulation (`round2_design_sim.py`) and the 1,000-draw power-gate reference run (`round3_power_gate.py`), with their outputs |

**Next steps, if approved:**
1. Implement and test the code changes in §14, including the null and power fixture gates (§13.2–13.3) and the
   extractor validation on held-out data.
2. Freeze F1.
3. Range-fetch the 20 videos (354.9 MB).
4. Freeze F2.
5. Run within the approved budget, proposed at 5 GPU-hours.
