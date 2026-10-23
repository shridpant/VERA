# experiment03B: appendix validation of VERA frame rankings

**Status: protocol v1.1 frozen; implementation validated on CPU fixtures and in a bounded GPU integration check. The
290-video run has NOT been launched.** For results and commands, see
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md).

**Question.** Does a ranking built from a separate numerical scorer pick frames whose grey replacement flips VERA's
*original generated alert* more often than a uniformly chosen pair?
- The ranking comes from blur-replacement changes in the label-only scorer's `S_hi`.
- The comparator is the exact mean over all 28 frame pairs.
- Secondary analyses measure how the ranking depends on the replacement, the numerical readout and earlier removals
  (fixed versus adaptive).
- This is a thesis appendix validation. It is not the thesis's main technical contribution, and it makes no
  novelty claim.

**Lineage.** It builds on experiment03A, the 30-window interval-perturbation protocol (`../experiment03A/`), and
reuses 3A's VERA window arithmetic, zip-listing parser and input-identity checks. It reuses experiment02's engine
and cache keys unchanged.

| Path | What |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | The frozen protocol v1.1: population, readouts, masks, fixed/adaptive definitions, endpoint, analysis, reporting, workflow |
| [`AMENDMENTS.md`](AMENDMENTS.md) | Decisions declared at v1.0, and the pre-data v1.0 → v1.1 changes from the adversarial review |
| [`RECORDS.md`](RECORDS.md) | The raw-record contract shared by the runner, the analyzer and the independent validator |
| [`config.json`](config.json) | The full-run configuration (290 videos, manifest v1, fetch mode) — not launched |
| [`config.integration.json`](config.integration.json) | The bounded integration check (two development windows, ≤ 100 new inputs, ≤ 0.5 GPU-hour, ≤ 500 MiB) |
| [`config.retrieval_rehearsal.json`](config.retrieval_rehearsal.json) | A CPU/network rehearsal of the fetch path on a development video |
| [`config.schema.json`](config.schema.json) | The configuration schema (enforced by `code/config3b.py`) |
| [`manifest/`](manifest) | The locked window manifest v1 (452 window records, 433 unique inputs) and its CSV |
| [`code/`](code) | The runner, core library, engine adapter, retrieval, analyzer, table generator, independent validator and tests |
| [`review/`](review) | The adversarial code/protocol review (41 confirmed findings) and the independent validator's build record |
| `runs/integration_001/` | The bounded GPU integration run (development cases only; not results) |
| `runs/retrieval_rehearsal_001/` | The fetch-path rehearsal on Normal_Videos_018 |
| `runs/synthetic_001/` | A mock-engine run used to test the table pipeline (**synthetic; not results**) |
