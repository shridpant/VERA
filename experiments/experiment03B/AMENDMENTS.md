# experiment03B amendments and declared deviations

Changes to `PROTOCOL.md` after v1.0 are logged here, each with a date and reason, and each makes a new protocol
version. Results computed under one version are never relabelled under another. An empty amendments section means
v1.0 is in force.

## Declared at v1.0 (before any experiment03B model output)

| Item | Decision | Reason |
|---|---|---|
| Ranking readout | `S_hi` ranks frames. Native BF16 `S` is a sensitivity check. | The governing brief freezes `S_hi` before the dataset run. The older fixed/adaptive brief's native-primary default is superseded. |
| Masks | All 28 pairs in both families, and one fixed plus one adaptive prefix at `k = 1..4`. No triple or quadruple sweep. | Governing brief. The older brief's `exhaustive_small_groups` preset (56 triples) and its 80–100 GPU-hour estimate are superseded. |
| Integration cases | experiment02's R-accident (anchor 2240) and R-pedestrian (anchor 496), both from the development video RoadAccidents127. | These are the earlier brief's named development checks. Both have `L0 = 1` in experiment02, and identical cache keys let experiment02's records be reused. They are not manifest windows. |
| `analysis.secondary_scope` | `confirmation` (development videos excluded) for the full run, and `all` for the integration check only. | Every integration input comes from a development video. Without this setting, every secondary table would be empty. The primary estimate always excludes development videos. |
| Bootstrap RNG | Python's `random.Random(seed)` with `randrange` draws, and the percentile by linear interpolation at `(B − 1)·q`. | This specification lets an independent standard-library validator reproduce the interval exactly. |
| Sparse frame extraction | Only the needed frames are written, using VERA's own decode loop. | The brief forbids bulk frame caches. Byte-identity with a full extraction is checked on a development video. |

## v1.0 → v1.1 (2026-10-23, before any experiment03B model output)

A 46-agent adversarial review checked v1.0 against the handoffs, the statistics, the runner and retrieval. Each
finding went to a skeptic who tried to refute it. 41 findings were confirmed and 1 refuted; the record is in
`review/code_review_v1.0.json`. No scientific setting changed: the readout, endpoint, population, masks,
replacements and bootstrap are as in v1.0. The changes:

| Area | v1.1 |
|---|---|
| Native margin | It is computed as `logit_1 − logit_0`, the exact BF16 logits. The stored `S` carries about 1e-7 of log-softmax noise, which hid exact native ties (§2.3). |
| Tie-break text | v1.0 said the tie-break was "display only". In fact it determines `π`, the prefixes and `T_w`, and an all-tied window gets `{0, 1}` (§5.1). |
| Statuses | Input statuses are `ok`, `fetch_failed` and `invalid_index`. Metadata mismatches are flags, not statuses, so the window still runs as v1.0 intended (§3.1). |
| Retries | Errors are retried inside the request, at most 3 attempts per request counted across logs. v1.0 retried only if a request happened to be issued again (§6). |
| Limits | A reached limit stops the run (failed stage, no analysis). In v1.0's code it became `skipped_budget` lines that the analysis read as missing labels. The scratch limit applies per fetch (§9). |
| Concurrency and completeness | One lock per run directory. Analysis is gated on completeness, overridden only by `--allow-partial` (results flagged) (§9). |
| Code freeze | It covers the analyzer, validator and table generator too (§9). |
| Fetch path | A passing fidelity check is required before extraction. Reused frame identities are re-hashed. The fetched member and byte size are checked against the manifest. Partial downloads are purged (§3.1, §9). |
| Saved artefacts | Prompt strings and the no-sound diff are saved. Replacement examples are saved locally. Adaptive per-state call accounting is recorded (§5.2, §9). |
| Reporting | `N complete` means all 28 labels. A coverage block was added. The prompt identity is in the table note. Secondary aggregates follow the declared scope. The normal-video artefact rate has full denominators. Grey-pair scorer agreement is aggregated (§7.2, §8). |
| Records | `primary.json` count fields follow `RECORDS.md`. Cases runs write a manifest with video lengths, so the validator can check VERA's window arithmetic (`RECORDS.md` v1.1). |
| Integration | A retrieval rehearsal on the development video Normal_Videos_018 was added (§10). |

**Independent validator.** It was amended for the v1.1 native-margin rule only: `validate3b.py` reads
`logit_1 − logit_0`. Its two pin tests were updated to match; all its other code is as its independent author left
it.

## Amendments after v1.1

None.
