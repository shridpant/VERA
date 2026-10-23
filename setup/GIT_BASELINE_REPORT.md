# Git baseline report: setup, experiment01 and experiment02 in `shridpant/VERA`

Written on 2026-10-23, when this baseline was put under version control. Three things are recorded outside this
file, so that no commit has to describe itself:
- the final commit SHAs;
- the push and PR state;
- the re-run of the checks on the exact final commit.

They are in the finalization note (`~/vera_research_gitwork/FINALIZATION.md` on Oz, with a copy in the Mac
mirror) and in the descriptions of the stacked pull requests.

## 1. Summary

- `~/vera_research` on Oz (`/home/stu11/s5/sp2117/vera_research`) is the working tree of the fork
  [`shridpant/VERA`](https://github.com/shridpant/VERA).
- The upstream history is kept unchanged up to `15b8bcb8`. On top of it sit four commits:
  - the original source change `008ed401`, which the runs executed;
  - Commit A: setup and experiment01;
  - Commit B: experiment02;
  - Commit C: the erratum, the source-for-new-runs migration, and these records.
- Every commit is an import of work that was already done. Nothing was re-run to make them, and no model was
  loaded.
- The data stays local. That covers the model, videos, frames, features, the authors' archives and the images
  derived from dataset frames. Git holds their provenance, sizes and hashes instead.
- The experiment03 drafts are untouched and untracked.

## 2. Repository and connection identifiers (no credentials)

| Item | Value |
|---|---|
| Host / login | `oz.gccis.rit.edu` / `sp2117` (see [`connection.md`](connection.md)) |
| Project path | `/home/stu11/s5/sp2117/vera_research` |
| Fork | `https://github.com/shridpant/VERA`. Public; default branch `main`; GitHub parent `vera-framework/VERA`. |
| Fork state before this work | only `main`, at `15b8bcb8574a977c229c577f50bfe6f06d07106e`, which is upstream `main`. No other branches and no pull requests. Nothing in the fork needed preserving or merging. |
| Upstream | `https://github.com/vera-framework/VERA`, `main` = `15b8bcb8…` (43 commits) |
| Remotes in the working tree | `origin` = the fork, `upstream` = vera-framework |
| Commit identity | `Shrid Pant <sp2117@rit.edu>` (Oz `git config`), with a Claude co-author trailer |

**Push route.** Oz has no GitHub credentials: the credential store file is empty, there is no `gh`, and there is
no SSH key pair. A dry-run push from Oz fails with
`fatal: could not read Username for 'https://github.com': terminal prompts disabled`. The Mac's existing keychain
credential authenticates as `shridpant`, with push and admin rights on the fork. So the commits are created on Oz.
After this commit they are carried to a scratch clone on the Mac as a `git bundle` and pushed from there. Git object IDs are content
hashes, so the SHAs are the same on both machines. The Mac mirror is not a Git checkout.

## 3. Parentage and commit boundaries

```text
15b8bcb8  upstream main (fork main unchanged)
  └ 008ed401  vera-research: importable one-video inference fixes for generate_initial_score.py   [existing commit]
      └ da469b92  A  Establish reproducible VERA setup and one-video baseline
          └ f12cacc9  B  Add temporal attribution feasibility experiment and results
              └ (this commit)  C  Record experiment02 precision erratum and pin tracked VERA source for new runs
```

| Commit | Files | Bytes added/changed | Contents |
|---|---|---:|---|
| `008ed401` | 1 M | 11,137 | `generate_initial_score.py`. This is the exact source revision the historical configs name as `expected_head`, made on 2026-10-22 before experiment01. It is carried over, not re-created. |
| A `da469b92` | 145 A, 1 M | 2,511,088 | the integrated `README.md`, `.gitignore`, `EXPERIMENTS.md` (experiment01 only), `setup/` (environment specs, scripts, patch, logs, status, connection, source README, ledger rows up to experiment01), `shared/` provenance, `experiments/experiment01/` (both runs) |
| B `f12cacc9` | 138 A, 5 M | 4,021,508 | `experiments/experiment02/` (code, brief, config, both runs), the Normal_Videos_018 provenance and fetch log, registry row, setup-status note, ledger rows for experiment02, `.gitignore` lines for experiment02's local-only images |
| C (this commit) | see `git show --stat` | — | `experiments/experiment02/ERRATA.md` and `errata/`; `setup/scripts/{pin_tracked_source.sh, check_source_resolution.py, check_doc_links.py}`; README, source README, setup status and registry updates; `setup/git_baseline/`; this report |

Commits A and B add 6.53 MB of blobs. The largest file is `experiments/experiment02/runs/run_002/results/perturbation_responses.jsonl`
(1.58 MB) and no other file exceeds 1.1 MB. The only binary files are 13 small numeric `.npy` arrays (≤ 33 KB).
The repository's packed history is under 3 MB.

**How the A/B split was established, without guessing:**
- **Setup files.** `setup/scripts/*`, `requirements.txt`, `pip_freeze.txt` and the patch match the hashes recorded
  before experiment02 began (`experiments/experiment02/runs/*/preflight/setup_files_before.sha256`): 13/13.
- **experiment01.** Its files match `experiment01_files_before.sha256`: 71/71. No experiment01 file was modified
  after that preflight.
- **Registry and status docs.** `EXPERIMENTS.md` and `setup/SETUP_STATUS.md` changed during experiment02, so
  commit A carries their experiment01-era texts, rebuilt from the session record. Replaying the recorded
  experiment02 edits reproduces the current `SETUP_STATUS.md` byte for byte. It reproduces `EXPERIMENTS.md` too,
  except for the experiment02 row, which was claim-corrected later; commit B carries the current files.
- **Setup logs.** The archive listings for UCF parts 1, 2 and 4 were written later, for the experiment03 draft,
  and are left out. The Normal_Videos_018 fetch log and listing date from experiment02 and are in B.
- **Job ledger.** It is append-only. A carries its first five lines (the header, two smoke tests and the two
  experiment01 runs); B carries the whole file.
- **Written at import time.** These are new: `README.md` (integration), `.gitignore`, `setup/source/README.md` and
  `shared/README.md`.

## 4. Which source runs import

**Historical runs (experiment01, experiment02).** They imported `generate_initial_score.py` from
`setup/source/VERA`. That is a separate clone with its own `.git`, on branch `vera-research-local` at `008ed401`.
It stays in place and is ignored (`/setup/source/*/`), so it is never committed as an embedded repository or
gitlink. Its configs and run manifests are unedited.
- **Equivalence.** `008ed401` is now in this repository's history. The clone's tree equals `008ed401^{tree}`
  (`41c689ac…`), and the clone is clean.
- **Same files at the root.** All 26 files of `008ed401` are identical at the repository root; `README.md` was
  deliberately integrated. See [`source/README.md`](source/README.md) for file hashes and for how to recreate the
  checkout from this repository with `git worktree`.

**New runs.** These import a pinned, detached worktree of this repository, `setup/source/tracked-<sha12>`.
- **Making the pin.** [`scripts/pin_tracked_source.sh`](scripts/pin_tracked_source.sh) creates the worktree. By
  default it pins the last commit that changed the VERA source (`008ed401` today) and prints the config's
  `source:` block.
- **Checking it.** [`scripts/check_source_resolution.py`](scripts/check_source_resolution.py) verifies, without
  loading weights:
  - the HEAD and cleanliness the runners require;
  - that the commit is in this history;
  - that the tree is equal;
  - byte identity with the tracked root;
  - that the import resolves to the pinned file.
- **Runners unchanged.** No runner was modified: the runtime-tested code in A and B is exactly what ran.
  experiment02's validator (V12) would also flag any change to `setup/scripts` files that existed before
  experiment02.

## 5. What is tracked, local, redacted and uncommitted

**Tracked:** source, configs, tests, environment specifications, compact manifests and checksums, reports,
compact raw records (JSONL), CSVs, small numeric arrays, and logs. The `pid` and lock files are left out.

**Local only (ignored).** These are listed with sizes and hashes in
[`git_baseline/local_only_artifacts.tsv`](git_baseline/local_only_artifacts.tsv):

| Item | Why it stays local |
|---|---|
| model weights and repo files (16.15 GB) | external; hub checksums in `MODEL_PROVENANCE.json` (tracked) |
| videos (2), frame caches (3,761 JPEGs), the ImageBind feature array, the authors' score archive and extracted JSON, the UCF annotation file | dataset or third-party data; provenance and hashes are tracked |
| experiment02 `results/example_inputs/*.png`, `spatial/preview_gradcam_global_norm.png` | images derived from dataset frames |
| experiment02 `spatial/pixel_grad_abs_448.npy` (3.2 MB) | image-like gradient maps of dataset frames |
| `setup/environment/env/`, `shared/models/hf_home/` | environment and cache |
| `setup/source/VERA/` | historical runtime checkout with its own `.git` (§4) |
| `setup/cleanup_inventory/`, `setup/cleanup_manifest.md`, `setup/scripts/cleanup_obsolete_experiments.sh` | records about unrelated folders on the account; not research artifacts |
| `setup/logs/setup_env*.nohup`, `pid` files, `.run.lock`, `setup/jobs/.submit.lock` | duplicates, and transient process files |

**Untracked and not ignored** (it shows in `git status` on purpose): `experiments/experiment03/`, a protocol draft
that has not been executed, plus the archive listings `setup/logs/ucf_part{1,2,4}_listing.*` that were made for
it. They belong to later commits. The `.gitignore` rules are path-specific, so no experiment folder is ignored
wholesale.

**Redaction: none was needed.** Every staged file was scanned for keys, tokens, passwords, private keys and signed
or expiring URLs, and none was found.
- **Dropbox links.** The UCF-Crime provenance records keep the dataset's Dropbox folder link with its `rlkey`.
  This is the public link Dropbox serves to anyone who opens the official project page's download link; the
  hash of the anonymously served key matches. It is not a credential.
- **Google Drive links.** The Drive URLs are the public file IDs from the upstream README.
- **Hardware identifiers.** GPU UUIDs, the host name and the account's home path appear in run records. They
  are identifiers, not credentials. Editing them out would have broken the recorded hashes of the raw records.

**Root README.** The pre-integration research README is kept on Oz at
`~/vera_research_gitwork/backup_docs/README.md.orig` (sha256 `60258536…`). So are the pre-commit-C copies of
`EXPERIMENTS.md`, `setup/SETUP_STATUS.md` and `experiments/experiment02/README.md`. Its content is integrated into
the new `README.md`, which reproduces the upstream README verbatim. The upstream repository has no licence file;
the README says so and keeps the attribution and citation.

## 6. Validation

All validation runs on CPU, with no model inference. Each run writes only to a new output directory; validators
and analyzers run on copies of the run directories.
- [`git_baseline/validate_baseline.sh`](git_baseline/validate_baseline.sh): `R=<repo> O=<new dir> bash setup/git_baseline/validate_baseline.sh`
- [`git_baseline/cleanclone_test.sh`](git_baseline/cleanclone_test.sh): `SRC_REPO=<repo> BRANCH=research/validated-baseline C=<new dir> O=<new dir> bash setup/git_baseline/cleanclone_test.sh`
- [`git_baseline/fingerprint.py`](git_baseline/fingerprint.py) and [`git_baseline/compare_preservation.py`](git_baseline/compare_preservation.py): fingerprint the live tree and compare two fingerprints

**Results on the candidate `f955fec6`.** Its tree equals this commit's except for this report,
`cleanclone_test.sh`, the preservation scripts and `git_baseline/validation/`. The summaries are in [`git_baseline/validation/`](git_baseline/validation).

| Check | Result |
|---|---|
| Ancestry `15b8bcb8 ← 008ed401 ← A ← B ← C`; fork `main` still `15b8bcb8` | PASS |
| Source equivalence: all 26 files of `008ed401` identical at HEAD (README integrated); live `setup/source/VERA` tree = `41c689ac`, clean | PASS |
| No gitlinks, no nested `.git`, no new file > 2 MB, no caches/weights/videos/frames/archives tracked | PASS |
| Secret and signed-URL scan of the whole tree | PASS (no hits) |
| CPU tests `setup/scripts/test_vera_pipeline.py`, `experiments/experiment02/code/test_attr_lib.py`; every tracked `.py` compiles | PASS |
| experiment01 `validate_run.py` on copies of run_001 (`--require-report`) and run_002; `summarize_run.py` | PASS; outputs byte-identical to the originals |
| experiment02 `validate_attr.py` (V01–V18), `analyze_attr.py`, `spatial_border_check.py` on copies of run_002 | PASS; outputs byte-identical to the originals |
| Erratum measurement `errata/precision_crossings.py` | PASS (numbers in [`ERRATA.md`](../experiments/experiment02/ERRATA.md)) |
| Source resolution: experiment01 and experiment02 configs, plus a pinned new-run config; S1–S6 including the import | PASS (18/18) |
| Documentation links | 25/26. The one failure was the link to this report, which did not exist yet in the candidate. |
| Preservation: the live tree re-fingerprinted after validation against the pre-change fingerprints ([`git_baseline/preserved_files_before.tsv`](git_baseline/preserved_files_before.tsv); full list kept on Oz) | 4,148 files before and after: 0 changed, 0 added, 0 removed; nested checkout state identical |

**Clean clone of the candidate.** Fresh clone, tracked files only; the conda env interpreter is the one external
input.
- **PASS:**
  - all 319 tracked files present, and the clone is clean;
  - everything compiles;
  - the historical checkout recreated from the repository's own history with `git worktree add … 008ed401`;
  - `test_vera_pipeline.py`;
  - the pin helper, and source resolution S1–S6 for all three configs;
  - the erratum JSON reproduced byte for byte;
  - `analyze_attr.py` on the tracked records reproduces the tracked results byte for byte.
- **Expected failures (local data needed):**
  - `test_attr_lib.py` needs the frame cache;
  - experiment01 `validate_run.py` passes C01–C04 and C06–C10, then stops at C05/C11, which need the video,
    frames and features;
  - experiment02 `validate_attr.py` passes V01, V03–V11 and V13–V18; V02 needs the frames and video, and V12
    needs the local-only cleanup script that its pre-experiment02 fingerprint lists.
  - Also the same documentation link as above.

So a clean clone is **not** fully reproducible. It verifies the code paths, the source identity and the recorded
results. Re-running the historical validators in full, or any inference, needs the external artifacts in
`shared/README.md`, the conda env (`setup/environment/REBUILD.md`), and for V12 the local cleanup script.

## 7. Scientific and precision caveats

- **What the commits claim.** They import completed runs and make no new scientific claim. The validators check
  implementation properties, not explanation faithfulness.
- **experiment02's numeric score.** It is an auxiliary label-only condition with a post-hoc gate (run_002's
  scoring contract). VERA's original output is the binary label.
- **The grouping benefit is not established.** Adjacency was confounded with evidence content; see run_002's
  `REPORT.md` and `NEXT_STEP.md`.
- **Erratum E1.** S_hi changes smaller than one bf16 step can still move the deployed bf16 score: 36 of 109
  sub-step rows did, and in 4 rows the two deltas have opposite signs. The run documents are not edited; see
  [`../experiments/experiment02/ERRATA.md`](../experiments/experiment02/ERRATA.md).

## 8. Publication

**Branches.** The plan, carried out after this commit was created: push without force as three stacked
branches, each reviewed as a draft pull request against the branch below it. Nothing is merged, and `main` is
untouched.

| Branch | Tip | Draft PR base |
|---|---|---|
| `research/validated-baseline-1-setup-experiment01` | A (includes `008ed401`) | `main` |
| `research/validated-baseline-2-experiment02` | B | `research/validated-baseline-1-setup-experiment01` |
| `research/validated-baseline` | C | `research/validated-baseline-2-experiment02` |

The verified remote refs, commit URLs and PR links are in the finalization note and the PR descriptions.

## 9. Working tree after promotion

**How the live tree is promoted** (done after this commit was created). Nothing in the live tree is reset, cleaned
or checked out over.
1. A clone of the prepared commits is made with `--no-checkout`. Its `.git` is moved into `~/vera_research`.
2. The index was filled from HEAD (`git reset`, mixed; this leaves files alone).
3. Only paths missing from the live tree are checked out: the upstream source, the new docs and the scripts.
4. Four docs that this baseline deliberately changes are replaced after backing them up (§5): `README.md`,
   `EXPERIMENTS.md`, `setup/SETUP_STATUS.md` and `experiments/experiment02/README.md`.

**Expected `git status`.** Nothing modified or deleted. Untracked: `experiments/experiment03/` and
`setup/logs/ucf_part{1,2,4}_listing.*`. The verified status is in the finalization note.

## 10. Inspecting the baseline and continuing

```bash
cd ~/vera_research
git log --oneline --graph 15b8bcb8..research/validated-baseline   # the four commits on top of upstream
git show --stat <sha>                                              # one commit's file boundary
git status --short                                                 # experiment03 drafts show as untracked
bash setup/git_baseline/validate_baseline.sh                       # with R=$PWD O=/tmp/<new dir>
```

**To add an experiment:**
1. Create `experiments/experimentNN/` and its registry row.
2. Pin the source with `bash setup/scripts/pin_tracked_source.sh`, and paste the `source:` block into the config.
3. Check it with `setup/scripts/check_source_resolution.py`, then run through `setup/scripts/launch.sh`.
4. Commit the experiment with explicit paths on a new branch from `research/validated-baseline`, for example
   `git add -- experiments/experimentNN EXPERIMENTS.md`.

Never `git add -A` at the root: local data is ignored, but the experiment03 drafts are deliberately untracked.
