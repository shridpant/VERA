# Selection checks (protocol v0.3–v0.4; last run 2026-10-23; before any model call)

These checks were rerun on the v0.3 outputs. The v0.2 record checked only the 30 primary windows and `windows.json`, on
a pre-v0.2 version of the selection.

| Check | Scope | Result |
|---|---|---|
| Frame indices and window bounds vs `setup/scripts/vera_pipeline.window_for_anchor` (numpy 1.26.4, Python 3.10.22, VERA arithmetic), run on oz | all 133 records: 30 windows + 103 pre-listed backups | 133/133 identical |
| Reproducibility: `select_windows.py` rerun on oz (Python 3.13.5) and on the Mac (Python 3.14.2), each into a fresh directory | all four outputs | byte-identical; hashes below |
| Replacement step: `test_replacement.py` (slot-stable IDs, class exhaustion, next class, S3 backup, backup-record equality, unknown id rejected, dropped S1/S2 pair, dropped S3 slot) | oz (3.13.5) and the Mac (3.14.2) | ALL PASS on both (rerun after the v0.4 dropped-slot cases were added) |
| Backup S2 records name their S1 (`paired_with`) | 49 backup S2 records | 49/49 |
| Unchanged by v0.3 | `windows.json`, `windows.csv` | byte-identical to v0.2 (`2e833576…`, `9ff9b70b…`) |
| Inputs | — | sha256 values in `selection_log.json` → `inputs` |

On oz, the scripts were copied through stdin into a temporary directory under `/tmp` and run from there; it was
removed afterwards. No file under `~/vera_research` on oz was written.

**sha256 of the checked files:**

| File | sha256 |
|---|---|
| `select_windows.py` | `020e083bcb3584bf1816288ba77c5fcaa884b76d46d8e9005ccecb575f7bf57a` |
| `test_replacement.py` | `857b61d909c8a5d92302cccfb9fbe2f6e373bf688ef7ec0c7a430d4f53cb9752` |
| `windows.json` | `2e83357694a46289bccba7353dd7a0d359935e4e15d7023f7e1b29b73bb580b9` |
| `backup_windows.json` | `1aba9fe759cf9b8f7ca47b5e6763318daf8ece47cf252f03440b0a1dbdce192d` |
| `selection_log.json` | `b92c6007d743b5e03211a9f69bd2feb47b5164bdcd7fbd76ed8dd49a63bb0953` |
| `windows.csv` | `9ff9b70b632e4de123e841264a39d9004ba952ae7acfbf5cd14398d3af0576fd` |

The §12 validator must check these hashes against the frozen files, so that any later regeneration makes this record
visibly stale.

**Inputs used:** metadata only, about 2 MB in total.

- The released-label zip, 655,612 bytes. It is stored at
  `shared/reference/vera_official_ucf_initial_scores/VERA_prediction_scores_UCF_Crime.zip` on oz and on the Mac.
- Five official zip directory listings, in `setup/logs/ucf_*_listing.tsv`. Listing them fetched about 1.4 MB over HTTP
  range requests.

No video was downloaded and no model was loaded.
