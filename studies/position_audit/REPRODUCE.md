# CPU-only reproduction

Protocol freeze:819ba57. Verified inputs and initial implementation:3da8543. Absolute evidence-path invocation fix:bc7e03f. Exact-upstream short-video replay fix:3cce2c8. Both failed attempts remain in evidence. No shifted statistics were inspected to select these fixes. Source math and success criteria remain unchanged.

## Environment

Python3.14.2, NumPy2.4.1, SciPy1.18.0, scikit-learn1.9.0 on macOS; see reconstruction provenance for exact platform/BLAS. No PyTorch/model/CUDA dependency is needed. Set CUDA_VISIBLE_DEVICES to an empty string; set OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS and OMP_NUM_THREADS to4. The code also applies threadpoolctl limits. Both main stages enforce1,800-second alarms.

## External assets

Keep data outside Git. Obtain UCF feature Drive file1gwkgIKiPg6fpRaO5yXkxuRcN3X1-97LF from the upstream README. The connected Drive downloader cannot handle its310MB size; the public direct endpoint used was:

    curl --fail --location --retry 2 --max-time 600 'https://drive.usercontent.google.com/download?id=1gwkgIKiPg6fpRaO5yXkxuRcN3X1-97LF&export=download&confirm=t' -o ucf_crime_vision_features.zip

The already cached author score ZIP was copied from Oz without modification. Canonical release links are in the repository README. Check features ZIP SHA2566b3e0a17091524c1635b68f18f8b1c5daaf067192d194e27d39de758db5f29e4 and scores ZIP SHA256b7457cee812510ec2331c24cb5da451327c9500152e2437839d31af29166a3b0. Metadata and source hashes are in PROTOCOL.md. No expiring URL is required or recorded.

Use a fresh extraction directory. verify_assets.py validates CRCs, all290IDs, dimensions, row counts, finite values, nonzero norms, expected byte count and the previously verified RoadAccidents127 checksum:

    python3 studies/position_audit/verify_assets.py --root /absolute/repo --assets /absolute/assets --out /absolute/repo/studies/position_audit/evidence/new_assets.json

## Replay and statistics

Run check_metrics.py first. It compares whole-video multiplicity weighting with explicit duplicated-frame sklearn calculations, including ties, constant scores and circular permutations. The reference runner stages only paths and runs the unchanged upstream compute_auc.py.

Use absolute paths, fresh output directories and capture stdout/stderr. The runner changes working directory while executing the reference source; a relative evidence path was the preserved first-launch failure.

    python3 studies/position_audit/reconstruct.py --root /absolute/repo --assets /absolute/assets --out /absolute/new-cache --evidence /absolute/new-evidence
    python3 studies/position_audit/analyze.py --manifest /absolute/new-evidence/reconstructed_manifest.json --out /absolute/new-results

The manifest contains absolute local cache paths; relocating requires a documented path-only rewrite or regeneration. Arrays stay outside Git; per-file and per-array hashes are committed. Do not mix outputs from failed/restarted runs. The final reconstruction must match all upstream scores and labels before analysis. The initial output cache was incomplete; successful cache is recorded as vera-position-reconstruction-v3.

plot.py regenerates scientific figures and tables from the complete summary. Read RESULTS/report sources for the pass/fail decision and limits. No GPU work or automatic XD confirmation follows from this audit.
