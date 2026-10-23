# Shared external artifacts (provenance only)

Experiments share one copy of the model, videos, frame caches, features and reference outputs under `shared/` on
the host. The data is **not** in Git: it is large, comes from third parties with their own terms, and includes
dataset media. Git tracks only the provenance and identity files that let you fetch and verify each artifact. The
whitelist is the `shared/` block in the root `.gitignore`.

| Local path | Bytes | sha256 | Tracked identity file | Source |
|---|---:|---|---|---|
| `models/OpenGVLab/InternVL2-8B/6fb9ad6924f69424e57fab2ab061d707688f0296/` (23 files, 16.15 GB) | — | per file in `MODEL_PROVENANCE.json` (all verified against the hub) | `MODEL_PROVENANCE.json` | Hugging Face `OpenGVLab/InternVL2-8B` at that revision; fetch with `setup/scripts/fetch_hf_model.py` |
| `videos/ucf_crime/RoadAccidents127_x264.mp4` | 19,361,185 | `618a6a63273de405bb70b4af61aa1d7d0ffb279dd41e531b1575332be48e06ff` | `*.mp4.provenance.json` (archive member, CRC, fetch method) | UCF-Crime `Anomaly-Videos-Part-3.zip`, one member range-fetched with `setup/scripts/fetch_zip_member.py` |
| `frames/ucf_crime/RoadAccidents127_x264/` (2,580 JPEGs, 73.1 MB) | — | digest `d14b4bf8…` over the per-frame list | `RoadAccidents127_x264.frames.json`, `.frames.sha256` | extracted locally with upstream `Preprocessing/extract_frames.py` |
| `features/vera_official_ucf_imagebind/RoadAccidents127_x264.npy` | 1,327,232 | `f0c2470ccdae5e6d8091dd3e41fe281f02243f37b80bfbf71fd0e0ed668eb657` | `*.npy.provenance.json` | VERA authors' UCF-Crime ImageBind features (Drive link in the upstream README), one member |
| `reference/vera_official_ucf_initial_scores/RoadAccidents127_x264.json` | 8,485 | `2c565bf7cbe663dccc78c7fd4b8ca065954296205e3fee6d620854b8792dbcb1` | `*.json.provenance.json` | VERA authors' released initial scores (Drive link in the upstream README), one member |
| `reference/vera_official_ucf_initial_scores/VERA_prediction_scores_UCF_Crime.zip` | 655,612 | `b7457cee812510ec2331c24cb5da451327c9500152e2437839d31af29166a3b0` | (recorded here) | the same release, the whole archive |

**Download links.** The UCF-Crime provenance files record the dataset's Dropbox folder link, including its `rlkey`.
This is the public link that Dropbox serves to anyone who opens the download link on the official project page,
https://www.crcv.ucf.edu/projects/real-world/. It is not a private credential. No expiring or signed redirect URL
was stored: only the redirect host name is recorded.

**Verifying.** Experiment configs pin these hashes, and the runners and validators check them before use. To
verify by hand, run `sha256sum <file>` and compare it with the table or the provenance file.
