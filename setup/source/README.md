# Which VERA source the runs import

## Historical runs: `setup/source/VERA` (local checkout, not tracked)

experiment01 and experiment02 imported `generate_initial_score.py` from `setup/source/VERA`. This is a clone of
[`vera-framework/VERA`](https://github.com/vera-framework/VERA) with its own `.git`:

| Item | Value |
|---|---|
| Upstream base | `15b8bcb8574a977c229c577f50bfe6f06d07106e` (tree `e3a2e27b`) |
| Local branch | `vera-research-local` |
| HEAD | `008ed4019781c39e7cb16244b7500244f3986f92` (tree `41c689ac`). It is base + [`../patches/0001-…patch`](../patches). |
| Changed file | `generate_initial_score.py` only (88 insertions, 88 deletions) |

The repository root ignores this directory (`/setup/source/*/` in `.gitignore`), so it is never committed as an
embedded repository or gitlink. It stays in place because both experiments' configs name it:
`source.checkout: setup/source/VERA` and `source.expected_head: 008ed401…`. Before importing, the runners check that
its HEAD equals `expected_head` and that the checkout is clean. Those configs, and the run manifests that record
the same head, describe the original runs. They are not edited.

## Equivalence with the tracked code

Commit `008ed401` is itself part of this repository's history, directly on top of `15b8bcb8`. The repository root
therefore holds the same files as the checkout, byte for byte. The files the runs actually used:

| File | sha256 | git blob |
|---|---|---|
| `generate_initial_score.py` (patched) | `8447432b69909f3b1c7788c23679c42e7299ca423c29bd21bca27c8b85859224` | `bba11194` |
| `compute_auc.py` (unmodified; Steps 2–3) | `21e9705da3e8b04deb5855665a0c8fbc41e6baa705374f2c362b814d24b5eda8` | `aa7f26dc` |
| `Preprocessing/extract_frames.py` (frame cache extractor) | `79d709e5d190020b001c64d687083fc57429aa9f3e179c1742def01b77a74697` | `e822467d` |
| `Data/UCF_Eval.json` | `f3857cddd623f767e7b86bd50a4aeb122445fc6e241ddd7130b7e3ec0c4d8b9f` | `843d1a69` |

The unpatched upstream `generate_initial_score.py` has sha256 `6c17e5a1…`.

Check the local checkout against the tracked history, from the repository root:

```bash
git -C setup/source/VERA rev-parse HEAD                                 # 008ed4019781c39e7cb16244b7500244f3986f92
git -C setup/source/VERA status --porcelain --untracked-files=all       # empty
git rev-parse 008ed401^{tree} && git -C setup/source/VERA rev-parse 'HEAD^{tree}'   # both 41c689ac…
```

If the checkout is missing, you can recreate it at the same path from this repository's own history with
`git worktree add --detach setup/source/VERA 008ed4019781c39e7cb16244b7500244f3986f92`. The runners only check
HEAD and cleanliness.
