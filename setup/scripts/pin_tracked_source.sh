#!/usr/bin/env bash
# Pin this repository's tracked VERA source for new runs and print the matching config `source` block.
#
#   bash setup/scripts/pin_tracked_source.sh            # pin the last commit that changed the VERA source
#   bash setup/scripts/pin_tracked_source.sh <commit>   # pin a specific commit
#
# Creates setup/source/tracked-<sha12>, a detached git worktree of this repository (ignored by .gitignore; the
# runners read it but never write into it). "VERA source" means every tracked file outside the research additions
# (setup/, shared/, experiments/, README.md, EXPERIMENTS.md, .gitignore). Pinning the commit that last changed those
# files keeps a run manifest's `local_patch_vs_base` stat limited to source changes. Verify the result with
# setup/scripts/check_source_resolution.py.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
if [ $# -ge 1 ]; then
  rev=$(git rev-parse --verify "$1^{commit}")
else
  rev=$(git log -1 --format=%H HEAD -- . ':(exclude)setup' ':(exclude)shared' ':(exclude)experiments' \
        ':(exclude)README.md' ':(exclude)EXPERIMENTS.md' ':(exclude).gitignore')
fi
git merge-base --is-ancestor "$rev" HEAD || { echo "$rev is not an ancestor of HEAD" >&2; exit 1; }
dir=setup/source/tracked-${rev:0:12}
if [ -e "$dir" ]; then
  [ "$(git -C "$dir" rev-parse HEAD)" = "$rev" ] || { echo "$dir exists but is not at $rev" >&2; exit 1; }
else
  git worktree add --detach "$dir" "$rev" >&2
fi
[ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ] || { echo "$dir has local changes" >&2; exit 1; }
cat <<EOT
source:
  repo: https://github.com/shridpant/VERA
  base_commit: 15b8bcb8574a977c229c577f50bfe6f06d07106e
  checkout: $dir
  expected_head: $rev
EOT
