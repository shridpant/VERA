"""Compare two fingerprint TSVs from fingerprint.py: report changed, added and removed paths (sha256 and size)."""
import sys
def load(p):
    rows = [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8")][1:]
    return {r[3]: (r[0], r[1]) for r in rows}
a, b = load(sys.argv[1]), load(sys.argv[2])
changed = sorted(p for p in a.keys() & b.keys() if a[p] != b[p])
added, removed = sorted(b.keys() - a.keys()), sorted(a.keys() - b.keys())
print(f"before {len(a)} files, after {len(b)} files; changed {len(changed)}, added {len(added)}, removed {len(removed)}")
for tag, ps in (("changed", changed), ("added", added), ("removed", removed)):
    for p in ps[:40]: print(f"  {tag}: {p}")
