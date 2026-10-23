#!/usr/bin/env python3
"""Download one pinned Hugging Face model revision into shared/models and verify every file.

LFS files are checked against the hub's SHA-256 (lfs.oid); other files against their git blob SHA-1.
Files that already exist and verify are not downloaded again. Example files and logs are skipped.

Usage:
  python3 -I fetch_hf_model.py --repo OpenGVLab/InternVL2-8B \
      --revision 6fb9ad6924f69424e57fab2ab061d707688f0296 --out-root ~/vera_research/shared/models
Writes <out-root>/<org>/<name>/<revision>/ and MODEL_PROVENANCE.json beside the files.
Standard library plus curl.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

SKIP_PREFIXES = ("examples/",)
SKIP_SUFFIXES = (".log", ".mp4", ".jpg", ".png")


def file_ok(path, entry):
    if not os.path.isfile(path) or os.path.getsize(path) != entry["size"]:
        return False
    if entry.get("lfs"):
        h = hashlib.sha256()
        want = entry["lfs"]["oid"]
    else:
        h = hashlib.sha1(b"blob %d\0" % entry["size"])
        want = entry["oid"]
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 22), b""):
            h.update(buf)
    return h.hexdigest() == want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    if len(a.revision) != 40:
        sys.exit("--revision must be a full 40-character commit hash")

    api = f"https://huggingface.co/api/models/{a.repo}/tree/{a.revision}?recursive=1"
    with urllib.request.urlopen(api, timeout=60) as r:
        tree = [e for e in json.load(r) if e["type"] == "file"]
    keep = [e for e in tree if not e["path"].startswith(SKIP_PREFIXES) and not e["path"].endswith(SKIP_SUFFIXES)]
    out = os.path.join(os.path.expanduser(a.out_root), a.repo, a.revision)
    os.makedirs(out, exist_ok=True)

    for e in keep:
        dst = os.path.join(out, e["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if file_ok(dst, e):
            print(f"ok (cached) {e['path']}", flush=True)
            continue
        url = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/{e['path']}"
        part = dst + ".part"
        subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", part, url], check=True)
        os.replace(part, dst)
        if not file_ok(dst, e):
            sys.exit(f"checksum mismatch for {e['path']}")
        print(f"ok (downloaded) {e['path']} {e['size']}", flush=True)

    prov = {
        "repo": a.repo,
        "revision": a.revision,
        "source": f"https://huggingface.co/{a.repo}/tree/{a.revision}",
        "files": {e["path"]: {"bytes": e["size"], "sha256" if e.get("lfs") else "git_blob_sha1":
                              e["lfs"]["oid"] if e.get("lfs") else e["oid"]} for e in keep},
        "skipped": sorted(e["path"] for e in tree if e not in keep),
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out, "MODEL_PROVENANCE.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"all {len(keep)} files verified in {out}")


if __name__ == "__main__":
    main()
