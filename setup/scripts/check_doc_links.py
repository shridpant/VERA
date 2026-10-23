#!/usr/bin/env python3
"""Check relative links in the repository's Markdown files.

Every relative link target (`[text](path)`, external URLs and pure #anchors skipped) is resolved against the
linking file's directory and classified as:
  tracked     the target is a file or directory in the git index (or HEAD) of this repository;
  local-only  the target is missing from git but exists on disk (ignored or untracked artifacts, e.g. dataset images);
  broken      the target exists nowhere.
Exit status 1 if any link is broken. Writes JSON to --out.

    python3 setup/scripts/check_doc_links.py --out /tmp/doc_links.json [--include-untracked]
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent)).resolve()
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-untracked", action="store_true", help="also check untracked .md files (not ignored ones)")
    a = ap.parse_args()
    ls = lambda *x: subprocess.run(["git", "ls-files", "-z", *x], cwd=ROOT, capture_output=True, text=True).stdout.split("\0")
    tracked = {p for p in ls() if p}
    dirs = {str(Path(p).parent) for p in tracked} | {str(q) for p in tracked for q in Path(p).parents}
    docs = sorted(p for p in tracked if p.endswith(".md"))
    if a.include_untracked:
        docs += sorted(p for p in ls("--others", "--exclude-standard") if p.endswith(".md"))
    rows = []
    for doc in docs:
        text = (ROOT / doc).read_text(encoding="utf-8")
        text = re.sub(r"```.*?```", "", text, flags=re.S)  # links inside code blocks are not links
        for m in LINK.finditer(text):
            target = m.group(1)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            rel = os.path.normpath(os.path.join(os.path.dirname(doc), path)).rstrip("/")
            if rel in tracked or rel in dirs:
                kind = "tracked"
            elif (ROOT / rel).exists():
                kind = "local-only"
            else:
                kind = "broken"
            rows.append({"doc": doc, "target": target, "resolved": rel, "kind": kind})
    counts = {k: sum(r["kind"] == k for r in rows) for k in ("tracked", "local-only", "broken")}
    Path(a.out).write_text(json.dumps({"root": str(ROOT), "docs": len(docs), "links": len(rows), "counts": counts,
                                       "non_tracked": [r for r in rows if r["kind"] != "tracked"]}, indent=1) + "\n",
                           encoding="utf-8")
    print(f"{len(docs)} docs, {len(rows)} relative links: {counts}")
    for r in rows:
        if r["kind"] != "tracked":
            print(f"  {r['kind']}: {r['doc']} -> {r['target']}")
    sys.exit(1 if counts["broken"] else 0)


if __name__ == "__main__":
    main()
