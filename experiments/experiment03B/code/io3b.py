"""experiment03B file I/O: append-only JSONL that survives hard kills, and atomic JSON writes.

A kill during an append can leave a partial last line. Readers ignore a final segment that does not end with a
newline; the first append of a process to such a file moves the fragment to <file>.torn.<n> (kept for audit) and
truncates back to the last complete line, so a resume never appends onto a fragment.
"""
import json
import os
import tempfile
from pathlib import Path

_checked = set()


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    parts = text.split("\n")  # "\n" only: str.splitlines() would also split on U+2028/U+0085 inside a record
    complete = parts if text.endswith("\n") or not text else parts[:-1]
    return [json.loads(l) for l in complete if l.strip()]


def _repair_tail(p):
    if p in _checked:
        return
    _checked.add(p)
    if not p.exists() or p.stat().st_size == 0:
        return
    data = p.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    n = 0
    while Path(f"{p}.torn.{n}").exists():
        n += 1
    Path(f"{p}.torn.{n}").write_bytes(data[cut:])
    with open(p, "r+b") as f:
        f.truncate(cut)


def append_jsonl(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    _repair_tail(p)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True, allow_nan=True) + "\n")
        f.flush()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def write_text_atomic(p, text):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, p)
