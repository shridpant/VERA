#!/usr/bin/env python3
"""Replacement-step tests for select_windows.py (protocol v0.3 §3). Standard library only; metadata only.

Checks:
  1. a rerun without exclusions reproduces the committed windows.json, windows.csv and backup_windows.json byte for byte;
  2. excluding a slot's primary video uses that slot's first backup, and keeps every other window identical (slot-stable IDs);
  3. exhausting a class (Abuse: primary + its only backup) moves the slot to the first next class, keeping the slot ID;
  4. exhausting that next class's first candidate too uses its second candidate;
  5. excluding an S3 video uses the first S3 backup in that slot only;
  6. every replacement record equals its pre-listed backup record, apart from the ID and the replacement fields;
  7. an --exclude id that no slot can use is rejected;
  8. exhausting a slot's class and every next class drops that S1/S2 pair (logged); all other windows are unchanged;
  9. exhausting an S3 slot and all 5 S3 backups drops that S3 slot (logged); all other windows are unchanged.

Usage: python3 -I -B test_replacement.py --root ~/Documents/research/vera_research
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPL_FIELDS = {"window_id", "paired_with", "replaces_video", "backup_record"}


def run(root, out, exclude=()):
    cmd = [sys.executable, "-I", "-B", str(HERE / "select_windows.py"), "--root", str(root), "--out", str(out)]
    if exclude:
        cmd += ["--exclude", *exclude]
    return subprocess.run(cmd, capture_output=True, text=True)


def load(d):
    w = {r["window_id"]: r for r in json.loads((d / "windows.json").read_text())["windows"]}
    b = {r["window_id"]: r for x in json.loads((d / "backup_windows.json").read_text())["backups"]
         for r in x.values() if isinstance(r, dict)}
    return w, b, json.loads((d / "selection_log.json").read_text())


def strip(r):
    return {k: v for k, v in r.items() if k not in REPL_FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = Path(ap.parse_args().root).expanduser()
    fails = []

    def check(name, ok):
        print(("PASS " if ok else "FAIL ") + name)
        if not ok:
            fails.append(name)

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        r = run(root, t / "base")
        check("1 rerun exits 0", r.returncode == 0)
        for f in ("windows.json", "windows.csv", "backup_windows.json"):
            check(f"1 {f} byte-identical to committed", (t / "base" / f).read_bytes() == (HERE / f).read_bytes())
        base, bk, log = load(t / "base")
        abuse_slot = f"S1-{log['slot_classes'].index('Abuse') + 1:02d}"
        k = abuse_slot[3:]
        cases = {
            "2 primary excluded -> own backup": ([base[abuse_slot]["video_id"]], {abuse_slot: f"S1-{k}b1"}),
            "3 class exhausted -> next class": ([base[abuse_slot]["video_id"], bk[f"S1-{k}b1"]["video_id"]],
                                                {abuse_slot: "S1-next1b1"}),
            "4 next class's first candidate excluded too": ([base[abuse_slot]["video_id"], bk[f"S1-{k}b1"]["video_id"],
                                                             bk["S1-next1b1"]["video_id"]], {abuse_slot: "S1-next1b2"}),
            "5 S3 excluded -> first S3 backup": ([base["S3-03"]["video_id"]], {"S3-03": "S3-b1"}),
        }
        for name, (excl, expect) in cases.items():
            out = t / name.split()[0]
            r = run(root, out, excl)
            check(f"{name}: exits 0", r.returncode == 0)
            if r.returncode:
                print(r.stderr)
                continue
            w, _, lg = load(out)
            check(f"{name}: same 30 slot IDs", sorted(w) == sorted(base))
            moved = {wid for wid in w if strip(w[wid]) != strip(base[wid])}
            pair = {f"S2-{s[3:]}" for s in expect if s.startswith("S1-")}
            check(f"{name}: only the expected slots changed", moved == set(expect) | pair)
            for slot, src in expect.items():
                check(f"{name}: {slot} equals backup {src}", strip(w[slot]) == strip(bk[src]))
                if slot.startswith("S1-"):
                    s2, src2 = f"S2-{slot[3:]}", "S2-" + src[3:]
                    check(f"{name}: {s2} equals backup {src2}", strip(w[s2]) == strip(bk[src2]))
            check(f"{name}: replacements logged", len(lg["replacements"]) == len(expect) and not lg["dropped_slots"])
        r = run(root, t / "bad", ["NoSuchVideo_x264"])
        check("7 unknown --exclude id rejected", r.returncode != 0)
        nxt = sorted({bk[b]["video_id"] for b in bk if b.startswith("S1-next")})
        s3b = sorted({bk[b]["video_id"] for b in bk if b.startswith("S3-b")})
        drops = {
            "8 class and all next classes exhausted -> pair dropped": (
                [base[abuse_slot]["video_id"], bk[f"S1-{k}b1"]["video_id"], *nxt], {abuse_slot, f"S2-{k}"}, [f"{abuse_slot}/S2-{k}"]),
            "9 S3 slot and all S3 backups exhausted -> slot dropped": ([base["S3-03"]["video_id"], *s3b], {"S3-03"}, ["S3-03"]),
        }
        for name, (excl, gone, logged) in drops.items():
            out = t / name.split()[0]
            r = run(root, out, excl)
            check(f"{name}: exits 0", r.returncode == 0)
            if r.returncode:
                print(r.stderr)
                continue
            w, _, lg = load(out)
            check(f"{name}: exactly the dropped IDs are missing", set(base) - set(w) == gone and not set(w) - set(base))
            check(f"{name}: all other windows unchanged", all(strip(w[i]) == strip(base[i]) for i in w))
            check(f"{name}: drop logged", lg["dropped_slots"] == logged)
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
