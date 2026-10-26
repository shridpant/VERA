"""Append-only, fsynced JSONL files and the study's unit ledger.

A unit is one model call: (video, anchor, condition). ledger.jsonl gets a 'start' event before a
call and an 'end' event (complete, failed or timeout) after it; a start without an end is an
interrupted attempt. Every attempt's full record goes to records/<condition>.jsonl. Nothing is
rewritten or deleted, so every failed or retried attempt stays on disk.
"""
import datetime
import json
import os
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def unit_key(video, anchor, condition):
    return f'{video}|{anchor}|{condition}'


def seal(path):
    """Make an existing JSONL file end with a complete line before anything is appended.

    A process killed mid-write leaves a last line without its newline. If that line is complete
    JSON, only the newline is added. Otherwise its bytes move to '<file>.torn.<n>' and the file is
    cut back to its last complete line, so later appends never fuse with a fragment. Returns the
    sidecar path or None."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = path.read_bytes()
    if data.endswith(b'\n'):
        return None
    cut = data.rfind(b'\n') + 1
    tail = data[cut:]
    try:
        json.loads(tail.decode('utf-8'))
        with path.open('ab') as f:
            f.write(b'\n')
        return None
    except (UnicodeDecodeError, json.JSONDecodeError):
        n = 1
        while path.with_name(f'{path.name}.torn.{n}').exists():
            n += 1
        sidecar = path.with_name(f'{path.name}.torn.{n}')
        sidecar.write_bytes(tail)
        with path.open('r+b') as f:
            f.truncate(cut)
        return sidecar


class JsonlAppender:
    """One open handle; each line is flushed and fsynced before the call returns."""
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.sealed = seal(path)
        self.handle = open(path, 'a', encoding='utf-8')

    def write(self, record):
        self.handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        self.handle.close()


def read_jsonl(path):
    """Records of a JSONL file. A torn last line (process killed mid-write) is skipped and
    reported; a bad line anywhere else is an error."""
    path = Path(path)
    if not path.exists():
        return [], None
    lines = path.read_text(encoding='utf-8').split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    out, torn = [], None
    for n, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if n != len(lines) - 1:
                raise ValueError(f'{path}: corrupt line {n + 1}')
            torn = line[:200]
    return out, torn


class Ledger:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / 'ledger.jsonl'
        events, self.torn = read_jsonl(self.path)
        self.attempts, self.ends = {}, {}
        for e in events:
            if e['event'] == 'start':
                self.attempts[e['unit']] = max(self.attempts.get(e['unit'], 0), e['attempt'])
            elif e['event'] == 'end':
                self.ends.setdefault(e['unit'], []).append(e)
        self.out = JsonlAppender(self.path)

    def start(self, unit, session):
        attempt = self.attempts.get(unit, 0) + 1
        self.attempts[unit] = attempt
        self.out.write({'t': now(), 'event': 'start', 'unit': unit, 'attempt': attempt, 'session': session})
        return attempt

    def end(self, unit, attempt, session, status, **fields):
        event = {'t': now(), 'event': 'end', 'unit': unit, 'attempt': attempt, 'session': session, 'status': status, **fields}
        self.out.write(event)
        self.ends.setdefault(unit, []).append(event)

    def note(self, session, **fields):
        self.out.write({'t': now(), 'event': 'note', 'session': session, **fields})

    def close(self):
        self.out.close()
