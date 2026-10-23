#!/usr/bin/env python3
"""Extract one member of a remote zip archive over HTTP byte ranges, without downloading the archive.

Only the end-of-central-directory, the central directory and the selected member's bytes are fetched.
The member's CRC-32 is checked by zipfile while reading, and its SHA-256 is recorded.

Usage:
  python3 -I fetch_zip_member.py --url URL --list
  python3 -I fetch_zip_member.py --url URL --member PATH_IN_ZIP --out FILE

Writes FILE and FILE.provenance.json (URL, archive size, member name, sizes, CRC, SHA-256, bytes fetched).
Standard library only.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

UA = {"User-Agent": "Mozilla/5.0 (vera_research fetch_zip_member)"}
BLOCK = 1 << 20


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, fetched in 1 MiB-aligned blocks via HTTP Range."""

    def __init__(self, url):
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                raise RuntimeError(f"server ignored Range (HTTP {r.status}); refusing a full download")
            self.size = int(r.headers["Content-Range"].split("/")[-1])
            self.resolved_host = urllib.request.urlparse(r.geturl()).netloc
        # Dropbox's redirect target is single-use (403 on reuse), so every range request starts from the original URL.
        self.url = url
        self.pos = 0
        self.cache = {}
        self.fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            lo, hi = i * BLOCK, min((i + 1) * BLOCK, self.size) - 1
            req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != hi - lo + 1:
                raise IOError(f"short range read {lo}-{hi}: got {len(data)} bytes")
            self.fetched += len(data)
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[i] = data
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            b = self._block(self.pos // BLOCK)
            off = self.pos % BLOCK
            chunk = b[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--member")
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    rf = RangeFile(a.url)
    zf = zipfile.ZipFile(rf)
    if a.list:
        for zi in zf.infolist():
            print(f"{zi.file_size}\t{zi.filename}")
        print(f"# archive bytes {rf.size}, members {len(zf.infolist())}, fetched {rf.fetched}", file=sys.stderr)
        return
    if not (a.member and a.out):
        ap.error("--member and --out are required unless --list")
    matches = [zi for zi in zf.infolist() if zi.filename == a.member]
    if len(matches) != 1:
        sys.exit(f"expected exactly one member named {a.member!r}, found {len(matches)}")
    zi = matches[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    part = a.out + ".part"
    h = hashlib.sha256()
    with zf.open(zi) as src, open(part, "wb") as dst:  # ZipExtFile verifies CRC-32 at EOF
        while True:
            buf = src.read(BLOCK)
            if not buf:
                break
            h.update(buf)
            dst.write(buf)
    if os.path.getsize(part) != zi.file_size:
        sys.exit(f"size mismatch: wrote {os.path.getsize(part)}, central directory says {zi.file_size}")
    os.replace(part, a.out)
    prov = {
        "source_url": a.url,
        "resolved_url_host": rf.resolved_host,
        "archive_bytes": rf.size,
        "archive_members": len(zf.infolist()),
        "member": zi.filename,
        "member_bytes": zi.file_size,
        "member_compressed_bytes": zi.compress_size,
        "member_crc32": f"{zi.CRC:08x}",
        "member_date_time": list(zi.date_time),
        "sha256": h.hexdigest(),
        "bytes_fetched_over_http": rf.fetched,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "HTTP Range reads of the zip central directory and this member only (fetch_zip_member.py)",
    }
    with open(a.out + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
