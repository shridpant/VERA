"""Fingerprint ~/vera_research (read-only). Writes a TSV: sha256, bytes, mtime_ns, path.

Skips the conda prefix, the HF module cache, __pycache__ and the nested checkout's .git (its HEAD and status are
recorded separately). Model weight shards (*.safetensors) get size and mtime only ("-" for sha256); their hub
checksums are in MODEL_PROVENANCE.json.
"""
import hashlib, os, subprocess, sys
root = os.path.realpath(sys.argv[1]); out = sys.argv[2]
SKIP_DIRS = {"setup/environment/env", "shared/models/hf_home", "setup/source/VERA/.git"}
rows = []
for d, dirs, files in os.walk(root):
    rel = os.path.relpath(d, root)
    dirs[:] = sorted(x for x in dirs if x != "__pycache__" and os.path.normpath(os.path.join(rel, x)) not in SKIP_DIRS)
    for f in sorted(files):
        p = os.path.join(d, f); r = os.path.normpath(os.path.join(rel, f))
        st = os.lstat(p)
        if os.path.islink(p):
            rows.append(("link:" + os.readlink(p), st.st_size, st.st_mtime_ns, r)); continue
        if f.endswith(".safetensors"):
            rows.append(("-", st.st_size, st.st_mtime_ns, r)); continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
        rows.append((h.hexdigest(), st.st_size, st.st_mtime_ns, r))
rows.sort(key=lambda x: x[3])
with open(out, "w", encoding="utf-8") as fh:
    fh.write("sha256\tbytes\tmtime_ns\tpath\n")
    for r in rows: fh.write("%s\t%d\t%d\t%s\n" % r)
nested = os.path.join(root, "setup/source/VERA")
g = lambda *a: subprocess.run(["git", *a], cwd=nested, capture_output=True, text=True).stdout.strip()
with open(out + ".nested_git.txt", "w") as fh:
    fh.write(f"HEAD {g('rev-parse', 'HEAD')}\nbranch {g('rev-parse', '--abbrev-ref', 'HEAD')}\n"
             f"status_porcelain_all [{g('status', '--porcelain', '--untracked-files=all')}]\n"
             f"main {g('rev-parse', 'main')}\norigin/main {g('rev-parse', 'origin/main')}\n")
print(len(rows), "files")
