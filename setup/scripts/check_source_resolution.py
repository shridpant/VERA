#!/usr/bin/env python3
"""Check which VERA source an experiment config makes the runners import. Does not load model weights.

The runners (setup/scripts/run_experiment.py, experiments/experiment02/code/run_attr.py) import
generate_initial_score.py from ROOT / config["source"]["checkout"], after checking that the checkout's HEAD equals
config["source"]["expected_head"] and that it has no uncommitted changes to tracked files. For each config this
script resolves the same path and checks:

  S1  the checkout exists and is a git checkout;
  S2  HEAD == expected_head and no uncommitted changes to tracked files (the runners' own rule);
  S3  expected_head is a commit in this repository and an ancestor of this repository's HEAD;
  S4  the checkout's tree equals that commit's tree in this repository;
  S5  the VERA source files are identical (same file set, same git blobs) in the checkout and in this repository's
      tracked root at HEAD, and the root working copies of those files are unmodified, i.e. the pin is current;
  S6  with --import-python, `import generate_initial_score` (in a subprocess, CPU only, weights untouched)
      resolves to the checkout's file and exposes the inference classes.

"VERA source files" means every file tracked at HEAD outside the research additions (setup/, shared/,
experiments/, README.md, EXPERIMENTS.md, .gitignore).

    setup/environment/env/bin/python setup/scripts/check_source_resolution.py \
        --config experiments/experiment01/config.yaml --import-python setup/environment/env/bin/python --out /tmp/x.json

Exit status 1 if any check fails. Writes only to --out.
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent)).resolve()
RESEARCH_PATHS = ("setup/", "shared/", "experiments/")
RESEARCH_FILES = ("README.md", "EXPERIMENTS.md", ".gitignore")


def git(*args, cwd):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def vera_source_files(cwd, rev):
    rc, out = git("ls-tree", "-r", rev, cwd=cwd)
    files = {}
    for line in out.splitlines() if rc == 0 else []:
        meta, path = line.split("\t", 1)
        if path.startswith(RESEARCH_PATHS) or path in RESEARCH_FILES:
            continue
        files[path] = meta.split()[2]
    return files


def check_config(cfg_path, import_python):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    src = ROOT / cfg["source"]["checkout"]
    want = cfg["source"]["expected_head"]
    res = {"config": str(cfg_path), "checkout": cfg["source"]["checkout"], "resolved_path": str(src),
           "expected_head": want, "checks": {}}
    c = res["checks"]

    rc, head = git("rev-parse", "HEAD", cwd=src) if src.is_dir() else (1, "")
    c["S1_checkout_exists"] = {"ok": rc == 0, "detail": "git checkout" if rc == 0 else "missing or not a git checkout"}
    if rc != 0:
        return res
    _, dirty = git("status", "--porcelain", "--untracked-files=no", cwd=src)
    _, kind = git("rev-parse", "--git-dir", cwd=src)
    c["S2_head_and_clean"] = {"ok": head == want and not dirty,
                              "detail": f"HEAD {head}; tracked changes: {'none' if not dirty else dirty.splitlines()[:5]}; "
                                        f"git dir {kind}"}

    rc_c, _ = git("cat-file", "-e", f"{want}^{{commit}}", cwd=ROOT)
    rc_a, _ = git("merge-base", "--is-ancestor", want, "HEAD", cwd=ROOT)
    c["S3_commit_in_this_repository"] = {"ok": rc_c == 0 and rc_a == 0,
                                         "detail": f"present {rc_c == 0}; ancestor of HEAD {rc_a == 0}"}

    _, t_src = git("rev-parse", "HEAD^{tree}", cwd=src)
    _, t_root = git("rev-parse", f"{want}^{{tree}}", cwd=ROOT)
    c["S4_tree_matches"] = {"ok": bool(t_src) and t_src == t_root, "detail": f"checkout {t_src[:12]}, repository {t_root[:12]}"}

    in_src, in_root = vera_source_files(src, "HEAD"), vera_source_files(ROOT, "HEAD")
    diff = sorted(p for p in set(in_src) | set(in_root) if in_src.get(p) != in_root.get(p))
    _, root_dirty = git("status", "--porcelain", "--untracked-files=no", "--", *sorted(in_root), cwd=ROOT)
    c["S5_matches_tracked_root"] = {"ok": not diff and bool(in_root) and not root_dirty,
                                    "detail": f"{len(in_root)} VERA source files; differing {diff[:5]}; "
                                              f"root working copies modified: {root_dirty.splitlines()[:5] if root_dirty else 'none'}",
                                    "generate_initial_score.py_blob": in_root.get("generate_initial_score.py")}

    if import_python:
        code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import generate_initial_score as g; "
                "print(json.dumps({'file': g.__file__, 'classes': [n for n in ('VideoAnomalyDetectionModelInference',) "
                "if hasattr(g, n)]}))")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
        with tempfile.TemporaryDirectory() as td:
            p = subprocess.run([import_python, "-B", "-c", code, str(src)], cwd=td, env=env, capture_output=True, text=True)
        try:
            got = json.loads(p.stdout.strip().splitlines()[-1])
            ok = Path(got["file"]).resolve() == (src / "generate_initial_score.py").resolve() and bool(got["classes"])
            detail = got
        except Exception:
            ok, detail = False, (p.stderr or p.stdout)[-600:]
        c["S6_import_resolves_to_checkout"] = {"ok": ok, "detail": detail}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--import-python", help="interpreter for the S6 import test (omit to skip it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, repo_head = git("rev-parse", "HEAD", cwd=ROOT)
    results = [check_config(p, a.import_python) for p in a.config]
    ok = all(v["ok"] for r in results for v in r["checks"].values()) and all(len(r["checks"]) >= 5 for r in results)
    out = {"repository_root": str(ROOT), "repository_head": repo_head, "passed": ok, "results": results}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for r in results:
        print(r["config"], "->", r["checkout"])
        for k, v in r["checks"].items():
            print(f"  {'PASS' if v['ok'] else 'FAIL'} {k}: {v['detail']}")
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
