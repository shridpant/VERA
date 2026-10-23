#!/usr/bin/env bash
# Clean-clone test: clone the proposed branch fresh and run what works from tracked files alone (no live data).
# The only external input is the Python interpreter (the conda env, rebuilt from setup/environment/).
#   SRC_REPO  repository to clone from;  BRANCH  branch to test;  C  clone directory (new);  O  output directory (new)
set -uo pipefail
PY=$HOME/vera_research/setup/environment/env/bin/python
SRC_REPO=${SRC_REPO:?}; BRANCH=${BRANCH:?}; C=${C:?}; O=${O:?}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= VERA_PROJECT_ROOT=$C
mkdir -p $O/logs; S=$O/SUMMARY.txt; : > $S
res(){ printf '%-5s %s\n' "$1" "$2" | tee -a $S; }
run(){ local name=$1; shift; ( "$@" ) > $O/logs/$name.log 2>&1; local rc=$?; [ $rc -eq 0 ] && res PASS "$name (exit 0)" || res FAIL "$name (exit $rc)"; }
git clone -q --branch $BRANCH $SRC_REPO $C && cd $C || { res FAIL clone; exit 1; }
echo "clone of $BRANCH at $(git rev-parse HEAD) in $C" | tee -a $S
n=$(git ls-files | wc -l); miss=$(git ls-files -z | xargs -0 -I{} sh -c '[ -e "{}" ] || echo "{}"' | wc -l)
[ "$miss" -eq 0 ] && res PASS "all $n tracked files present" || res FAIL "$miss tracked files missing"
[ -z "$(git status --porcelain --untracked-files=all)" ] && res PASS "fresh clone is clean" || res FAIL "fresh clone not clean"
run compile_all_python $PY -B -c "import subprocess; fs=subprocess.run(['git','ls-files','*.py'],capture_output=True,text=True).stdout.split(); [compile(open(f,encoding='utf-8').read(), f, 'exec') for f in fs]; print(len(fs), 'files compile')"
run recreate_historical_checkout git worktree add --detach setup/source/VERA 008ed4019781c39e7cb16244b7500244f3986f92
run test_vera_pipeline $PY -B setup/scripts/test_vera_pipeline.py
run pin_tracked_source bash setup/scripts/pin_tracked_source.sh
pin=$(sed -n 's/^  checkout: //p' $O/logs/pin_tracked_source.log); hd=$(sed -n 's/^  expected_head: //p' $O/logs/pin_tracked_source.log)
python3 -I -B -c "import sys,re; s=open(sys.argv[1]).read(); s=re.sub(r'(?m)^  checkout: .*$', '  checkout: '+sys.argv[2], s); s=re.sub(r'(?m)^  expected_head: .*$', '  expected_head: '+sys.argv[3], s); s=re.sub(r'(?m)^  branch: .*\n', '', s); open(sys.argv[4],'w').write(s)" experiments/experiment01/config.yaml "$pin" "$hd" $O/future_run_config.yaml
run source_resolution $PY -B setup/scripts/check_source_resolution.py --config experiments/experiment01/config.yaml --config experiments/experiment02/config.yaml --config $O/future_run_config.yaml --import-python $PY --out $O/source_resolution.json
run doc_links python3 -I -B setup/scripts/check_doc_links.py --out $O/doc_links.json
run precision_crossings python3 -I -B experiments/experiment02/errata/precision_crossings.py --effects experiments/experiment02/runs/run_002/results/temporal_effects.csv --out $O/precision_crossings.json
cmp -s $O/precision_crossings.json experiments/experiment02/errata/precision_crossings.json && res PASS "precision_crossings.json reproduces the tracked copy" || res FAIL "precision_crossings.json differs"
# Expected to be limited: these need local-only data (frames, videos, features, reference scores).
run test_attr_lib $PY -B experiments/experiment02/code/test_attr_lib.py
cp -a experiments/experiment01/runs/run_001 $O/exp01_run_001; cp -a experiments/experiment02/runs/run_002 $O/exp02_run_002_v; cp -a experiments/experiment02/runs/run_002 $O/exp02_run_002_a
run validate_exp01_run_001 $PY -B setup/scripts/validate_run.py --run-dir $O/exp01_run_001 --require-report
run validate_exp02_run_002 $PY -B experiments/experiment02/code/validate_attr.py --run-dir $O/exp02_run_002_v
run analyze_exp02_run_002 $PY -B experiments/experiment02/code/analyze_attr.py --run-dir $O/exp02_run_002_a
ch=$(cd $O/exp02_run_002_a && find . -type f ! -path './logs/*' -print0 | sort -z | xargs -0 sha256sum | while read -r h p; do [ "$h" = "$(sha256sum < $C/experiments/experiment02/runs/run_002/$p | cut -c1-64)" ] || echo "${p#./}"; done)
[ -z "$ch" ] && res PASS "analyzer outputs from tracked records are byte-identical to the tracked results" || res INFO "analyzer outputs differing: $ch"
for v in validate_exp01_run_001 validate_exp02_run_002; do echo "  $v per-check:" | tee -a $S; grep -E '^(PASS|FAIL) +[CV][0-9]+' $O/logs/$v.log | sed -E 's/^(PASS|FAIL) +([CV][0-9]+).*/\1 \2/' | sort -k2 | awk '{a[$1]=a[$1]" "$2} END {for (k in a) print "    "k":"a[k]}' | tee -a $S; done
echo "done: $(grep -c '^PASS' $S) PASS, $(grep -c '^FAIL' $S) FAIL, $(grep -c '^INFO' $S) INFO" | tee -a $S
