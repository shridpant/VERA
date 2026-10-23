#!/usr/bin/env bash
# Repository-integration validation for the baseline (CPU only; no model inference; no writes to the live tree).
#   R  repository working tree to test (bootstrap, clean clone, or the live tree)
#   O  new output directory for logs, copies and JSON
# Historical data (frames, videos, features, experiment01 records) are read from the live tree via VERA_PROJECT_ROOT
# only where a step says so.
set -uo pipefail
L=$HOME/vera_research; R=${R:?}; O=${O:?}; PY=$L/setup/environment/env/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
mkdir -p $O/logs; S=$O/SUMMARY.txt; : > $S
res(){ printf '%-5s %s\n' "$1" "$2" | tee -a $S; }
run(){ local name=$1; shift; ( "$@" ) > $O/logs/$name.log 2>&1; local rc=$?; [ $rc -eq 0 ] && res PASS "$name (exit 0)" || res FAIL "$name (exit $rc)"; return 0; }
cd $R
echo "R=$R HEAD=$(git rev-parse HEAD) O=$O" | tee -a $S

# 1. Ancestry and source equivalence
base=15b8bcb8574a977c229c577f50bfe6f06d07106e; src=008ed4019781c39e7cb16244b7500244f3986f92
git merge-base --is-ancestor $base HEAD && git merge-base --is-ancestor $src HEAD && [ "$(git rev-parse $src^)" = $base ] \
  && res PASS "ancestry: $base <- $src <- ... <- HEAD" || res FAIL "ancestry"
fork=$(GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/shridpant/VERA.git refs/heads/main | cut -f1)
[ "$fork" = $base ] && res PASS "fork main still $base" || res FAIL "fork main is $fork"
paths=$(git ls-tree -r --name-only $src); nd=0
for p in $paths; do [ "$(git rev-parse $src:$p)" = "$(git rev-parse HEAD:$p 2>/dev/null)" ] || { [ "$p" = README.md ] || { nd=$((nd+1)); echo "  differs: $p"; }; }; done
[ $nd -eq 0 ] && res PASS "source equivalence: all $(echo "$paths" | wc -w) files of $src identical at HEAD (README.md integrated)" || res FAIL "source equivalence: $nd differ"
[ "$(git -C $L/setup/source/VERA rev-parse 'HEAD^{tree}')" = "$(git rev-parse "$src^{tree}")" ] && [ -z "$(git -C $L/setup/source/VERA status --porcelain --untracked-files=all)" ] \
  && res PASS "live setup/source/VERA = tree of $src, clean" || res FAIL "live nested checkout differs or dirty"

# 2. Tree hygiene
[ "$(git ls-tree -r HEAD | awk '$1=="160000"' | wc -l)" -eq 0 ] && res PASS "no gitlinks" || res FAIL "gitlinks present"
big=$(git ls-tree -r -l HEAD | awk '$4>2097152 && $5!="Data/name_gt_pair.json"{print $5}')
[ -z "$big" ] && res PASS "no new file > 2 MB (inherited Data/name_gt_pair.json 11.7 MB excluded)" || res FAIL "large: $big"
hits=$(git grep -l -I -i -E '(passw(or)?d *[:=]|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20}|github_pat_|hf_[A-Za-z0-9]{30}|AKIA[0-9A-Z]{16}|HF_TOKEN=|GITHUB_TOKEN=|x-amz-signature|[?&](sig|signature|token|expires)=)' HEAD -- . ':(exclude)setup/git_baseline/*.sh' | head)
[ -z "$hits" ] && res PASS "secret/signed-URL scan: no hits" || res FAIL "secret scan hits: $hits"
cache=$(git ls-tree -r --name-only HEAD | grep -E '(^|/)(__pycache__|env|hf_home|\.cache)/|\.(safetensors|mp4|zip|jpg|pyc)$' | head)
[ -z "$cache" ] && res PASS "no caches, weights, videos, frames or archives tracked" || res FAIL "tracked: $cache"

# 3. CPU tests (code from R; historical source and data from the live tree)
run test_vera_pipeline env VERA_PROJECT_ROOT=$L $PY -B $R/setup/scripts/test_vera_pipeline.py
run test_attr_lib env VERA_PROJECT_ROOT=$L $PY -B $R/experiments/experiment02/code/test_attr_lib.py
run compile_all_python $PY -B -c "import subprocess; fs=subprocess.run(['git','ls-files','*.py'],capture_output=True,text=True).stdout.split(); [compile(open(f,encoding='utf-8').read(), f, 'exec') for f in fs]; print(len(fs), 'tracked .py files compile')"

# 4. Validators and analyzers on COPIES of existing outputs (no inference; originals untouched)
for r in run_001 run_002; do cp -a $L/experiments/experiment01/runs/$r $O/exp01_$r; done
run validate_exp01_run_001 env VERA_PROJECT_ROOT=$L $PY -B $R/setup/scripts/validate_run.py --run-dir $O/exp01_run_001 --require-report
run validate_exp01_run_002 env VERA_PROJECT_ROOT=$L $PY -B $R/setup/scripts/validate_run.py --run-dir $O/exp01_run_002
run summarize_exp01_run_001 env VERA_PROJECT_ROOT=$L $PY -B $R/setup/scripts/summarize_run.py --run-dir $O/exp01_run_001
cp -a $L/experiments/experiment02/runs/run_002 $O/exp02_run_002_validate; cp -a $L/experiments/experiment02/runs/run_002 $O/exp02_run_002_analyze
run validate_exp02_run_002 env VERA_PROJECT_ROOT=$L $PY -B $R/experiments/experiment02/code/validate_attr.py --run-dir $O/exp02_run_002_validate
run analyze_exp02_run_002 env VERA_PROJECT_ROOT=$L $PY -B $R/experiments/experiment02/code/analyze_attr.py --run-dir $O/exp02_run_002_analyze
run spatial_border_check env VERA_PROJECT_ROOT=$L $PY -B $R/experiments/experiment02/code/spatial_border_check.py $O/exp02_run_002_analyze $L
for d in exp01_run_001:experiments/experiment01/runs/run_001 exp01_run_002:experiments/experiment01/runs/run_002 \
         exp02_run_002_validate:experiments/experiment02/runs/run_002 exp02_run_002_analyze:experiments/experiment02/runs/run_002; do
  c=${d%%:*}; o=${d#*:}
  ch=$(cd $O/$c && find . -type f ! -path './logs/*' -print0 | sort -z | xargs -0 sha256sum | while read -r h p; do [ "$h" = "$(sha256sum < $L/$o/$p | cut -c1-64)" ] || echo "${p#./}"; done)
  [ -z "$ch" ] && res PASS "rerun outputs byte-identical to originals: $c" || res INFO "rerun outputs that differ from originals in $c: $(echo $ch | tr '\n' ' ')"
done

# 5. Erratum measurement and source resolution (code from R)
run precision_crossings python3 -I -B $R/experiments/experiment02/errata/precision_crossings.py --effects $L/experiments/experiment02/runs/run_002/results/temporal_effects.csv --out $O/precision_crossings.json
[ -e $R/setup/source/VERA ] || git -C $R worktree add --detach $R/setup/source/VERA $src > $O/logs/worktree_historical.log 2>&1
run pin_tracked_source bash $R/setup/scripts/pin_tracked_source.sh
pin=$(sed -n 's/^  checkout: //p' $O/logs/pin_tracked_source.log); head_=$(sed -n 's/^  expected_head: //p' $O/logs/pin_tracked_source.log)
python3 -I -B -c "import sys,re; s=open(sys.argv[1]).read(); s=re.sub(r'(?m)^  checkout: .*$', '  checkout: '+sys.argv[2], s); s=re.sub(r'(?m)^  expected_head: .*$', '  expected_head: '+sys.argv[3], s); s=re.sub(r'(?m)^  branch: .*\n', '', s); open(sys.argv[4],'w').write(s)" \
  $R/experiments/experiment01/config.yaml "$pin" "$head_" $O/future_run_config.yaml
run source_resolution env VERA_PROJECT_ROOT=$R $PY -B $R/setup/scripts/check_source_resolution.py --config experiments/experiment01/config.yaml --config experiments/experiment02/config.yaml --config $O/future_run_config.yaml --import-python $PY --out $O/source_resolution.json
run doc_links env VERA_PROJECT_ROOT=$R python3 -I -B $R/setup/scripts/check_doc_links.py --out $O/doc_links.json
echo "done: $(grep -c '^PASS' $S) PASS, $(grep -c '^FAIL' $S) FAIL, $(grep -c '^INFO' $S) INFO" | tee -a $S
