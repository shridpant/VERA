#!/bin/bash
# Build the Training 1 report from an analysis folder: pdflatex + bibtex, then the Overleaf-ready source zip.
# Usage: bash studies/firearm_training/report/build.sh ANALYSIS_DIR [OUT_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS="$(cd "$1" && pwd)"
OUT="${2:-$ANALYSIS/report}"
rm -rf "$OUT"
mkdir -p "$OUT/tables" "$OUT/figures"
cp "$HERE"/{main.tex,references.bib,abstract.tex,results.tex,limitations.tex} "$OUT/"
cp "$ANALYSIS"/tables/*.tex "$OUT/tables/"
cp "$ANALYSIS"/figures/*.pdf "$OUT/figures/"
cp "$ANALYSIS"/{numbers.tex,qualitative.tex,questions_seed.txt,questions_selected.txt,learner_instruct.txt,optimizer_instruct.txt} "$OUT/"
cd "$OUT"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
if grep -q -E "Citation .* undefined|Reference .* undefined" main.log; then grep -E "undefined" main.log; exit 1; fi
if pdftotext main.pdf - 2>/dev/null | grep -q '??'; then echo "an undefined \\val{} number is printed as ??" >&2; exit 1; fi
rm -f main.aux main.bbl main.blg main.out
zip -q -r ../report_source.zip . -x main.pdf build.log main.log
echo "$OUT/main.pdf"
