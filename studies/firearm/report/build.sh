#!/bin/bash
# Build the report from an analysis folder: copy its tables, figures and numbers next to main.tex,
# compile with pdflatex + bibtex, and zip the Overleaf-ready source.
# Usage: bash studies/firearm/report/build.sh ANALYSIS_DIR [OUT_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS="$(cd "$1" && pwd)"
OUT="${2:-$ANALYSIS/report}"
rm -rf "$OUT"
mkdir -p "$OUT/tables" "$OUT/figures"
cp "$HERE/main.tex" "$HERE/references.bib" "$HERE/abstract.tex" "$HERE/results.tex" "$HERE/limitations.tex" "$HERE/refinement_text.tex" "$OUT/"
cp "$ANALYSIS"/tables/*.tex "$OUT/tables/"
cp "$ANALYSIS"/figures/*.pdf "$OUT/figures/"
cp "$ANALYSIS/numbers.tex" "$ANALYSIS/qualitative_cases.tex" "$OUT/"
cp "$HERE/../PROMPT_LOG.md" "$OUT/" 2>/dev/null || true
for f in A G AG; do cp "$ANALYSIS/prompts/$f.txt" "$OUT/prompt_$f.txt"; done
cd "$OUT"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log
if grep -q -E "Citation .* undefined|Reference .* undefined" main.log; then
  grep -E "undefined" main.log; echo "unresolved references" >&2; exit 1
fi
if grep -q '??' <(pdftotext main.pdf - 2>/dev/null || true); then echo "an undefined \\val{} number is printed as ??" >&2; exit 1; fi
rm -f main.aux main.bbl main.blg main.out main.toc
zip -q -r ../report_source.zip . -x 'main.pdf' -x 'build.log' -x 'main.log'
echo "$OUT/main.pdf"
