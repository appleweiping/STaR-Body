#!/usr/bin/env bash
# Build the CVPR 2027 submission (main.pdf) and the supplement (supp.pdf).
# The supplement reads cross-references from main.aux, so main is built first.
# Usage: bash build.sh            build both
#        bash build.sh clean      remove LaTeX build products (keeps PDFs)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
  latexmk -c main.tex supp.tex
  rm -f ./*.bbl ./*.brf
  exit 0
fi

LATEXMK=(latexmk -pdf -interaction=nonstopmode -halt-on-error)
"${LATEXMK[@]}" main.tex
"${LATEXMK[@]}" supp.tex

for doc in main supp; do
  if grep -q "There were undefined references\|Citation .* undefined\|Reference .* undefined" "$doc.log"; then
    echo "WARNING: undefined references or citations in $doc.log" >&2
  fi
done

python3 - <<'PY' || true
import re
try:
    import pypdf
except ImportError:
    raise SystemExit('pypdf not installed: page check skipped')
for doc in ('main', 'supp'):
    reader = pypdf.PdfReader(f'{doc}.pdf')
    pages = len(reader.pages)
    note = ''
    if doc == 'main':
        last_body = None
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ''
            if 'LLM Usage Statement' in text or re.search(r'^References\s*$', text, re.M):
                last_body = i
                break
        note = f'; LLM statement/references start on page {last_body} (main text must end by page 8)'
    print(f'{doc}.pdf: {pages} pages{note}')
PY
