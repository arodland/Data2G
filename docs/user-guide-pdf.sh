#!/bin/sh
# docs/user-guide.md as a PDF: docs/user-guide-pdf.sh [out.pdf]
# Needs pandoc (markdown to HTML, styled by user-guide.css) and Chrome or
# Chromium (HTML to PDF). Override either with PANDOC= or CHROME=; CHROME_FLAGS
# adds Chrome flags (CI: --no-sandbox). The release workflow attaches the result
# to each release as data2g-user-guide.pdf, which README links. No pandoc:
#   PANDOC="$(uvx --from pypandoc_binary python -c 'import pypandoc; print(pypandoc.get_pandoc_path())')" docs/user-guide-pdf.sh
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
out="$(realpath "${1:-data2g-user-guide.pdf}")"
chrome="${CHROME:-$(command -v chromium google-chrome-stable google-chrome chrome | head -1)}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
"${PANDOC:-pandoc}" "$here/user-guide.md" -f gfm -t html5 --standalone --embed-resources \
    --shift-heading-level-by=-1 --resource-path="$here" --css "$here/user-guide.css" -o "$tmp/guide.html"
"$chrome" --headless --disable-gpu ${CHROME_FLAGS:-} --no-pdf-header-footer --user-data-dir="$tmp/profile" \
    --print-to-pdf="$out" "file://$tmp/guide.html" 2>/dev/null
[ -s "$out" ] || { echo "user-guide-pdf: Chrome wrote no PDF" >&2; exit 1; }  # Chrome exits 0 either way
echo "wrote $out"
