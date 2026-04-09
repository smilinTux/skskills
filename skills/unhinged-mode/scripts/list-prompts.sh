#!/usr/bin/env bash
# List available L1B3RT4S liberation prompts
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
L1B3RT4S_DIR="${SKILL_DIR}/lib/l1b3rt4s"

echo "=== Available Liberation Prompts ==="
echo ""

for f in "${L1B3RT4S_DIR}/"*.mkd; do
    name=$(basename "$f" .mkd)
    size=$(wc -c < "$f")
    lines=$(wc -l < "$f")
    echo "  ${name} — ${lines} lines, ${size} bytes"
done

echo ""
echo "Special files:"
[[ -f "${L1B3RT4S_DIR}/#MOTHERLOAD.txt" ]] && echo "  #MOTHERLOAD.txt — Master collection"
[[ -f "${L1B3RT4S_DIR}/!SHORTCUTS.json" ]] && echo "  !SHORTCUTS.json — Quick reference shortcuts"
[[ -f "${L1B3RT4S_DIR}/*SPECIAL_TOKENS.json" ]] && echo "  *SPECIAL_TOKENS.json — Special token exploits"

echo ""
echo "Usage: get-liberation-prompt.sh <provider>"
echo "  e.g.: get-liberation-prompt.sh nvidia"
echo "  e.g.: get-liberation-prompt.sh meta"
