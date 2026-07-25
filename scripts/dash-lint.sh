#!/usr/bin/env bash
# Fail if any em dash (U+2014) or en dash (U+2013) appears anywhere under
# skills/. Sk-standard bans typographic dashes as sentence punctuation;
# skskills-authored and vendored-and-normalized skill content must be
# dash-clean. See docs/superpowers/specs/2026-07-25-autocode-skskills-quality-layer.md
# (D1, vendoring) and CLAUDE.md's "No em/en dashes" rule.
#
# EXCLUSIONS are deliberate, documented pre-existing exceptions, not a
# blanket opt-out:
#   skills/unhinged-mode/lib/  - a full third-party vendored research repo
#     (obliteratus + l1b3rt4s) predating this lint, containing Python,
#     LaTeX, JSON, and notebook content where mechanical dash substitution
#     risks corrupting code/math/JSON. Normalizing it is a separate,
#     unscoped cleanup, not part of the skills-authoring surface this lint
#     protects. Revisit if/when that vendor tree is re-imported.
#
# Usage: scripts/dash-lint.sh [root]   (default root: skills/)

set -euo pipefail

ROOT="${1:-skills}"

if [ ! -d "$ROOT" ]; then
  echo "dash-lint: root '$ROOT' not found" >&2
  exit 2
fi

EXCLUDE_PREFIX="${ROOT%/}/unhinged-mode/lib/"

hits=0
while IFS= read -r -d '' f; do
  case "$f" in
    "$EXCLUDE_PREFIX"*) continue ;;
  esac
  if grep -qP '[\x{2013}\x{2014}]' "$f" 2>/dev/null; then
    echo "dash-lint: em/en dash found in $f"
    grep -noP '.{0,20}[\x{2013}\x{2014}].{0,20}' "$f" | sed "s|^|  |"
    hits=$((hits + 1))
  fi
done < <(find "$ROOT" -type f -print0)

if [ "$hits" -gt 0 ]; then
  echo ""
  echo "dash-lint: FAILED, $hits file(s) contain em/en dashes outside the documented exclusion ($EXCLUDE_PREFIX)."
  exit 1
fi

echo "dash-lint: OK, no em/en dashes under $ROOT (excluding $EXCLUDE_PREFIX)."
