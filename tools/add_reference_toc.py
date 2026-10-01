#!/usr/bin/env python3
"""Add a "## Contents" list to long reference Markdown files inside skills.

A "skill directory" is any directory that directly contains a ``SKILL.md``
file. Every Markdown file (``.md`` / ``.markdown``) found anywhere under such
a directory (recursively), other than ``SKILL.md`` itself, that has MORE than
100 lines gets a generated contents list inserted right after its H1 title
(or right after YAML frontmatter when there is no H1).

The contents list is built from every ``##`` and ``###`` heading in the
document (document order), skipping the ``#`` title and ``####``+ headings,
and skipping anything inside fenced code blocks. ``###`` entries are nested
two spaces under the preceding ``##`` entry. Anchors follow GitHub's heading
slug rules and are checked against ALL headings in the document (any level)
so duplicate-heading suffixing (``-1``, ``-2``, ...) matches what GitHub
actually generates.

Running the script twice produces no further changes (idempotent): a file
that already carries a correct contents list near the top is left alone,
and a file whose existing contents list is stale is rewritten in place.

Usage:
    add_reference_toc.py apply <root> [<root> ...]
    add_reference_toc.py check <root> [<root> ...]

``apply`` writes changes and prints one line per file touched.
``check`` makes no changes; it verifies every anchor in every inserted
contents list resolves to a real heading in its file, and exits non-zero on
any file that is missing a needed TOC, has a stale TOC, or has a TOC with a
dangling anchor.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VENDORED_DIR_NAMES = {"node_modules", ".git", "dist", "build"}
CONTENTS_TEXT_RE = re.compile(r"^(table of contents|contents|toc)$", re.IGNORECASE)
CONTENTS_HEADING_RE = re.compile(r"^#{1,6}\s*(table of contents|contents|toc)\s*$", re.IGNORECASE)
# Matches either a bullet ("- [text](#anchor)") or an ordered ("1. [text](#anchor)")
# in-page anchor-link list item, so a pre-existing TOC written either way is
# recognized and fully consumed when we need to replace it.
LIST_ITEM_RE = re.compile(r"^(\s*)(?:-|\d+\.)\s+\[(.+?)\]\(#([^)]*)\)\s*$")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FRONTMATTER_DELIM = "---"
MAX_SCAN_LINES_FOR_EXISTING_TOC = 40


class Heading:
    __slots__ = ("level", "text", "line_index")

    def __init__(self, level: int, text: str, line_index: int):
        self.level = level
        self.text = text
        self.line_index = line_index


def find_skill_dirs(root: Path) -> list[Path]:
    """Directories that directly contain a SKILL.md, skipping vendored dirs."""
    out = []
    for skill_md in root.rglob("SKILL.md"):
        if any(part in VENDORED_DIR_NAMES for part in skill_md.parts):
            continue
        out.append(skill_md.parent)
    return out


def find_target_files(roots: list[Path]) -> list[Path]:
    """Every .md/.markdown file (over 100 lines) under any skill dir, minus SKILL.md."""
    files: set[Path] = set()
    for root in roots:
        root = root.resolve()
        for skill_dir in find_skill_dirs(root):
            for pattern in ("*.md", "*.markdown"):
                for md in skill_dir.rglob(pattern):
                    if md.name == "SKILL.md":
                        continue
                    if any(part in VENDORED_DIR_NAMES for part in md.parts):
                        continue
                    files.add(md)
    out = []
    for f in sorted(files):
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if text.count("\n") + (1 if text and not text.endswith("\n") else 0) > 100:
            out.append(f)
    return out


def strip_inline_markdown(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code
    text = re.sub(r"(\*\*\*|___)(.+?)\1", r"\2", text)  # bold italic
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)  # bold
    text = re.sub(r"(?<!\w)(\*|_)(.+?)\1(?!\w)", r"\2", text)  # italic
    text = re.sub(r"<[^>]+>", "", text)  # bare html tags
    return text.strip()


def slugify(text: str) -> str:
    text = text.strip().lower()
    out = [ch for ch in text if ch.isalnum() or ch in (" ", "-")]
    slug = "".join(out)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


def iter_non_fenced_lines(lines: list[str]):
    """Yield (index, line) for lines outside fenced code blocks."""
    fence_char = None
    fence_len = 0
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line)
        if m:
            char = m.group(2)[0]
            length = len(m.group(2))
            if fence_char is None:
                fence_char = char
                fence_len = length
                continue
            if char == fence_char and length >= fence_len:
                fence_char = None
                fence_len = 0
                continue
        if fence_char is None:
            yield i, line


def parse_headings(lines: list[str]) -> list[Heading]:
    headings = []
    for i, line in iter_non_fenced_lines(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        text = strip_inline_markdown(m.group(2))
        headings.append(Heading(level, text, i))
    return headings


def assign_slugs(headings: list[Heading]) -> list[str]:
    """GitHub-style slug assignment across ALL headings, in document order."""
    seen: dict[str, int] = {}
    slugs = []
    for h in headings:
        base = slugify(h.text)
        count = seen.get(base, 0)
        slug = base if count == 0 else f"{base}-{count}"
        seen[base] = count + 1
        slugs.append(slug)
    return slugs


def build_contents_entries(lines: list[str]) -> list[tuple[bool, str, str]] | None:
    """Return (nested, text, anchor) tuples for the contents list, in document
    order, or None if there are no ## / ### headings to list. ``nested`` is
    True for a ### entry (indented under its preceding ##)."""
    headings = parse_headings(lines)
    slugs = assign_slugs(headings)  # all headings, so duplicate numbering matches GitHub
    entries = []
    for h, slug in zip(headings, slugs):
        if CONTENTS_TEXT_RE.match(h.text):
            continue  # never list the Contents heading itself
        if h.level == 2:
            entries.append((False, h.text, slug))
        elif h.level == 3:
            entries.append((True, h.text, slug))
    return entries or None


def render_entries(entries: list[tuple[bool, str, str]]) -> list[str]:
    lines = []
    for nested, text, anchor in entries:
        prefix = "  - " if nested else "- "
        lines.append(f"{prefix}[{text}](#{anchor})")
    return lines


def normalize_entry(line: str) -> tuple[bool, str, str] | None:
    """Parse any existing list-link line (bullet or ordered) into the same
    (nested, text, anchor) shape ``build_contents_entries`` produces, so a
    hand-written TOC in a different style can still be compared for a match."""
    m = LIST_ITEM_RE.match(line)
    if not m:
        return None
    indent, text, anchor = m.groups()
    return (len(indent) > 0, text, anchor)


def find_frontmatter_end(lines: list[str]) -> int:
    """Return the index of the first line after frontmatter, or 0 if none."""
    if not lines or lines[0].strip() != FRONTMATTER_DELIM:
        return 0
    for i in range(1, len(lines)):
        if lines[i].strip() == FRONTMATTER_DELIM:
            return i + 1
    return 0


def find_h1(lines: list[str], start: int) -> int | None:
    for i, line in iter_non_fenced_lines(lines):
        if i < start:
            continue
        if not line.strip():
            continue
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            return i
        return None  # first non-blank content line isn't an H1
    return None


def find_existing_toc_span(lines: list[str]):
    """Look for an existing contents block in the first 40 lines of the file.

    Returns (start, end, had_heading) where lines[start:end] is the full
    span to replace (heading line, if any, plus the list items), or None if
    no existing contents block was found.
    """
    scan_end = min(len(lines), MAX_SCAN_LINES_FOR_EXISTING_TOC)

    # Case 1: an explicit Contents / Table of Contents / TOC heading.
    for i in range(0, scan_end):
        if CONTENTS_HEADING_RE.match(lines[i].strip()):
            j = i + 1
            if j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines) and LIST_ITEM_RE.match(lines[j]):
                j += 1
            return (i, j, True)

    # Case 2: a bare bullet list of in-page anchor links, no heading. Only
    # the start of a contiguous run counts, so we don't re-match the tail
    # of a block we'd already have matched from its first line.
    i = 0
    while i < scan_end:
        if LIST_ITEM_RE.match(lines[i]) and (i == 0 or not LIST_ITEM_RE.match(lines[i - 1])):
            j = i
            while j < len(lines) and LIST_ITEM_RE.match(lines[j]):
                j += 1
            return (i, j, False)
        i += 1

    return None


def process_file(path: Path, apply: bool) -> tuple[bool, str]:
    """Returns (changed, reason)."""
    original = path.read_text(encoding="utf-8")
    had_trailing_newline = original.endswith("\n")
    lines = original.split("\n")
    if had_trailing_newline:
        lines = lines[:-1]

    entries = build_contents_entries(lines)
    if entries is None:
        return False, "no ## / ### headings"
    body = render_entries(entries)

    existing = find_existing_toc_span(lines)

    if existing is not None:
        start, end, had_heading = existing
        current_span = lines[start:end]
        current_entries = [normalize_entry(ln) for ln in current_span if normalize_entry(ln) is not None]
        if current_entries == entries:
            return False, "existing contents list already matches"
        if apply:
            if had_heading:
                replacement = [current_span[0], ""] + body
            else:
                replacement = list(body)
            new_lines = lines[:start] + replacement + lines[end:]
            _write(path, new_lines, had_trailing_newline)
        return True, "replaced stale contents list"

    fm_end = find_frontmatter_end(lines)
    h1_idx = find_h1(lines, fm_end)
    insert_at = (h1_idx + 1) if h1_idx is not None else fm_end

    if apply:
        before = lines[:insert_at]
        after = lines[insert_at:]
        new_block = ["## Contents", ""] + body
        lead_sep = [] if not before else [""]
        trail_sep = [] if (after and after[0] == "") else [""]
        new_lines = before + lead_sep + new_block + trail_sep + after
        _write(path, new_lines, had_trailing_newline)
    return True, "inserted new contents list"


def _write(path: Path, lines: list[str], had_trailing_newline: bool) -> None:
    text = "\n".join(lines)
    if had_trailing_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def check_file(path: Path) -> list[str]:
    """Return a list of problems with this file's contents list (empty = ok)."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    entries = build_contents_entries(lines)
    if entries is None:
        return []

    existing = find_existing_toc_span(lines)

    problems = []
    if existing is None:
        return [f"{path}: missing contents list"]

    start, end, had_heading = existing
    current_span = lines[start:end]
    current_entries = [normalize_entry(ln) for ln in current_span if normalize_entry(ln) is not None]
    if current_entries != entries:
        problems.append(f"{path}: contents list is stale")

    headings = parse_headings(lines)
    valid_slugs = set(assign_slugs(headings))
    for m in (LIST_ITEM_RE.match(ln) for ln in current_span):
        if not m:
            continue
        anchor = m.group(3)
        if anchor not in valid_slugs:
            problems.append(f"{path}: dangling anchor #{anchor}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["apply", "check"])
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args(argv)

    files = find_target_files(args.roots)

    if args.mode == "check":
        problems = []
        for f in files:
            problems.extend(check_file(f))
        if problems:
            for p in problems:
                print(p)
            print(f"FAIL: {len(problems)} problem(s) across {len(files)} candidate file(s)")
            return 1
        print(f"OK: {len(files)} candidate file(s), all contents lists valid")
        return 0

    changed = 0
    for f in files:
        did_change, reason = process_file(f, apply=True)
        status = "changed" if did_change else "skipped"
        print(f"{status}: {f} ({reason})")
        if did_change:
            changed += 1
    print(f"Done: {changed} file(s) changed, {len(files) - changed} unchanged, {len(files)} candidate(s) total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
