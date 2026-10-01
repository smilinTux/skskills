"""Tests for tools/add_reference_toc.py."""
import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "add_reference_toc.py"
_spec = importlib.util.spec_from_file_location("add_reference_toc", MODULE_PATH)
toc = importlib.util.module_from_spec(_spec)
sys.modules["add_reference_toc"] = toc
_spec.loader.exec_module(toc)


def make_long_doc(heading_block: str, filler_lines: int = 110) -> str:
    """A doc long enough (>100 lines) with the given heading/body block."""
    filler = "\n".join(f"Body line {i}." for i in range(filler_lines))
    return heading_block.rstrip("\n") + "\n\n" + filler + "\n"


def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def make_skill(tmp_path: Path, skill_name: str = "demo") -> Path:
    skill_dir = tmp_path / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo Skill\n\nA short skill file.\n", encoding="utf-8")
    return skill_dir


# --- adds TOC ----------------------------------------------------------

def test_adds_toc_to_long_reference_file(tmp_path):
    skill_dir = make_skill(tmp_path)
    ref_dir = skill_dir / "references"
    ref_dir.mkdir()
    doc = make_long_doc(
        "# Reference Guide\n\n"
        "Intro paragraph.\n\n"
        "## Setup\n\nSetup text.\n\n"
        "### Install\n\nInstall text.\n\n"
        "## Usage\n\nUsage text.\n"
    )
    f = ref_dir / "guide.md"
    f.write_text(doc, encoding="utf-8")

    changed, reason = toc.process_file(f, apply=True)
    assert changed is True

    out = f.read_text(encoding="utf-8")
    lines = out.split("\n")
    assert lines[0] == "# Reference Guide"
    assert lines[1] == ""
    assert lines[2] == "## Contents"
    assert lines[3] == ""
    assert "- [Setup](#setup)" in lines
    assert "  - [Install](#install)" in lines
    assert "- [Usage](#usage)" in lines
    # original intro content is preserved further down
    assert "Intro paragraph." in out


def test_short_file_is_left_untouched(tmp_path):
    skill_dir = make_skill(tmp_path)
    f = skill_dir / "short.md"
    f.write_text("# Short\n\n## A\n\nnot long enough.\n", encoding="utf-8")
    original = f.read_text(encoding="utf-8")

    files = toc.find_target_files([tmp_path])
    assert f not in files
    assert f.read_text(encoding="utf-8") == original


def test_skill_md_itself_is_never_touched(tmp_path):
    skill_dir = make_skill(tmp_path)
    big_skill_md = "\n".join(f"## Section {i}\n\nbody\n" for i in range(60))
    (skill_dir / "SKILL.md").write_text("# Demo\n\n" + big_skill_md, encoding="utf-8")

    files = toc.find_target_files([tmp_path])
    assert all(p.name != "SKILL.md" for p in files)


# --- idempotent ----------------------------------------------------------

def test_running_twice_is_idempotent(tmp_path):
    skill_dir = make_skill(tmp_path)
    ref_dir = skill_dir / "references"
    ref_dir.mkdir()
    doc = make_long_doc(
        "# Reference\n\n## One\n\ntext\n\n## Two\n\n### Sub Two\n\ntext\n"
    )
    f = ref_dir / "doc.md"
    f.write_text(doc, encoding="utf-8")

    changed1, _ = toc.process_file(f, apply=True)
    assert changed1 is True
    after_first = f.read_text(encoding="utf-8")

    changed2, reason2 = toc.process_file(f, apply=True)
    assert changed2 is False
    assert "already matches" in reason2
    assert f.read_text(encoding="utf-8") == after_first


def test_check_mode_passes_after_apply(tmp_path):
    skill_dir = make_skill(tmp_path)
    doc = make_long_doc("# Ref\n\n## A\n\ntext\n\n## B\n\ntext\n")
    f = skill_dir / "doc.md"
    f.write_text(doc, encoding="utf-8")

    toc.process_file(f, apply=True)
    problems = toc.check_file(f)
    assert problems == []


# --- skips fenced code block headings -------------------------------------

def test_skips_headings_inside_code_fences(tmp_path):
    skill_dir = make_skill(tmp_path)
    doc = make_long_doc(
        "# Ref\n\n"
        "## Real Heading\n\n"
        "```markdown\n"
        "## Not A Real Heading\n"
        "```\n\n"
        "## Another Real Heading\n\ntext\n"
    )
    f = skill_dir / "doc.md"
    f.write_text(doc, encoding="utf-8")

    toc.process_file(f, apply=True)
    out = f.read_text(encoding="utf-8")
    assert "- [Real Heading](#real-heading)" in out
    assert "- [Another Real Heading](#another-real-heading)" in out
    assert "Not A Real Heading" not in out.split("## Contents")[1].split("```")[0]


# --- slug duplicates -------------------------------------------------------

def test_duplicate_headings_get_numbered_slugs(tmp_path):
    skill_dir = make_skill(tmp_path)
    doc = make_long_doc(
        "# Ref\n\n## Setup\n\ntext\n\n## Setup\n\ntext\n\n## Setup\n\ntext\n"
    )
    f = skill_dir / "doc.md"
    f.write_text(doc, encoding="utf-8")

    toc.process_file(f, apply=True)
    out = f.read_text(encoding="utf-8")
    assert "- [Setup](#setup)" in out
    assert "- [Setup](#setup-1)" in out
    assert "- [Setup](#setup-2)" in out


# --- existing TOC skip / replace -------------------------------------------

def test_existing_matching_contents_heading_is_skipped(tmp_path):
    skill_dir = make_skill(tmp_path)
    doc = make_long_doc(
        "# Ref\n\n"
        "## Contents\n\n"
        "- [One](#one)\n"
        "- [Two](#two)\n\n"
        "## One\n\ntext\n\n## Two\n\ntext\n"
    )
    f = skill_dir / "doc.md"
    f.write_text(doc, encoding="utf-8")
    before = f.read_text(encoding="utf-8")

    changed, reason = toc.process_file(f, apply=True)
    assert changed is False
    assert f.read_text(encoding="utf-8") == before


def test_existing_stale_contents_list_is_replaced_in_place(tmp_path):
    skill_dir = make_skill(tmp_path)
    doc = make_long_doc(
        "# Ref\n\n"
        "## Table of Contents\n\n"
        "- [Old](#old)\n\n"
        "## One\n\ntext\n\n## Two\n\ntext\n"
    )
    f = skill_dir / "doc.md"
    f.write_text(doc, encoding="utf-8")

    changed, reason = toc.process_file(f, apply=True)
    assert changed is True
    out = f.read_text(encoding="utf-8")
    # heading wording is preserved, only the list is corrected
    assert "## Table of Contents" in out
    assert "- [Old](#old)" not in out
    assert "- [One](#one)" in out
    assert "- [Two](#two)" in out

    # second pass is now a no-op
    changed2, _ = toc.process_file(f, apply=True)
    assert changed2 is False


def test_existing_numbered_toc_is_fully_replaced_not_duplicated(tmp_path):
    """A hand-written numbered TOC ("1. [x](#x)") must be entirely swallowed
    on replace, not left behind alongside the new bullet list."""
    skill_dir = make_skill(tmp_path)
    doc = make_long_doc(
        "# Ref\n\n"
        "## Table of Contents\n\n"
        "1. [One](#one)\n"
        "2. [Two](#two)\n\n"
        "## One\n\ntext\n\n## Two\n\n### Sub\n\ntext\n"
    )
    f = skill_dir / "doc.md"
    f.write_text(doc, encoding="utf-8")

    changed, _ = toc.process_file(f, apply=True)
    assert changed is True
    out = f.read_text(encoding="utf-8")
    assert out.count("[One](#one)") == 1
    assert out.count("[Two](#two)") == 1
    assert "1. [One](#one)" not in out
    assert "- [One](#one)" in out
    assert "  - [Sub](#sub)" in out

    changed2, _ = toc.process_file(f, apply=True)
    assert changed2 is False


def test_check_mode_reports_stale_toc(tmp_path):
    skill_dir = make_skill(tmp_path)
    doc = make_long_doc(
        "# Ref\n\n## Contents\n\n- [Wrong](#wrong)\n\n## Real\n\ntext\n"
    )
    f = skill_dir / "doc.md"
    f.write_text(doc, encoding="utf-8")

    problems = toc.check_file(f)
    assert any("stale" in p for p in problems)


# --- discovery --------------------------------------------------------

def test_find_target_files_only_inside_skill_dirs(tmp_path):
    make_skill(tmp_path, "demo")
    outside = tmp_path / "not_a_skill"
    outside.mkdir()
    long_text = "# Title\n\n" + "\n".join(f"## H{i}\n\nbody\n" for i in range(60))
    (outside / "orphan.md").write_text(long_text, encoding="utf-8")

    files = toc.find_target_files([tmp_path])
    assert all("not_a_skill" not in str(p) for p in files)


def test_vendored_dirs_are_skipped(tmp_path):
    skill_dir = make_skill(tmp_path)
    vendored = skill_dir / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    long_text = "# Title\n\n" + "\n".join(f"## H{i}\n\nbody\n" for i in range(60))
    (vendored / "readme.md").write_text(long_text, encoding="utf-8")

    files = toc.find_target_files([tmp_path])
    assert all("node_modules" not in str(p) for p in files)
